"""The prediction log: PostgreSQL, and the no-op that stands in for it.

This is the first application state in the project. Everything else on disk is
*derived* — delete it and `make reproduce` rebuilds it byte for byte — and a
served prediction is the opposite: it happened once, at a time, from inputs
that will not be the same tomorrow, and nothing regenerates it. That difference
is the condition for adding a second store, and it is met here.

**Why PostgreSQL and not DuckDB.** The analytical store is right for column
scans over Parquet and wrong for this: DuckDB takes a single writer, and the
deployment shape is several uvicorn workers behind one port. A log that
serialised the workers, or lost a row when two of them wrote, would be a log
nobody could compute a calibration from.

**Why it is optional.** ``PREDICTION_LOG_DSN`` unset means
:class:`NullPredictionLog`, and the service runs without a database. A clean
checkout has no Postgres, CI has no Postgres, and an API that would not start
without one is an API that cannot be tried. What is lost when it is off is
stated in ``/health`` rather than inferred from silence.

**The connection is injected.** :class:`PostgresPredictionLog` takes a factory,
which is what lets the unit suite drive every branch of it with a fake that
records SQL, exactly as :mod:`src.utils.http` is driven by a stubbed session.
An integration test marked ``integration`` runs the same class against the real
server in ``docker-compose.yml``, so neither test is asked to prove what the
other one proves.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import pandas as pd

from src.storage.base import StorageError
from src.utils.logging import get_logger

logger = get_logger(__name__)

TABLE = "predictions"

COLUMNS: tuple[str, ...] = (
    "predicted_at",
    "match_id",
    "competition_id",
    "home_team",
    "away_team",
    "match_date",
    "model",
    "model_version",
    "prob_home",
    "prob_draw",
    "prob_away",
    "in_sample",
)
"""The stored columns, in insert order.

One tuple, used to build the ``INSERT`` column list, the placeholder list and
the ordered values of every row. Three lists that have to agree is three lists
that eventually do not, and the symptom of that is a log whose probabilities
are in the wrong columns and whose rows still look like rows.
"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id             BIGSERIAL PRIMARY KEY,
    predicted_at   TIMESTAMPTZ      NOT NULL,
    match_id       TEXT             NOT NULL,
    competition_id TEXT             NOT NULL,
    home_team      TEXT             NOT NULL,
    away_team      TEXT             NOT NULL,
    match_date     DATE             NOT NULL,
    model          TEXT             NOT NULL,
    model_version  TEXT             NOT NULL,
    prob_home      DOUBLE PRECISION NOT NULL,
    prob_draw      DOUBLE PRECISION NOT NULL,
    prob_away      DOUBLE PRECISION NOT NULL,
    in_sample      BOOLEAN          NOT NULL
)
"""
"""Deliberately not an ORM migration.

One table, created if absent, with no column ever dropped or renamed. A
migration tool earns its keep on the second schema change; declaring one for
the first is a dependency, a directory of versioned scripts and a second
description of a table that already fits on a screen.
"""

INDEX = f"CREATE INDEX IF NOT EXISTS {TABLE}_match_idx ON {TABLE} (match_id, predicted_at DESC)"
"""The one query a consumer runs: what did we say about this match, latest first."""


class Cursor(Protocol):
    """The cursor surface this module uses. Four methods, not a driver's API."""

    def execute(self, query: str, params: Sequence[Any] | None = None, /) -> Any: ...

    def executemany(self, query: str, params: Sequence[Sequence[Any]], /) -> Any: ...

    def fetchall(self) -> list[Any]: ...

    def close(self) -> None: ...


class Connection(Protocol):
    """The connection surface this module uses."""

    def cursor(self) -> Cursor: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


Connect = Callable[[], Connection]
"""How a connection is obtained. Injected, so the unit suite needs no server."""


def psycopg_connect(dsn: str) -> Connect:
    """A factory that opens ``dsn`` with psycopg.

    Imported inside the function rather than at module scope so that importing
    this module — which the API does to reach :class:`NullPredictionLog` — does
    not require the driver to be installed. The image that serves without a log
    can then leave it out.
    """

    def connect() -> Connection:
        import psycopg

        connection: Connection = psycopg.connect(dsn)
        return connection

    return connect


class NullPredictionLog:
    """The log when none is configured. Records nothing and says so once.

    Not ``None`` at the call site: a service that has to check whether it has a
    log before every write grows that check in three places and forgets it in
    the fourth. Satisfies :class:`~src.storage.base.PredictionLog`.
    """

    enabled = False

    def ensure_schema(self) -> None:
        logger.info("prediction log disabled: PREDICTION_LOG_DSN is not set")

    def record(self, predictions: Sequence[Mapping[str, Any]]) -> int:
        """Discard them, and say nothing was written.

        The parameter keeps the protocol's own name — structural typing
        matches on names, so renaming it to please a linter would stop this
        class satisfying the protocol it exists to satisfy.
        """
        logger.debug("log disabled: %d prediction(s) not recorded", len(predictions))
        return 0

    def recent(self, limit: int = 20) -> pd.DataFrame:
        """An empty frame with the real columns, so a caller reads it the same
        way whether or not a log is configured."""
        return pd.DataFrame(columns=list(COLUMNS)).head(limit)

    def close(self) -> None:
        return None


class PostgresPredictionLog:
    """Append-only prediction storage in PostgreSQL.

    Satisfies :class:`~src.storage.base.PredictionLog`. One connection, opened
    lazily on first use and reopened if it was closed, because a service that
    connected at import time would fail to start whenever the database was a
    second slower to come up than it was.
    """

    enabled = True

    def __init__(self, connect: Connect) -> None:
        self._connect = connect
        self._connection: Connection | None = None

    # ---- connection --------------------------------------------------------

    def connection(self) -> Connection:
        """The open connection, opening one if there is not one already.

        Raises:
            StorageError: If the connection cannot be opened. Wrapped rather
                than let through, because the driver's exception names the
                driver and the caller here is an HTTP handler that has to
                decide on a status code.
        """
        if self._connection is None:
            try:
                self._connection = self._connect()
            except Exception as error:  # drivers raise their own exception tree
                raise StorageError(f"could not connect to the prediction log: {error}") from error
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _run(self, query: str, params: Sequence[Any] | None = None) -> list[Any]:
        """Execute one statement, commit it, and return whatever it selected.

        Committed here rather than by the caller: every statement this class
        runs is either a schema change or a single append, and an uncommitted
        append is a prediction the log did not record while reporting that it
        had. A failure rolls back, so a half-applied batch cannot survive.
        """
        connection = self.connection()
        cursor = connection.cursor()
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall() if query.lstrip().upper().startswith("SELECT") else []
            connection.commit()
            return rows
        except Exception as error:  # drivers raise their own exception tree
            connection.rollback()
            raise StorageError(f"prediction log query failed: {error}") from error
        finally:
            cursor.close()

    # ---- the protocol ------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create the table and its index if they are absent."""
        self._run(SCHEMA)
        self._run(INDEX)
        logger.info("prediction log ready: table %s", TABLE)

    def record(self, predictions: Sequence[Mapping[str, Any]]) -> int:
        """Append every prediction in one round trip.

        Values are bound, never interpolated — the only text that reaches the
        server is this module's own SQL, and every field of a request travels
        as a parameter.
        """
        if not predictions:
            return 0
        placeholders = ", ".join("%s" for _ in COLUMNS)
        statement = f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) VALUES ({placeholders})"
        rows = [[record.get(column) for column in COLUMNS] for record in predictions]

        connection = self.connection()
        cursor = connection.cursor()
        try:
            cursor.executemany(statement, rows)
            connection.commit()
        except Exception as error:  # drivers raise their own exception tree
            connection.rollback()
            raise StorageError(f"could not record {len(rows)} prediction(s): {error}") from error
        finally:
            cursor.close()
        logger.debug("recorded %d prediction(s)", len(rows))
        return len(rows)

    def recent(self, limit: int = 20) -> pd.DataFrame:
        """The newest ``limit`` predictions.

        Column names come from :data:`COLUMNS` rather than from the cursor's
        description, so the frame's shape is a fact about this module rather
        than about which driver returned the rows.
        """
        rows = self._run(
            f"SELECT {', '.join(COLUMNS)} FROM {TABLE} "
            "ORDER BY predicted_at DESC, id DESC LIMIT %s",
            [int(limit)],
        )
        return pd.DataFrame(list(rows), columns=list(COLUMNS))


def open_prediction_log(
    dsn: str | None, *, connect: Connect | None = None
) -> NullPredictionLog | PostgresPredictionLog:
    """The configured log, or the no-op one when no DSN is set.

    The schema is created here rather than on first write. A service that
    discovered a missing table on the first request would have already answered
    it, and that row is the one that is lost.

    Args:
        dsn: PostgreSQL connection string, or ``None`` to disable the log.
        connect: Connection factory, overriding the one built from ``dsn``.
            The seam the unit suite drives, and the reason no test here needs a
            server to prove the schema is created exactly once.
    """
    if not dsn and connect is None:
        log: NullPredictionLog | PostgresPredictionLog = NullPredictionLog()
    else:
        log = PostgresPredictionLog(connect or psycopg_connect(str(dsn)))
    log.ensure_schema()
    return log
