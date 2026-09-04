"""The prediction log, driven by a fake connection.

No server. The class takes a connection factory precisely so the unit suite can
hand it one that records SQL, which is what makes every branch — the commit,
the rollback, the reconnect, the refusal — reachable offline. What a fake
cannot prove is that PostgreSQL accepts the statements; that is
`tests/integration/test_real_prediction_log.py`, which runs the same class
against the server in docker-compose and skips when there is not one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from src.storage.base import PredictionLog, StorageError
from src.storage.predictions import (
    COLUMNS,
    TABLE,
    NullPredictionLog,
    PostgresPredictionLog,
    open_prediction_log,
    psycopg_connect,
)

ROW = {
    "predicted_at": datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
    "match_id": "ENG_1:2024-25:00123",
    "competition_id": "ENG_1",
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "match_date": "2025-03-01",
    "model": "ensemble-calibrated",
    "model_version": "0.11.0",
    "prob_home": 0.48,
    "prob_draw": 0.26,
    "prob_away": 0.26,
    "in_sample": False,
}


class FakeCursor:
    """Records what it was asked to run, and answers with what it was given."""

    def __init__(self, owner: FakeConnection) -> None:
        self.owner = owner
        self.closed = False

    def execute(self, query: str, params: Any = None, /) -> None:
        if self.owner.fail_on is not None and self.owner.fail_on in query:
            raise RuntimeError("the server said no")
        self.owner.statements.append((query, params))

    def executemany(self, query: str, params: Any, /) -> None:
        if self.owner.fail_on is not None and self.owner.fail_on in query:
            raise RuntimeError("the server said no")
        self.owner.statements.append((query, params))

    def fetchall(self) -> list[Any]:
        return self.owner.rows

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, rows: list[Any] | None = None, fail_on: str | None = None) -> None:
        self.statements: list[tuple[str, Any]] = []
        self.rows = rows or []
        self.fail_on = fail_on
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.cursors: list[FakeCursor] = []

    def cursor(self) -> FakeCursor:
        made = FakeCursor(self)
        self.cursors.append(made)
        return made

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


def _log(connection: FakeConnection) -> PostgresPredictionLog:
    return PostgresPredictionLog(lambda: connection)


# ---- the no-op log -----------------------------------------------------------


def test_the_null_log_satisfies_the_protocol_and_stores_nothing() -> None:
    """It stands in for a real log at every call site, so it has to be
    substitutable rather than merely similar."""
    log = NullPredictionLog()
    assert isinstance(log, PredictionLog)
    assert log.enabled is False
    log.ensure_schema()
    assert log.record([ROW]) == 0
    assert list(log.recent().columns) == list(COLUMNS)
    assert log.recent().empty
    log.close()


# ---- schema ------------------------------------------------------------------


def test_ensure_schema_creates_the_table_and_the_index() -> None:
    connection = FakeConnection()
    _log(connection).ensure_schema()
    ran = " ".join(query for query, _ in connection.statements)
    assert f"CREATE TABLE IF NOT EXISTS {TABLE}" in ran
    assert f"CREATE INDEX IF NOT EXISTS {TABLE}_match_idx" in ran
    assert connection.commits == 2


def test_the_postgres_log_satisfies_the_protocol() -> None:
    assert isinstance(_log(FakeConnection()), PredictionLog)
    assert PostgresPredictionLog(lambda: FakeConnection()).enabled is True


# ---- writing -----------------------------------------------------------------


def test_a_recorded_prediction_binds_every_value_in_column_order() -> None:
    """The one failure this class can have that still looks like success: a row
    written with its probabilities in the wrong columns."""
    connection = FakeConnection()
    log = _log(connection)
    assert log.record([ROW]) == 1

    query, params = connection.statements[-1]
    assert query.startswith(f"INSERT INTO {TABLE} ({', '.join(COLUMNS)})")
    assert query.count("%s") == len(COLUMNS)
    assert params == [[ROW[column] for column in COLUMNS]]
    assert connection.commits == 1


def test_recording_nothing_touches_no_connection() -> None:
    """An empty batch is not an error and should not open a socket for it."""
    connection = FakeConnection()
    assert _log(connection).record([]) == 0
    assert connection.statements == []


def test_a_missing_field_is_stored_as_null_rather_than_shifting_the_row() -> None:
    """A record short of a column binds ``None`` there. The alternative — a
    shorter parameter list — moves every value after it one column left."""
    connection = FakeConnection()
    _log(connection).record([{**ROW, "model_version": None}])
    _, params = connection.statements[-1]
    assert params[0][COLUMNS.index("model_version")] is None
    assert params[0][COLUMNS.index("prob_home")] == ROW["prob_home"]


def test_a_failed_insert_rolls_back_and_reports_how_many_were_lost() -> None:
    connection = FakeConnection(fail_on="INSERT")
    with pytest.raises(StorageError, match="could not record 2 prediction"):
        _log(connection).record([ROW, ROW])
    assert connection.rollbacks == 1
    assert connection.commits == 0
    assert connection.cursors[-1].closed


# ---- reading -----------------------------------------------------------------


def test_recent_returns_a_frame_with_the_stored_columns() -> None:
    values = [ROW[column] for column in COLUMNS]
    connection = FakeConnection(rows=[values, values])
    frame = _log(connection).recent(limit=5)

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == list(COLUMNS)
    assert len(frame) == 2
    assert frame.iloc[0]["match_id"] == ROW["match_id"]

    query, params = connection.statements[-1]
    assert "ORDER BY predicted_at DESC" in query
    assert params == [5]


def test_a_failed_query_rolls_back_and_names_the_log() -> None:
    connection = FakeConnection(fail_on="SELECT")
    with pytest.raises(StorageError, match="prediction log query failed"):
        _log(connection).recent()
    assert connection.rollbacks == 1


# ---- connection lifecycle ----------------------------------------------------


def test_the_connection_is_opened_once_and_reused() -> None:
    connection = FakeConnection()
    opened = 0

    def connect() -> FakeConnection:
        nonlocal opened
        opened += 1
        return connection

    log = PostgresPredictionLog(connect)
    log.ensure_schema()
    log.record([ROW])
    assert opened == 1


def test_a_refused_connection_is_reported_as_a_storage_error() -> None:
    """The driver's own exception names the driver. The caller here is an HTTP
    handler that has to pick a status code."""

    def connect() -> FakeConnection:
        raise OSError("connection refused")

    with pytest.raises(StorageError, match="could not connect to the prediction log"):
        PostgresPredictionLog(connect).ensure_schema()


def test_close_is_idempotent_and_a_later_call_reconnects() -> None:
    connections = [FakeConnection(), FakeConnection()]
    log = PostgresPredictionLog(lambda: connections[min(len(connections) - 1, 0)])

    log.ensure_schema()
    log.close()
    assert connections[0].closed
    log.close()  # a second close must not raise on a connection already gone
    log.record([ROW])
    assert connections[0].statements[-1][0].startswith("INSERT")


# ---- the factory -------------------------------------------------------------


def test_no_dsn_gives_the_null_log() -> None:
    assert isinstance(open_prediction_log(None), NullPredictionLog)
    assert isinstance(open_prediction_log(""), NullPredictionLog)


def test_a_dsn_gives_a_postgres_log_with_its_schema_already_created() -> None:
    """The schema is created at startup, not on the first write: a service that
    discovered a missing table on the first request has already answered it."""
    connection = FakeConnection()
    log = open_prediction_log("postgresql://ignored", connect=lambda: connection)
    assert isinstance(log, PostgresPredictionLog)
    assert any("CREATE TABLE" in query for query, _ in connection.statements)


def test_the_psycopg_factory_is_built_without_connecting() -> None:
    """Building the factory must not open a socket. The DSN is only used when
    the first statement runs, which is what lets the app start before the
    database does."""
    assert callable(psycopg_connect("postgresql://nobody@127.0.0.1:1/none"))


def test_the_psycopg_factory_passes_the_dsn_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``psycopg.connect`` is patched rather than called: the unit suite runs
    offline, and what is worth asserting is that the DSN reaches the driver
    unmodified — a factory that rewrote it would connect somewhere else."""
    import psycopg

    seen: list[str] = []

    def fake_connect(dsn: str) -> FakeConnection:
        seen.append(dsn)
        return FakeConnection()

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    connection = psycopg_connect("postgresql://user@host:5432/db")()

    assert seen == ["postgresql://user@host:5432/db"]
    assert isinstance(connection, FakeConnection)
