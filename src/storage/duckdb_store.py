"""The analytical store: DuckDB views over Parquet.

DuckDB reads Parquet in place. There is no load step, no second copy of the
data and no server, so the "database" is a few kilobytes of catalog next to the
files the ingest pipeline already wrote. That is the whole reason it was chosen
over loading 305k rows into PostgreSQL to run aggregate queries against them:
the analytical questions this project asks — coverage per competition, class
balance per season, how many matches carry shot data — are column scans, and a
columnar engine answers them from the same bytes the modelling code reads.

**Views, not tables.** ``attach_parquet`` creates a view, so the file on disk
stays the single source of truth and a re-ingest is visible to the next query
without a refresh step. A ``CREATE TABLE AS`` would have duplicated the data
into the catalog and then quietly served yesterday's copy.

**The dtype trap.** DuckDB hands pandas a NumPy ``int16`` for a column with no
nulls and a nullable ``Int16`` for one with them — so the dtype of
``home_goals`` would depend on which rows a query happened to select, and code
that worked on the full table would break on a filtered subset. Every read here
re-asserts :data:`~src.ingestion.base.CANONICAL_SCHEMA` on the way out, which
is also what makes a store read and a ``pd.read_parquet`` of the same file
compare equal.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pandas as pd

from src.ingestion.base import CANONICAL_COLUMNS, CANONICAL_SCHEMA
from src.storage.base import StorageError
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

MATCHES_VIEW = "matches"
"""The view name every downstream query uses. Fixed rather than configurable:
a table name that varies by deployment is a table name that appears in a
handwritten query somewhere and is wrong half the time."""

RATINGS_VIEW = "ratings"
"""The derived ratings table, when one has been built. Attached beside the
matches rather than joined into them: they are rebuilt on a different cadence,
and a consumer that wants both writes the join it needs."""

IN_MEMORY = ":memory:"

# SQL identifiers cannot be bound as parameters, so a view name reaches the
# engine as text. Restricting it to this shape is what keeps that from being an
# injection point — and it is also just what a sane table name looks like.
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def _quote_literal(value: str) -> str:
    """Return ``value`` as a SQL string literal.

    Used only for file paths, which cannot be parameterised inside a
    ``CREATE VIEW`` — a bound parameter is not stored in the view definition,
    so the view would fail on the next query with a missing-parameter error.
    Doubling single quotes is the SQL-standard escape and DuckDB follows it.
    """
    return "'" + value.replace("'", "''") + "'"


def _as_date_string(value: date | str) -> str:
    """Normalise a bound to ``YYYY-MM-DD``.

    Accepting both a ``date`` and a string keeps callers from converting at
    every call site; rejecting anything else keeps a stray ``datetime`` with a
    timezone from silently shifting a boundary match into the wrong side of a
    temporal split.
    """
    if isinstance(value, date):
        return value.isoformat()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"not a date: {value!r}")
    return str(parsed.date())


class DuckDBStore:
    """A DuckDB connection with the canonical match table attached.

    Satisfies :class:`~src.storage.base.MatchStore`. Usable as a context
    manager; the connection is closed on exit even if a query raised, which
    matters for a file-backed catalog because DuckDB holds a lock on it.
    """

    def __init__(self, database: Path | str = IN_MEMORY, *, read_only: bool = False) -> None:
        """Open ``database``, creating it if it does not exist.

        Args:
            database: Catalog path, or ``":memory:"`` for an ephemeral store.
                In-memory is the sensible default: the catalog holds only view
                definitions, and recreating it costs one statement.
            read_only: Open without taking a write lock. DuckDB allows several
                read-only readers of one file but only a single writer, so a
                dashboard and a training run can share a catalog this way.
        """
        self._database = str(database)
        try:
            self._connection = duckdb.connect(self._database, read_only=read_only)
        except duckdb.Error as error:  # pragma: no cover - depends on the filesystem
            raise StorageError(f"could not open DuckDB at {self._database}: {error}") from error

    # ---- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> DuckDBStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """The underlying connection, for the rare query this class does not wrap."""
        return self._connection

    # ---- wiring ------------------------------------------------------------

    def attach_parquet(self, name: str, path: Path) -> None:
        """Expose ``path`` as a view called ``name``.

        Raises:
            ValueError: If ``name`` is not a plain lowercase identifier.
            StorageError: If the file does not exist. DuckDB would otherwise
                accept the ``CREATE VIEW`` and fail on the first *query*, which
                reports the error a long way from its cause.
        """
        if not _IDENTIFIER.match(name):
            raise ValueError(f"not a usable view name: {name!r}")
        if not path.is_file():
            raise StorageError(f"no such parquet file: {path}")
        self._connection.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM read_parquet({_quote_literal(str(path))})"
        )
        logger.debug("attached %s -> %s", name, path)

    @classmethod
    def open_matches(
        cls,
        path: Path,
        *,
        ratings: Path | None = None,
        database: Path | str = IN_MEMORY,
    ) -> DuckDBStore:
        """Open a store with ``path`` attached as :data:`MATCHES_VIEW`.

        Args:
            path: The canonical match table.
            ratings: The ratings table, attached as :data:`RATINGS_VIEW` when
                given. Optional because ratings are built separately and a
                fresh checkout has none — a store that refused to open without
                them would make the ingest untestable.
            database: Catalog path, or in-memory.

        Raises:
            StorageError: If any source file is missing. Every path is checked
                *before* the catalog is opened, so a failed call leaves nothing
                behind. Checking as we went created the matches view, then
                failed on the ratings, and left a file-backed catalog holding
                half of what was asked for — which the next open would report
                as success.
        """
        sources = [path] if ratings is None else [path, ratings]
        for source in sources:
            if not source.is_file():
                raise StorageError(f"no such parquet file: {source}")

        store = cls(database)
        try:
            store.attach_parquet(MATCHES_VIEW, path)
            if ratings is not None:
                store.attach_parquet(RATINGS_VIEW, ratings)
        except Exception:  # pragma: no cover - the paths are checked above
            # Belt and braces: a leaked connection is a leaked lock on a
            # file-backed catalog, and DuckDB can still refuse a view for
            # reasons a path check cannot see.
            store.close()
            raise
        return store

    def views(self) -> tuple[str, ...]:
        """Every view and table currently visible, sorted."""
        rows = self._connection.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    # ---- reading -----------------------------------------------------------

    def query(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        """Run ``sql`` and return the result as a frame.

        Parameters are bound, never interpolated. Dtypes are whatever DuckDB
        infers — this is the escape hatch for aggregates, where the canonical
        schema does not apply. Use :meth:`read_matches` for match rows.
        """
        try:
            return self._connection.execute(sql, list(params or ())).df()
        except duckdb.Error as error:
            raise StorageError(f"query failed: {error}") from error

    def count(self) -> int:
        """Number of matches, computed in the engine."""
        result = self.query(f"SELECT count(*) AS n FROM {MATCHES_VIEW}")
        return int(result.iloc[0]["n"])

    def read_matches(
        self,
        *,
        competitions: Sequence[str] | None = None,
        since: date | str | None = None,
        until: date | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return matches, ordered by date, with the canonical dtypes restored.

        Filters are pushed into the query rather than applied to a materialised
        frame, so a point-in-time read of one competition does not first pull
        three hundred thousand rows into memory.

        Args:
            competitions: Restrict to these ``competition_id`` values. An empty
                sequence means "no competitions" and returns no rows, which is
                different from ``None`` meaning "no filter" — a caller that
                filtered a registry down to nothing should get nothing back,
                not everything.
            since: Inclusive lower bound on ``date``.
            until: Inclusive upper bound on ``date``.
            columns: Projection. Must be canonical column names.

        Raises:
            ValueError: On an unknown column name.
            StorageError: If no matches view is attached.
        """
        selected = self._resolve_columns(columns)
        projection = ", ".join(f'"{column}"' for column in selected)

        clauses: list[str] = []
        params: list[Any] = []
        if competitions is not None:
            # An empty IN list is a syntax error in DuckDB, so the no-rows case
            # is expressed as a false predicate rather than as an empty list.
            if len(competitions) == 0:
                clauses.append("FALSE")
            else:
                placeholders = ", ".join("?" for _ in competitions)
                clauses.append(f"competition_id IN ({placeholders})")
                params.extend(competitions)
        if since is not None:
            clauses.append("date >= ?")
            params.append(_as_date_string(since))
        if until is not None:
            clauses.append("date <= ?")
            params.append(_as_date_string(until))

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        # Ordered here, not left to the file's order: the canonical table is
        # written sorted, but a store is not entitled to assume its input was,
        # and every temporal operation downstream depends on it.
        sql = f"SELECT {projection} FROM {MATCHES_VIEW}{where} ORDER BY date, competition_id, match_id"

        frame = self.query(sql, params)
        return frame.astype({column: CANONICAL_SCHEMA[column] for column in selected})

    def read_ratings(self) -> pd.DataFrame:
        """Return the ratings table, keyed by ``match_id``.

        Unfiltered, and deliberately so: the ratings are one narrow row per
        match and the caller joins them to whatever slice of the canonical
        table it already holds. A second filter API here would be a second
        place for a point-in-time read to be subtly different.

        Raises:
            StorageError: If no ratings view is attached. That is the
                clean-checkout case, and the message has to say which file is
                missing rather than report an unknown table.
        """
        if RATINGS_VIEW not in self.views():
            raise StorageError(
                f"no {RATINGS_VIEW} view attached; open the store with "
                "`ratings=` or run scripts/build_ratings.py first"
            )
        return self.query(f"SELECT * FROM {RATINGS_VIEW}")

    def _resolve_columns(self, columns: Sequence[str] | None) -> tuple[str, ...]:
        """Validate a projection, preserving canonical order.

        Canonical order rather than the caller's: two reads of the same columns
        should produce frames that compare equal, and re-ordering on request
        makes that depend on how the argument was spelled.
        """
        if columns is None:
            return CANONICAL_COLUMNS
        unknown = sorted(set(columns) - set(CANONICAL_COLUMNS))
        if unknown:
            raise ValueError(f"not canonical columns: {unknown}")
        requested = set(columns)
        return tuple(column for column in CANONICAL_COLUMNS if column in requested)
