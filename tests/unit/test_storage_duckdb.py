"""The analytical store.

Two properties carry most of the value. A store read must equal a
``pd.read_parquet`` of the same file — otherwise "read through the store" is a
behaviour change rather than an indirection — and a filtered read must have the
same dtypes as an unfiltered one, which is the trap DuckDB sets by returning a
NumPy integer for a column with no nulls and a nullable one for a column with
them.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.base import CANONICAL_COLUMNS, CANONICAL_SCHEMA
from src.storage.base import MatchStore, StorageError
from src.storage.duckdb_store import (
    MATCHES_VIEW,
    RATINGS_VIEW,
    DuckDBStore,
    _as_date_string,
    _quote_literal,
)
from tests.factories import league_frame


@pytest.fixture(scope="module")
def matches_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("processed") / "matches.parquet"
    league_frame().to_parquet(path, index=False)
    return path


@pytest.fixture
def store(matches_parquet: Path) -> DuckDBStore:
    with DuckDBStore.open_matches(matches_parquet) as opened:
        yield opened


def test_a_store_satisfies_the_protocol(store: DuckDBStore) -> None:
    assert isinstance(store, MatchStore)


def test_the_matches_view_is_attached(store: DuckDBStore) -> None:
    assert store.views() == (MATCHES_VIEW,)


def test_count_matches_the_file(store: DuckDBStore, matches_parquet: Path) -> None:
    assert store.count() == len(pd.read_parquet(matches_parquet))


def test_a_store_read_equals_a_pandas_read(store: DuckDBStore, matches_parquet: Path) -> None:
    """The property that makes the store an indirection rather than a change."""
    assert store.read_matches().equals(pd.read_parquet(matches_parquet))


def test_the_canonical_dtypes_survive(store: DuckDBStore) -> None:
    frame = store.read_matches()
    assert {column: str(dtype) for column, dtype in frame.dtypes.items()} == CANONICAL_SCHEMA


def test_a_filtered_read_has_the_same_dtypes_as_a_full_one(store: DuckDBStore) -> None:
    """DuckDB hands back a NumPy int16 for a column with no nulls and a nullable
    Int16 for one with them — so without re-asserting the schema, the dtype of
    home_goals would depend on which rows a query happened to select, and code
    that worked on the full table would break on a subset."""
    full = store.read_matches().dtypes
    filtered = store.read_matches(since="2020-08-01").dtypes
    assert full.equals(filtered)


def test_a_date_window_is_inclusive_at_both_ends(store: DuckDBStore) -> None:
    frame = store.read_matches(since="2015-08-01", until="2016-05-01")
    assert frame["date"].min() >= pd.Timestamp("2015-08-01")
    assert frame["date"].max() <= pd.Timestamp("2016-05-01")
    assert not frame.empty


def test_a_date_bound_accepts_a_date_object(store: DuckDBStore) -> None:
    by_object = store.read_matches(since=date(2015, 8, 1))
    by_string = store.read_matches(since="2015-08-01")
    assert by_object.equals(by_string)


def test_a_competition_filter_selects_only_that_competition(store: DuckDBStore) -> None:
    assert set(store.read_matches(competitions=["ENG_1"])["competition_id"]) == {"ENG_1"}
    assert store.read_matches(competitions=["NOWHERE_1"]).empty


def test_an_empty_competition_list_returns_nothing(store: DuckDBStore) -> None:
    """Different from `None`. A caller that filtered a registry down to nothing
    should get nothing back, not everything — and an empty SQL `IN ()` is a
    syntax error, so the case has to be handled rather than passed through."""
    frame = store.read_matches(competitions=[])
    assert frame.empty
    assert list(frame.columns) == list(CANONICAL_COLUMNS)


def test_a_projection_keeps_canonical_order(store: DuckDBStore) -> None:
    """Two reads of the same columns must compare equal, which they cannot if
    the order depends on how the argument was spelled."""
    frame = store.read_matches(columns=["result", "date", "home_team_id"])
    assert list(frame.columns) == ["date", "home_team_id", "result"]


def test_a_projection_of_pre_match_columns_is_expressible(store: DuckDBStore) -> None:
    """The read a feature builder makes: everything knowable before kick-off,
    and nothing else."""
    frame = store.read_matches(columns=["date", "home_team_id", "away_team_id", "season"])
    assert "result" not in frame.columns
    assert "home_shots" not in frame.columns


def test_an_unknown_column_is_rejected(store: DuckDBStore) -> None:
    with pytest.raises(ValueError, match="not canonical columns"):
        store.read_matches(columns=["expected_goals"])


def test_results_are_ordered_by_date(store: DuckDBStore) -> None:
    """A store is not entitled to assume its input was sorted, and every
    temporal operation downstream depends on it."""
    assert store.read_matches()["date"].is_monotonic_increasing


def test_an_invalid_view_name_is_rejected(store: DuckDBStore, matches_parquet: Path) -> None:
    """A view name reaches the engine as text because SQL identifiers cannot be
    bound as parameters, so its shape is the only thing preventing injection."""
    with pytest.raises(ValueError, match="not a usable view name"):
        store.attach_parquet("matches; DROP TABLE matches", matches_parquet)


def test_attaching_a_missing_file_fails_immediately(store: DuckDBStore, tmp_path: Path) -> None:
    """DuckDB would accept the CREATE VIEW and fail on the first query instead,
    reporting the error a long way from its cause."""
    with pytest.raises(StorageError, match="no such parquet file"):
        store.attach_parquet("other", tmp_path / "absent.parquet")


def test_a_failed_open_leaves_nothing_behind(tmp_path: Path) -> None:
    """On a file-backed catalog a leaked connection is a leaked lock, and the
    next open fails for a reason that has nothing to do with the real problem."""
    catalog = tmp_path / "store.duckdb"
    with pytest.raises(StorageError):
        DuckDBStore.open_matches(tmp_path / "absent.parquet", database=catalog)
    with DuckDBStore(catalog) as reopened:
        assert reopened.views() == ()


def test_ratings_can_be_attached_beside_the_matches(matches_parquet: Path, tmp_path: Path) -> None:
    """Attached rather than joined in: the two tables are rebuilt on different
    cadences, and a consumer that wants both writes the join it needs."""
    ratings = tmp_path / "ratings.parquet"
    frame = pd.read_parquet(matches_parquet)[["match_id"]].copy()
    frame["elo_home"] = 1500.0
    frame.to_parquet(ratings, index=False)

    with DuckDBStore.open_matches(matches_parquet, ratings=ratings) as store:
        assert set(store.views()) == {MATCHES_VIEW, RATINGS_VIEW}
        joined = store.query(
            f"SELECT count(*) AS n FROM {MATCHES_VIEW} JOIN {RATINGS_VIEW} USING (match_id)"
        )
        assert int(joined.iloc[0]["n"]) == len(frame)


def test_a_half_attachable_open_attaches_nothing(matches_parquet: Path, tmp_path: Path) -> None:
    """The matches file exists and the ratings file does not.

    Attaching as it went created the matches view, failed on the ratings, and
    left a file-backed catalog holding half of what was asked for — which the
    next open would have reported as success.
    """
    catalog = tmp_path / "store.duckdb"
    with pytest.raises(StorageError):
        DuckDBStore.open_matches(
            matches_parquet, ratings=tmp_path / "absent.parquet", database=catalog
        )
    with DuckDBStore(catalog) as reopened:
        assert reopened.views() == ()


def test_a_bad_query_raises_storage_error(store: DuckDBStore) -> None:
    with pytest.raises(StorageError, match="query failed"):
        store.query("SELECT * FROM no_such_table")


def test_query_binds_its_parameters(store: DuckDBStore) -> None:
    result = store.query(
        f"SELECT count(*) AS n FROM {MATCHES_VIEW} WHERE competition_id = ?", ["ENG_1"]
    )
    assert int(result.iloc[0]["n"]) == store.count()


def test_a_file_backed_catalog_keeps_its_views(tmp_path: Path, matches_parquet: Path) -> None:
    catalog = tmp_path / "store.duckdb"
    with DuckDBStore.open_matches(matches_parquet, database=catalog) as first:
        expected = first.count()
    with DuckDBStore(catalog, read_only=True) as second:
        assert second.count() == expected


def test_a_path_containing_a_quote_is_escaped(tmp_path: Path) -> None:
    """Paths are the one thing interpolated into SQL here, because a bound
    parameter is not stored in a view definition."""
    awkward = tmp_path / "O'Brien"
    awkward.mkdir()
    path = awkward / "matches.parquet"
    league_frame(seasons=["2024-25"]).to_parquet(path, index=False)
    with DuckDBStore.open_matches(path) as store:
        assert store.count() == 380


def test_quote_literal_doubles_single_quotes() -> None:
    assert _quote_literal("a'b") == "'a''b'"


def test_a_date_bound_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="not a date"):
        _as_date_string("last Tuesday")


def test_the_connection_is_reachable_for_queries_this_class_does_not_wrap(
    store: DuckDBStore,
) -> None:
    assert store.connection.execute("SELECT 1").fetchone() == (1,)


def test_closing_twice_is_harmless(matches_parquet: Path) -> None:
    store = DuckDBStore.open_matches(matches_parquet)
    store.close()
    store.close()
