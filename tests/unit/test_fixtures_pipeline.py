"""Design rows for matches that have not been played.

The claim under test is the holdout: **blank a played match's scoreline, hand
it back as a fixture, and the design row that comes out is the one the batch
build already wrote for it.** That is the whole correctness argument for
Milestone 20 — the service prices a fixture with the same thirty columns the
backtest was scored on, because the same producers built them — and it is
checked rather than reasoned about here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipelines.features import build_features
from src.pipelines.fixtures import design_rows, run_upcoming
from src.pipelines.ratings import build_ratings
from src.pipelines.serving import SERVED_COLUMNS, load_index
from src.pipelines.tables import TablePaths, read_upcoming
from tests.factories import league_frame

UNPLAYED: tuple[str, ...] = (
    "home_goals",
    "away_goals",
    "result",
    "ht_home_goals",
    "ht_away_goals",
    "ht_result",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellows",
    "away_yellows",
    "home_reds",
    "away_reds",
    "referee",
    "odds_home",
    "odds_draw",
    "odds_away",
)
"""Every column a match has and a fixture does not.

Blanking all of them is what makes the holdout honest: a stripped row that kept
its shot counts would be a fixture carrying the ninety minutes it is supposed
to be predicting.
"""


@pytest.fixture(scope="module")
def league() -> pd.DataFrame:
    return league_frame()


@pytest.fixture(scope="module")
def holdout(league: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The table minus its last matchday, and that matchday as fixtures."""
    cut = league["date"].max()
    fixtures = league[league["date"] == cut].copy()
    for column in UNPLAYED:
        fixtures[column] = pd.NA
    history = league[league["date"] < cut].reset_index(drop=True)
    # Retyped to the table it came from: assigning `pd.NA` down a column turns
    # it to `object`, and a fixture frame the real feed produces is typed to
    # the canonical schema.
    fixtures = fixtures.astype({column: history[column].dtype for column in history.columns})
    return history, fixtures.reset_index(drop=True)


class TestTheHoldout:
    def test_a_fixture_gets_the_design_row_the_batch_build_gives_the_match(
        self, league: pd.DataFrame, holdout: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        history, fixtures = holdout
        built = design_rows(history, fixtures).set_index("match_id").sort_index()

        # What `make features` and `make ratings` write, over the whole table
        # with the results still on it.
        whole = league.merge(build_ratings(league), on="match_id").merge(
            build_features(league), on="match_id"
        )
        expected = whole.set_index("match_id").loc[built.index, list(built.columns)]

        design = [column for column in built.columns if column not in fixtures.columns]
        pd.testing.assert_frame_equal(
            built[design].astype("float64"),
            expected[design].astype("float64"),
            check_names=False,
        )

    def test_appending_a_fixture_changes_no_historical_row(
        self, holdout: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """The causality argument, as an assertion.

        Every producer here reads matches strictly before the row it emits, so
        rows dated after the whole table cannot reach back into it. If that
        ever stops being true, the fixture build starts silently rewriting the
        history the backtest was scored on.
        """
        history, fixtures = holdout
        alone = build_features(history).set_index("match_id")
        with_fixtures = build_features(pd.concat([history, fixtures], ignore_index=True)).set_index(
            "match_id"
        )
        pd.testing.assert_frame_equal(alone, with_fixtures.loc[alone.index])


class TestDesignRows:
    def test_nothing_to_price_gives_an_empty_frame_with_the_served_columns(
        self, league: pd.DataFrame
    ) -> None:
        rows = design_rows(league, league.iloc[:0])
        assert rows.empty
        assert list(rows.columns) == list(SERVED_COLUMNS)

    def test_only_the_fixtures_come_back(self, holdout: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        history, fixtures = holdout
        rows = design_rows(history, fixtures)
        assert set(rows["match_id"]) == set(fixtures["match_id"])

    def test_a_producer_that_stops_filling_a_column_is_reported(
        self, holdout: tuple[pd.DataFrame, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        history, fixtures = holdout
        monkeypatch.setattr(
            "src.pipelines.fixtures.build_features",
            lambda frame, _builders: pd.DataFrame({"match_id": frame["match_id"]}),
        )
        with pytest.raises(ValueError, match="produced no"):
            design_rows(history, fixtures)


class TestRunUpcoming:
    def test_it_writes_the_table_and_reports_what_is_in_it(
        self, holdout: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
    ) -> None:
        history, fixtures = holdout
        report = run_upcoming(history, fixtures, tmp_path)

        assert report.fixtures == len(fixtures)
        assert report.output is not None and report.output.is_file()
        assert report.summary().startswith(f"{len(fixtures)} fixture(s)")
        written = read_upcoming(report.output)
        assert written is not None and len(written) == len(fixtures)

    def test_a_fixture_the_table_already_has_is_dropped(
        self, league: pd.DataFrame, holdout: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
    ) -> None:
        """The published file keeps a match in it for a while after it is played.

        Two rows with one ``match_id`` would give the index a duplicate key, so
        the played row wins and the fixture is counted rather than kept.
        """
        _, fixtures = holdout
        report = run_upcoming(league, fixtures, tmp_path)
        assert report.fixtures == 0
        assert report.already_played == len(fixtures)

    def test_an_empty_week_still_writes_the_table(
        self, league: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Leaving yesterday's file would leave the service offering to price
        matches that have since kicked off."""
        report = run_upcoming(league, league.iloc[:0], tmp_path)
        assert report.summary() == "no fixtures"
        assert report.output is not None and report.output.is_file()


class TestTheServiceIndexesThem:
    def test_a_fixture_becomes_priceable(
        self, holdout: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
    ) -> None:
        history, fixtures = holdout
        _write_tables(history, tmp_path)
        run_upcoming(history, fixtures, tmp_path)

        index = load_index(_paths(tmp_path))
        assert index is not None
        one = fixtures.iloc[0]
        assert index.resolve(match_id=str(one["match_id"])) is not None

    def test_no_fixture_table_indexes_exactly_what_it_always_did(
        self, holdout: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
    ) -> None:
        history, _ = holdout
        _write_tables(history, tmp_path)
        index = load_index(_paths(tmp_path))
        assert index is not None and len(index) == len(history)

    def test_a_fixture_table_from_a_different_design_is_refused(
        self, holdout: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Loudly, rather than quietly indexing the played matches alone.

        Silence here would look exactly like a week with no football in it,
        which is the one failure mode nobody would go looking for.
        """
        from src.pipelines.serving import ServingError

        history, fixtures = holdout
        _write_tables(history, tmp_path)
        run_upcoming(history, fixtures, tmp_path)
        path = tmp_path / "upcoming.parquet"
        narrowed = read_upcoming(path)
        assert narrowed is not None
        narrowed.drop(columns=["home_form_points_5"]).to_parquet(path, index=False)

        with pytest.raises(ServingError, match="upcoming table is missing"):
            load_index(_paths(tmp_path))


def _write_tables(matches: pd.DataFrame, directory: Path) -> None:
    matches.to_parquet(directory / "matches.parquet", index=False)
    build_ratings(matches).to_parquet(directory / "ratings.parquet", index=False)
    build_features(matches).to_parquet(directory / "features.parquet", index=False)


def _paths(directory: Path) -> TablePaths:
    return TablePaths(
        matches=directory / "matches.parquet",
        ratings=directory / "ratings.parquet",
        features=directory / "features.parquet",
        upcoming=directory / "upcoming.parquet",
    )
