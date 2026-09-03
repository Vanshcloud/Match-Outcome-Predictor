"""The backtest against the real table.

Skips without `python scripts/fetch_data.py` and `scripts/build_ratings.py`.
What these add over the unit suite is the only thing that matters about a
backtest: the numbers. A synthetic league can prove the bookkeeping is right;
it cannot say whether Dixon-Coles is worth building, and that is the question
Milestone 7 exists to answer.

The bands are the measured figures with room for the provider adding a season.
Wide enough not to go red on new data, narrow enough that a change of sign or
an ordering flip cannot pass.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.baselines import default_forecasters
from src.models.splits import DEFAULT_FOLDS, walk_forward
from src.pipelines.backtest import (
    COMMON,
    POOLED,
    PRICED,
    per_competition_table,
    pooled_table,
    run_backtest,
)
from src.pipelines.ingest import MATCHES_FILENAME
from src.pipelines.ratings import RATINGS_FILENAME
from src.utils.config import load_settings
from src.validation.temporal import split_boundary

pytestmark = pytest.mark.integration

SETTINGS = load_settings()
MATCHES = SETTINGS.paths.processed_dir / MATCHES_FILENAME
RATINGS = SETTINGS.paths.features_dir / RATINGS_FILENAME

# Measured over five yearly folds, 62,036 evaluation matches.
BOOKMAKER_LOG_LOSS = (0.98, 1.02)
DIXON_COLES_LOG_LOSS = (1.01, 1.05)
CLASS_PRIOR_LOG_LOSS = (1.06, 1.09)
DIXON_COLES_COVERAGE = 0.90
STRENGTH_CORRELATION = 0.8
"""How strongly the rating's edge over the prior tracks the spread of team
strength in a competition. Measured at 0.90; the floor is what a model that had
stopped being a strength model would fall through."""


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    if not MATCHES.is_file():
        pytest.skip(f"no ingested data at {MATCHES}; run scripts/fetch_data.py")
    if not RATINGS.is_file():
        pytest.skip(f"no ratings at {RATINGS}; run scripts/build_ratings.py")
    return pd.read_parquet(MATCHES).merge(pd.read_parquet(RATINGS), on="match_id", how="left")


@pytest.fixture(scope="module")
def scores(matches: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    report = run_backtest(matches, tmp_path_factory.mktemp("backtest"))
    assert report.scores is not None
    return report.scores


def test_the_history_supports_five_yearly_folds(matches: pd.DataFrame) -> None:
    assert len(list(walk_forward(matches))) == DEFAULT_FOLDS


def test_every_real_fold_passes_the_boundary_probe(matches: pd.DataFrame) -> None:
    """Thirty-nine competitions, three calendars, and matches on the same day
    in different countries — the case a synthetic league cannot produce."""
    for fold in walk_forward(matches):
        assert split_boundary(fold.train, fold.evaluate, name=f"fold {fold.index}").ok


def test_all_four_baselines_can_be_scored(matches: pd.DataFrame) -> None:
    assert len(default_forecasters(matches)) == 4


def test_the_bookmaker_wins(scores: pd.DataFrame) -> None:
    """The ceiling this project states up front. A model that beat the closing
    line on this data would be evidence of a leak, not of skill."""
    table = pooled_table(scores).set_index("forecaster")
    assert table.loc["bookmaker", "log_loss"] < table.loc["dixon_coles", "log_loss"]
    assert table.loc["bookmaker", "rps"] < table.loc["dixon_coles", "rps"]


def test_dixon_coles_beats_the_prior_it_has_to_beat(scores: pd.DataFrame) -> None:
    table = pooled_table(scores).set_index("forecaster")
    assert table.loc["dixon_coles", "log_loss"] < table.loc["class_prior", "log_loss"]


def test_the_measured_figures_are_where_they_were_measured(scores: pd.DataFrame) -> None:
    table = pooled_table(scores).set_index("forecaster")
    for name, (low, high) in (
        ("bookmaker", BOOKMAKER_LOG_LOSS),
        ("dixon_coles", DIXON_COLES_LOG_LOSS),
        ("class_prior", CLASS_PRIOR_LOG_LOSS),
    ):
        assert low <= table.loc[name, "log_loss"] <= high, name


def test_home_always_is_infinitely_bad_and_still_rankable(scores: pd.DataFrame) -> None:
    """Log loss sends it to infinity; RPS still says how wrong it was."""
    table = pooled_table(scores).set_index("forecaster")
    assert table.loc["home_always", "log_loss"] == float("inf")
    assert 0.4 < table.loc["home_always", "rps"] < 0.5


def test_the_two_subsets_are_not_the_same_matches(scores: pd.DataFrame) -> None:
    """The reason there are two tables at all."""
    pooled = scores[scores["competition_id"] == POOLED]
    priced = pooled[(pooled["subset"] == PRICED) & (pooled["forecaster"] == "class_prior")]
    common = pooled[(pooled["subset"] == COMMON) & (pooled["forecaster"] == "class_prior")]
    assert int(common["n"].sum()) < int(priced["n"].sum())


def test_dixon_coles_prices_most_of_the_recent_history(scores: pd.DataFrame) -> None:
    """It cannot price a competition's first seasons, and these folds are recent."""
    pooled = scores[(scores["competition_id"] == POOLED) & (scores["subset"] == PRICED)]
    counts = pooled.groupby("forecaster")["n"].sum()
    assert counts["dixon_coles"] / counts["class_prior"] > DIXON_COLES_COVERAGE


def test_the_rating_beats_the_prior_in_all_but_one_competition(scores: pd.DataFrame) -> None:
    """ARG_CUP is the exception, and it is a cup: teams from different tiers
    meet once, and a model fitted per competition has the least to say exactly
    where the strengths are most spread out."""
    table = per_competition_table(scores)
    worse = table[table["dixon_coles"] >= table["class_prior"]]
    assert list(worse["competition_id"]) == ["ARG_CUP"]


def test_the_ratings_edge_is_the_spread_of_team_strength(
    scores: pd.DataFrame, matches: pd.DataFrame
) -> None:
    """The finding, pinned. Dixon-Coles beats the prior by most where the sides
    differ most, which is what a strength model should do and is not something
    anyone told it to do."""
    table = per_competition_table(scores).set_index("competition_id")
    recent = matches[matches["date"] >= min(fold.start for fold in walk_forward(matches))]
    sides = pd.concat([recent["elo_home"], recent["elo_away"]])
    spread = sides.groupby(pd.concat([recent["competition_id"]] * 2)).std()

    edge = table["class_prior"] - table["dixon_coles"]
    assert edge.corr(spread.reindex(edge.index)) > STRENGTH_CORRELATION


def test_what_the_bookmaker_knows_is_not_strength(
    scores: pd.DataFrame, matches: pd.DataFrame
) -> None:
    """And the gap to the closing line does *not* track it — measured at -0.14.

    That is the useful half: the rating has already taken what strength
    explains, so the model zoo is chasing something else, and it is worth about
    the same amount in every competition.
    """
    table = per_competition_table(scores).set_index("competition_id")
    recent = matches[matches["date"] >= min(fold.start for fold in walk_forward(matches))]
    sides = pd.concat([recent["elo_home"], recent["elo_away"]])
    spread = sides.groupby(pd.concat([recent["competition_id"]] * 2)).std()

    gap = table["dixon_coles"] - table["bookmaker"]
    assert abs(gap.corr(spread.reindex(gap.index))) < 0.5
