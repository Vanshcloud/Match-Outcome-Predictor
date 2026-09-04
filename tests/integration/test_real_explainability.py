"""The two attribution methods against the real table, and the card they feed.

Skips without the three built tables. The cheap assertions fit one family over
one fold and check that both methods find the same three blocks; the expensive
one reads `docs/MODEL_CARD.md` if it is there and pins what it claims.

The finding this file exists to protect is the **disagreement**. Permutation
ranks the elo block far above form; the ablation ranks form above elo. Both are
right, and the gap between them is the substitution Milestone 8 measured: the
model leans on elo, and can do without it, because Dixon-Coles says most of the
same thing. A change that quietly made the two agree would have lost that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.model_card import CARD_FILENAME
from src.evaluation.reliability import expected_calibration_error, reliability
from src.explainability.permutation import importance
from src.explainability.shapley import TREE_FAMILIES, attribution
from src.models.dataset import BLOCKS, DESIGN_COLUMNS
from src.models.ensemble import FORECAST_COLUMNS
from src.models.splits import walk_forward
from src.models.zoo import build
from src.pipelines.backtest import BACKTEST_FILENAME
from src.pipelines.features import FEATURES_FILENAME
from src.pipelines.ingest import MATCHES_FILENAME
from src.pipelines.ratings import RATINGS_FILENAME
from src.pipelines.report import (
    SHIPPED,
    reliability_by_class,
    reliability_by_competition,
    shipped_forecaster,
)
from src.pipelines.train import ABLATION_SUBDIR, ablation_table
from src.utils.config import load_settings
from src.utils.paths import PROJECT_ROOT

pytestmark = pytest.mark.integration

SETTINGS = load_settings()
MATCHES = SETTINGS.paths.processed_dir / MATCHES_FILENAME
RATINGS = SETTINGS.paths.features_dir / RATINGS_FILENAME
FEATURES = SETTINGS.paths.features_dir / FEATURES_FILENAME
ABLATED = SETTINGS.paths.reports_dir / ABLATION_SUBDIR / BACKTEST_FILENAME
CARD = PROJECT_ROOT / "docs" / CARD_FILENAME

EXPLAINED = "lightgbm"
"""The family `make ablation` runs, so all three methods describe one model."""


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    for label, path in (("matches", MATCHES), ("ratings", RATINGS), ("features", FEATURES)):
        if not path.is_file():
            pytest.skip(f"no {label} at {path}")
    return (
        pd.read_parquet(MATCHES)
        .merge(pd.read_parquet(RATINGS), on="match_id", how="left")
        .merge(pd.read_parquet(FEATURES), on="match_id", how="left")
    )


@pytest.fixture(scope="module")
def broken(matches: pd.DataFrame) -> pd.DataFrame:
    """What breaking each block costs the model, on the most recent fold."""
    fold = list(walk_forward(matches))[-1]
    return importance(build(EXPLAINED), fold.train, fold.evaluate).set_index("block")


@pytest.fixture(scope="module")
def attributed(matches: pd.DataFrame) -> pd.DataFrame:
    fold = list(walk_forward(matches))[-1]
    return attribution(build(EXPLAINED), fold.train, fold.evaluate).set_index("block")


def _as_forecasts(stated: np.ndarray, evaluate: pd.DataFrame) -> pd.DataFrame:
    """One forecaster's probabilities in the shape the breakdowns group by."""
    forecasts = pd.DataFrame(stated, columns=list(FORECAST_COLUMNS))
    forecasts["result"] = evaluate["result"].to_numpy()
    forecasts["competition_id"] = evaluate["competition_id"].to_numpy()
    return forecasts


# ---- permutation, on real matches -------------------------------------------


def test_every_block_is_measured(broken: pd.DataFrame) -> None:
    assert set(broken.index) == set(BLOCKS)


def test_breaking_a_block_never_helps(broken: pd.DataFrame) -> None:
    """A block whose removal improved the model would be a block that should
    not have shipped, and Milestone 8's ablation already said none is."""
    assert (broken["delta_log_loss"] > -0.001).all()


def test_the_deltas_are_far_larger_than_the_noise_between_repeats(
    broken: pd.DataFrame,
) -> None:
    """The three blocks that matter clear their own spread by an order of
    magnitude. The two that do not are the two Milestone 8 measured at 0.0001."""
    real = broken[broken["delta_log_loss"] > 0.001]
    assert len(real) == 3
    assert (real["delta_log_loss"] > 10 * real["spread"]).all()


def test_the_model_leans_hardest_on_the_ratings(broken: pd.DataFrame) -> None:
    assert broken["delta_log_loss"].idxmax() == "elo"
    assert broken.loc["elo", "delta_log_loss"] > broken.loc["form", "delta_log_loss"]


def test_schedule_and_head_to_head_are_worth_nothing_here_either(
    broken: pd.DataFrame,
) -> None:
    """Milestone 8's ablation put both at 0.0001–0.0003. A second method
    agreeing is what makes that a property of the features rather than of the
    way they were measured."""
    assert broken.loc["schedule", "delta_log_loss"] < 0.001
    assert broken.loc["head_to_head", "delta_log_loss"] < 0.001


# ---- SHAP, on real matches ---------------------------------------------------


def test_the_shares_are_a_decomposition(attributed: pd.DataFrame) -> None:
    assert attributed["share"].sum() == pytest.approx(1.0)
    assert attributed["columns"].sum() == len(DESIGN_COLUMNS)


def test_the_same_three_blocks_carry_the_model(attributed: pd.DataFrame) -> None:
    """A different method, the same answer about which blocks matter — and
    they carry 95% of the arithmetic between them."""
    assert set(attributed.index[:3]) == {"elo", "form", "dixon_coles"}
    assert attributed["share"].head(3).sum() > 0.9


def test_the_families_shap_covers_are_the_ones_with_a_tree() -> None:
    assert EXPLAINED in TREE_FAMILIES


# ---- where the three methods disagree ---------------------------------------


def test_permutation_and_the_ablation_rank_the_top_two_differently(
    broken: pd.DataFrame,
) -> None:
    """The finding. Permutation asks "does this model use the block"; the
    ablation asks "would a model built without it be worse". Elo wins the first
    and form wins the second, because Dixon-Coles is a substitute for elo and
    nothing substitutes for form."""
    if not ABLATED.is_file():
        pytest.skip(f"no ablation at {ABLATED}; run scripts/train.py --ablate {EXPLAINED}")
    ablated = ablation_table(pd.read_parquet(ABLATED), EXPLAINED).set_index("block")
    assert broken["delta_log_loss"].idxmax() == "elo"
    assert ablated["delta_log_loss"].idxmax() == "form"
    assert ablated.loc["elo", "delta_log_loss"] > ablated.loc["dixon_coles", "delta_log_loss"]


def test_breaking_a_block_costs_more_than_never_having_had_it(
    broken: pd.DataFrame,
) -> None:
    """A retrained model reallocates onto what is left; one handed a shuffled
    column at prediction time cannot. The permutation deltas are an order of
    magnitude larger for that reason, and reading them as ablation numbers
    would overstate every block."""
    if not ABLATED.is_file():
        pytest.skip(f"no ablation at {ABLATED}")
    ablated = ablation_table(pd.read_parquet(ABLATED), EXPLAINED).set_index("block")
    for block in ("elo", "dixon_coles"):
        assert broken.loc[block, "delta_log_loss"] > ablated.loc[block, "delta_log_loss"]


# ---- the shipped model's own honesty -----------------------------------------


def test_the_shipped_model_is_the_one_the_card_is_written_about() -> None:
    assert shipped_forecaster().name == SHIPPED


def test_the_card_exists_and_names_what_it_measured() -> None:
    if not CARD.is_file():
        pytest.skip(f"no model card at {CARD}; run scripts/model_card.py")
    written = CARD.read_text(encoding="utf-8")
    assert written.startswith(f"# Model card — {SHIPPED}")
    assert "## What it must not be used for" in written
    assert "Betting" in written


def test_the_draw_column_is_the_one_the_model_has_no_opinion_about(
    matches: pd.DataFrame,
) -> None:
    """It states about a quarter for a draw and almost never more, while both
    other classes span the range. That is the shape of a model with nothing to
    say about draws rather than one that is wrong about them."""
    fold = list(walk_forward(matches, folds=1))[-1]
    stated = build(EXPLAINED).forecast(fold.train, fold.evaluate)
    per_class = reliability_by_class(_as_forecasts(stated, fold.evaluate))
    assert per_class["D"]["predicted"].max() < 0.45
    assert per_class["H"]["predicted"].max() > 0.8
    assert per_class["A"]["predicted"].max() > 0.8


def test_the_card_reports_the_draw_column_as_its_most_honest() -> None:
    """And that is honesty by refusing to have an opinion. A column stated at
    0.27 against a base rate of 0.26 cannot be far wrong, which is exactly why
    the card carries the score table beside the reliability one: reliable is
    not the same as useful, and the class prior is the proof.

    Read off the written card rather than recomputed. The claim is about the
    shipped model over five folds, and refitting a blend of three twice per
    fold to re-derive one ordering is an hour to confirm a document."""
    if not CARD.is_file():
        pytest.skip(f"no model card at {CARD}; run scripts/model_card.py")
    rows = [
        line.split("|")[1:4]
        for line in CARD.read_text(encoding="utf-8").splitlines()
        if line.startswith(("| H |", "| D |", "| A |"))
    ]
    errors = {row[0].strip(): float(row[2]) for row in rows}
    assert set(errors) == {"H", "D", "A"}
    assert min(errors, key=lambda label: errors[label]) == "D"


def test_the_least_reliable_competitions_are_the_smallest_ones(
    matches: pd.DataFrame,
) -> None:
    """Aggregate honesty hides them, which is the reason the card has this
    section: the pooled figure is dominated by the competitions with the most
    matches, and they are not the ones a reader should be careful about."""
    fold = list(walk_forward(matches, folds=1))[-1]
    stated = build(EXPLAINED).forecast(fold.train, fold.evaluate)
    table = reliability_by_competition(_as_forecasts(stated, fold.evaluate), minimum=100)
    pooled = expected_calibration_error(reliability(stated, fold.evaluate["result"]))
    assert len(table) > 10
    assert table["calibration_error"].max() > 3 * pooled
