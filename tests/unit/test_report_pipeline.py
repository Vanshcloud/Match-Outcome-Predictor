"""The two breakdowns calibration left pooled, and the card assembled from them.

Milestone 9 reported one calibration number on purpose. These are the questions
it hides — which class the model is dishonest about, and where — and both come
off the same per-match pass, so what is tested here is the grouping rather than
the arithmetic (`test_reliability.py` owns that).

The forecaster throughout is the class prior: real, causal, free to run, and
honest by construction on a synthetic league — which makes it exactly the right
thing to check a *breakdown* with, because any structure the tables show is
structure the grouping invented.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.metrics import CLASSES
from src.evaluation.model_card import CARD_FILENAME
from src.models.baselines import Bookmaker, ClassPrior
from src.models.dataset import DESIGN_COLUMNS
from src.models.ensemble import fold_forecasts
from src.pipelines.report import (
    SHIPPED,
    build_model_card,
    card_settings,
    reliability_by_class,
    reliability_by_competition,
    shipped_forecaster,
    write_model_card,
)
from tests.factories import modelled_frame, season_labels

LEAGUE = pd.concat(
    [
        modelled_frame(seasons=season_labels(2012, 10), teams=12),
        modelled_frame(
            seasons=season_labels(2012, 10), teams=10, competition_id="ESP_1", team_prefix="Club"
        ),
    ]
).sort_values("date", kind="stable", ignore_index=True)

FORECASTS = fold_forecasts(LEAGUE, [ClassPrior()], folds=2)
SCORES = pd.DataFrame(
    {
        "forecaster": ["class_prior", "bookmaker"],
        "subset": ["common", "common"],
        "competition_id": ["ALL", "ALL"],
        "n": [100, 100],
        "log_loss": [1.07, 0.99],
        "rps": [0.22, 0.20],
        "accuracy": [0.43, 0.50],
    }
)


# ---- by class ----------------------------------------------------------------


def test_each_class_gets_its_own_table() -> None:
    tables = reliability_by_class(FORECASTS)
    assert set(tables) == set(CLASSES)


def test_a_class_table_holds_one_statement_per_match() -> None:
    """Where the pooled table holds three. That is the whole difference: "is
    this model honest" against "which of the three is it dishonest about"."""
    tables = reliability_by_class(FORECASTS)
    for table in tables.values():
        assert table["n"].sum() == len(FORECASTS)


def test_the_three_are_not_the_same_table() -> None:
    """A draw is stated at around 0.27 and a home win at around 0.45, so they
    do not even land in the same bins — a grouping that returned three
    identical tables would be selecting the wrong column."""
    tables = reliability_by_class(FORECASTS)
    assert set(tables["H"]["bin"]) != set(tables["D"]["bin"])


def test_a_forecaster_that_priced_nothing_contributes_nothing() -> None:
    without_odds = LEAGUE.drop(columns=["odds_home", "odds_draw", "odds_away"])
    unpriced = fold_forecasts(without_odds, [Bookmaker()], folds=2)
    tables = reliability_by_class(unpriced)
    assert all(table.empty for table in tables.values())


# ---- by competition ----------------------------------------------------------


def test_one_row_per_competition_with_enough_matches() -> None:
    table = reliability_by_competition(FORECASTS, minimum=1)
    assert set(table["competition_id"]) == {"ENG_1", "ESP_1"}
    assert (table["matches"] > 0).all()


def test_the_worst_competition_is_first() -> None:
    """The order the question is asked in: a card's "where is it least
    reliable" section that had to be sorted by the reader is a list."""
    table = reliability_by_competition(FORECASTS, minimum=1)
    assert table["calibration_error"].is_monotonic_decreasing


def test_a_competition_too_small_to_measure_is_dropped() -> None:
    """Not reported with a wide error bar the table has no column for. A "least
    reliable" list of the smallest samples has said nothing."""
    assert reliability_by_competition(FORECASTS, minimum=10_000).empty
    assert list(reliability_by_competition(FORECASTS, minimum=10_000).columns) == [
        "competition_id",
        "matches",
        "calibration_error",
    ]


# ---- the card ----------------------------------------------------------------


def test_the_shipped_model_is_the_calibrated_blend() -> None:
    assert shipped_forecaster().name == SHIPPED
    assert [member.name for member in shipped_forecaster().inner.members] == [  # type: ignore[union-attr]
        "xgboost",
        "logistic_regression",
        "mlp",
    ]


def test_the_build_description_names_what_a_reader_needs() -> None:
    """ "Its hyperparameters" is four sets of them and a scalar fitted per fold.
    Which three families, over how many columns, is the answerable version."""
    settings = card_settings(DESIGN_COLUMNS, folds=5)
    assert settings["columns"] == 30
    assert "xgboost" in settings["members"]
    assert "tuning slice" in settings["selection"]


def test_the_card_is_assembled_from_the_tables_that_measured_it() -> None:
    card = build_model_card(
        SCORES, FORECASTS, LEAGUE, name="class_prior", columns=DESIGN_COLUMNS, folds=2
    )
    assert card.name == "class_prior"
    assert card.matches == len(FORECASTS)
    assert card.competitions == 2
    assert set(card.per_class) == set(CLASSES)
    assert len(card.columns) == 30


def test_the_card_spans_the_history_it_was_trained_on() -> None:
    card = build_model_card(SCORES, FORECASTS, LEAGUE, name="class_prior", folds=2)
    assert card.trained_from == f"{LEAGUE['date'].min():%Y-%m-%d}"
    assert card.trained_to == f"{LEAGUE['date'].max():%Y-%m-%d}"


def test_a_forecaster_the_pass_does_not_hold_produces_an_empty_card(
    tmp_path: Path,
) -> None:
    """Rather than the wrong model's numbers under the right model's name."""
    card = build_model_card(SCORES, FORECASTS, LEAGUE, name="nobody", folds=2)
    assert card.matches == 0
    assert card.reliability.empty
    assert write_model_card(card, tmp_path).is_file()


def test_writing_it_creates_the_directory_and_names_the_file(tmp_path: Path) -> None:
    card = build_model_card(SCORES, FORECASTS, LEAGUE, name="class_prior", folds=2)
    written = write_model_card(card, tmp_path / "docs")
    assert written == tmp_path / "docs" / CARD_FILENAME
    assert "# Model card — class_prior" in written.read_text(encoding="utf-8")


def test_regenerating_it_changes_nothing(tmp_path: Path) -> None:
    """The property that makes it reviewable: a run that measured the same
    thing produces the same bytes, so a diff is a change in the model."""
    card = build_model_card(SCORES, FORECASTS, LEAGUE, name="class_prior", folds=2)
    first = write_model_card(card, tmp_path).read_text(encoding="utf-8")
    second = write_model_card(card, tmp_path).read_text(encoding="utf-8")
    assert first == second


def test_the_settings_can_be_overridden_for_a_model_that_is_not_the_blend() -> None:
    card = build_model_card(
        SCORES, FORECASTS, LEAGUE, name="class_prior", folds=2, settings={"counted": "per fold"}
    )
    assert card.settings == {"counted": "per fold"}
    assert "| `counted` | per fold |" in card.render()


@pytest.mark.parametrize("bins", [2, 20])
def test_the_bin_count_reaches_every_breakdown(bins: int) -> None:
    card = build_model_card(SCORES, FORECASTS, LEAGUE, name="class_prior", folds=2, bins=bins)
    assert len(card.reliability) <= bins
    assert all(len(table) <= bins for table in card.per_class.values())
