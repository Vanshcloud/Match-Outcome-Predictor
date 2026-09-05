"""The dashboard's readers, over frames small enough to reason about.

Two properties matter here and neither is about arithmetic, because there is no
arithmetic: every number this module produces comes from
:mod:`src.pipelines` or :mod:`src.evaluation` unchanged. What it owns is
*which rows* go in, and what happens when there are none.

The empty cases are the ones worth the tests. A clean checkout has no reports,
a filter can select nothing, and both must produce a page rather than a
traceback.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.services import reports as data
from src.models.ensemble import SHIPPED

FORECASTERS = (SHIPPED, "bookmaker")


def scores_frame() -> pd.DataFrame:
    """A backtest table shaped like the persisted one, including its ALL row."""
    rows = []
    for fold in (0, 1):
        for competition in ("ENG_1", "ESP_1", "ALL"):
            for name in FORECASTERS:
                for subset in ("common", "priced"):
                    rows.append(
                        {
                            "fold": fold,
                            "competition_id": competition,
                            "forecaster": name,
                            "subset": subset,
                            "n": 100 if competition != "ALL" else 200,
                            "log_loss": 1.02 if name == SHIPPED else 0.99,
                            "rps": 0.21,
                            "accuracy": 0.49,
                        }
                    )
    return pd.DataFrame(rows)


def forecasts_frame(rows: int = 600) -> pd.DataFrame:
    """A diagnostic pass with two competitions, two folds and real outcomes."""
    generator = np.random.default_rng(20260905)
    home = generator.uniform(0.2, 0.6, rows)
    draw = np.full(rows, 0.26)
    away = 1.0 - home - draw
    stated = np.column_stack([home, draw, away])
    drawn = np.array(
        [generator.choice(["H", "D", "A"], p=row / row.sum()) for row in stated], dtype=object
    )
    return pd.DataFrame(
        {
            "fold": np.tile([0, 1], rows // 2),
            "match": np.arange(rows),
            "competition_id": np.tile(["ENG_1", "ESP_1"], rows // 2),
            "prob_home": home,
            "prob_draw": draw,
            "prob_away": away,
            "forecaster": SHIPPED,
            "result": drawn,
        }
    )


# ---- loading, and the empty state --------------------------------------------


def test_a_clean_checkout_loads_nothing_and_says_what_is_missing(tmp_path: Path) -> None:
    """The state a new reader meets, and the one CI runs in."""
    reports = data.load_reports(tmp_path)
    assert reports.scores is None and reports.forecasts is None
    assert not reports.has_scores and not reports.has_forecasts
    absent = reports.missing()
    assert len(absent) == 2
    assert "make ensemble" in absent[0] and "make card" in absent[1]


def test_both_tables_are_read_when_both_are_there(tmp_path: Path) -> None:
    directory = tmp_path / data.ENSEMBLE_SUBDIR
    directory.mkdir(parents=True)
    scores_frame().to_parquet(directory / "backtest.parquet", index=False)
    forecasts_frame().to_parquet(directory / "forecasts.parquet", index=False)

    reports = data.load_reports(tmp_path)
    assert reports.has_scores and reports.has_forecasts
    assert reports.missing() == ()


def test_one_table_without_the_other_is_a_real_state(tmp_path: Path) -> None:
    """`make ensemble` writes one and `make card` the other, so a reader can
    easily have the first and not the second."""
    directory = tmp_path / data.ENSEMBLE_SUBDIR
    directory.mkdir(parents=True)
    scores_frame().to_parquet(directory / "backtest.parquet", index=False)

    reports = data.load_reports(tmp_path)
    assert reports.has_scores and not reports.has_forecasts
    assert reports.missing() == (f"{data.ENSEMBLE_SUBDIR}/forecasts.parquet (run `make card`)",)


def test_an_empty_table_counts_as_absent(tmp_path: Path) -> None:
    """A file holding no rows is not something to render a chart from."""
    directory = tmp_path / data.ENSEMBLE_SUBDIR
    directory.mkdir(parents=True)
    scores_frame().head(0).to_parquet(directory / "backtest.parquet", index=False)
    assert not data.load_reports(tmp_path).has_scores


# ---- the tables a panel renders ----------------------------------------------


def test_the_leaderboard_is_the_pooled_table_unchanged() -> None:
    table = data.leaderboard(scores_frame())
    assert list(table["forecaster"]) == ["bookmaker", SHIPPED]
    assert table["log_loss"].iloc[0] == pytest.approx(0.99)


def test_per_competition_keeps_one_row_per_competition() -> None:
    table = data.by_competition(scores_frame())
    assert set(table["competition_id"]) == {"ENG_1", "ESP_1"}


def test_filtering_narrows_to_the_shipped_model_and_the_choices() -> None:
    forecasts = forecasts_frame()
    everything = data.filtered_forecasts(forecasts)
    assert len(everything) == len(forecasts)

    one = data.filtered_forecasts(forecasts, competitions=["ENG_1"])
    assert set(one["competition_id"]) == {"ENG_1"}

    fold = data.filtered_forecasts(forecasts, folds=[1])
    assert set(fold["fold"]) == {1}

    both = data.filtered_forecasts(forecasts, competitions=["ENG_1"], folds=[0])
    assert set(both["competition_id"]) == {"ENG_1"} and set(both["fold"]) == {0}


def test_another_forecasters_rows_are_not_the_shipped_models() -> None:
    forecasts = forecasts_frame()
    forecasts.loc[: len(forecasts) // 2, "forecaster"] = "bookmaker"
    assert set(data.filtered_forecasts(forecasts)["forecaster"]) == {SHIPPED}


def test_a_filter_that_selects_nothing_is_a_question_with_an_answer() -> None:
    """Not an error. The panel renders "no matches match that filter"."""
    empty = data.filtered_forecasts(forecasts_frame(), competitions=["NOWHERE"])
    assert empty.empty
    table = data.reliability_table(empty)
    assert table.empty
    assert np.isnan(data.calibration_error(table))
    per_class = data.per_class_tables(empty)
    assert set(per_class) == {"H", "D", "A"}
    assert all(one.empty for one in per_class.values())
    assert data.worst_competitions(empty).empty


def test_reliability_is_computed_over_whatever_rows_it_is_handed() -> None:
    forecasts = data.filtered_forecasts(forecasts_frame())
    table = data.reliability_table(forecasts, bins=5)
    assert len(table) <= 5
    assert set(table.columns) >= {"predicted", "observed", "gap", "n"}
    assert table["n"].sum() == len(forecasts) * 3  # three statements per match
    assert 0.0 <= data.calibration_error(table) <= 1.0


def test_bins_are_the_readers_choice() -> None:
    forecasts = data.filtered_forecasts(forecasts_frame())
    assert len(data.reliability_table(forecasts, bins=20)) >= len(
        data.reliability_table(forecasts, bins=5)
    )


def test_per_class_tables_cover_the_three_outcomes() -> None:
    tables = data.per_class_tables(data.filtered_forecasts(forecasts_frame()))
    assert set(tables) == {"H", "D", "A"}


def test_worst_competitions_is_ordered_worst_first() -> None:
    forecasts = data.filtered_forecasts(forecasts_frame(rows=1000))
    table = data.worst_competitions(forecasts)
    assert list(table["calibration_error"]) == sorted(table["calibration_error"], reverse=True)
