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


def market_frame() -> pd.DataFrame:
    """A disagreement table shaped like the one `make card` writes.

    All five bands, because a page finds its own row by band and a table
    missing the one a fixture falls in is indistinguishable from no table at
    all — which is a different sentence on the screen. The numbers are the ones
    the real run produced, so a reader of this file sees the finding rather
    than five placeholders.
    """
    return pd.DataFrame(
        {
            "band": ["<2%", "2-5%", "5-10%", "10-20%", ">20%"],
            "n": [9628, 21139, 20874, 9419, 829],
            "model": [0.9881, 1.0136, 1.0199, 1.0402, 1.0810],
            "market": [0.9872, 1.0099, 1.0059, 0.9864, 0.8975],
            "model_minus_market": [0.0009, 0.0037, 0.0140, 0.0538, 0.1835],
            "model_better": [0.4917, 0.4844, 0.4638, 0.4195, 0.3546],
        }
    )


def archive_frame(
    *, scored: int = 40, distinguishable: bool = False, spread: float = 0.3976
) -> pd.DataFrame:
    """A drift report shaped like the one `make archive` writes.

    One served version, and the young-archive case by default: rows logged,
    a handful scorable, and a difference smaller than the noise at that size.
    That is the state a real deployment is in for months, and it is the one the
    panel has to render without implying a finding.
    """
    return pd.DataFrame(
        {
            "model": [SHIPPED],
            "model_version": ["0.13.0"],
            "logged": [scored + 12],
            "in_sample": [8],
            "unresolved": [4],
            "n": [scored],
            "log_loss": [1.0800 if distinguishable else 1.0190],
            "rps": [0.2101],
            "accuracy": [0.4900],
            "baseline": [1.0165],
            "spread": [spread],
            "drift": [0.0635 if distinguishable else 0.0025],
            "detectable": [0.0123 if distinguishable else 0.1232],
            "distinguishable": [distinguishable],
            "first_served": [pd.Timestamp("2026-09-01T09:00:00Z")],
            "last_served": [pd.Timestamp("2026-09-07T21:00:00Z")],
        }
    )


# ---- loading, and the empty state --------------------------------------------


def test_a_clean_checkout_loads_nothing_and_says_what_is_missing(tmp_path: Path) -> None:
    """The state a new reader meets, and the one CI runs in."""
    reports = data.load_reports(tmp_path)
    assert reports.scores is None and reports.forecasts is None and reports.market is None
    assert not (reports.has_scores or reports.has_forecasts or reports.has_market)
    absent = reports.missing()
    assert len(absent) == 3
    assert "make ensemble" in absent[0]
    assert all("make card" in named for named in absent[1:])
    # The drift report is deliberately not in that list: the other three are
    # missing because a command has not been run, and this one is missing
    # because the service has not been called.
    assert reports.archive is None
    assert "has not been written" in str(reports.missing_archive())


def test_every_table_is_read_when_they_are_all_there(tmp_path: Path) -> None:
    directory = tmp_path / data.ENSEMBLE_SUBDIR
    directory.mkdir(parents=True)
    scores_frame().to_parquet(directory / "backtest.parquet", index=False)
    forecasts_frame().to_parquet(directory / "forecasts.parquet", index=False)
    market_frame().to_parquet(directory / "market.parquet", index=False)
    archive_frame().to_parquet(directory / "archive.parquet", index=False)

    reports = data.load_reports(tmp_path)
    assert reports.has_scores and reports.has_forecasts and reports.has_market
    assert reports.has_archive and reports.missing_archive() is None
    assert reports.missing() == ()


def test_one_table_without_the_other_is_a_real_state(tmp_path: Path) -> None:
    """`make ensemble` writes one and `make card` the other, so a reader can
    easily have the first and not the second."""
    directory = tmp_path / data.ENSEMBLE_SUBDIR
    directory.mkdir(parents=True)
    scores_frame().to_parquet(directory / "backtest.parquet", index=False)

    reports = data.load_reports(tmp_path)
    assert reports.has_scores and not reports.has_forecasts
    assert reports.missing() == (
        f"{data.ENSEMBLE_SUBDIR}/forecasts.parquet (run `make card`)",
        f"{data.ENSEMBLE_SUBDIR}/market.parquet (run `make card`)",
    )


def test_an_archive_that_holds_no_rows_says_the_log_is_empty(tmp_path: Path) -> None:
    """`make archive` writes a file even when the log has nothing in it, and
    "the report has not been written" and "the report is empty" are different
    sentences to put in front of a reader."""
    directory = tmp_path / data.ENSEMBLE_SUBDIR
    directory.mkdir(parents=True)
    archive_frame().head(0).to_parquet(directory / "archive.parquet", index=False)

    reports = data.load_reports(tmp_path)
    assert not reports.has_archive
    assert reports.missing_archive() == "the prediction log is empty"


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
