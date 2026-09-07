"""Scoring what was served, and refusing to read noise as a finding.

Milestone 19. Two things are worth testing here and they are different in kind.
The first is arithmetic: a repeat is one forecast, an in-sample row is not
evidence, an unplayed fixture is a count rather than a gap. The second is the
guard the whole milestone exists for — that a difference smaller than the
sampling noise at the archive's size is reported as indistinguishable rather
than as drift.

No database. The archive is a frame shaped exactly as
:meth:`~src.storage.predictions.PostgresPredictionLog.recent` returns one,
which is the contract the pipeline actually depends on.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.evaluation.archive import (
    FORECAST_COLUMNS,
    KEY_COLUMN,
    SUMMARY_COLUMNS,
    detectable,
    horizon,
    latest,
    needed,
    reference,
    resolved,
    scorable,
    summarise,
)
from src.evaluation.metrics import CLASSES, terms
from src.ingestion.base import Result
from src.models import ensemble
from src.storage import predictions as log

HOME, AWAY = Result.HOME, Result.AWAY
"""The canonical table's vocabulary — ``H``/``D``/``A``, not the API's
``home``/``draw``/``away``. The archive carries the service's probabilities and
joins the table's outcomes, so this is the pair of vocabularies that meet."""

MODEL = "ensemble-calibrated"
VERSION = "0.13.0"
SPREAD = 0.3976
"""The shipped model's per-match log-loss spread, measured on the folds."""

TYPICAL = (0.37, 0.33, 0.30)
"""A home forecast that scores about the baseline these tests compare against.

``-log(0.37)`` is 0.9943 against a baseline of 1.0, so a handful of these
produce a difference that is *small* — which is the case the sample-size guard
has to get right. A forecast that was firm and correct would score 0.36 and be
distinguishable from the baseline after two matches, which would test the
threshold with a model that had genuinely changed rather than with noise.
"""


def served(
    match_id: str,
    *,
    at: str = "2026-09-01T12:00:00Z",
    probabilities: tuple[float, float, float] = TYPICAL,
    in_sample: bool = False,
    model_version: str = VERSION,
) -> dict[str, object]:
    """One row of the prediction log."""
    home, draw, away = probabilities
    return {
        "predicted_at": pd.Timestamp(at),
        "match_id": match_id,
        "competition_id": "ENG_1",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "match_date": dt.date(2026, 9, 5),
        "model": MODEL,
        "model_version": model_version,
        "prob_home": home,
        "prob_draw": draw,
        "prob_away": away,
        "in_sample": in_sample,
    }


def archive(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def played(*pairs: tuple[str, str]) -> pd.DataFrame:
    """A match table holding an outcome for each ``(match_id, result)``."""
    return pd.DataFrame(
        {"match_id": [one for one, _ in pairs], "result": [two for _, two in pairs]}
    )


def folds(n: int = 500, *, seed: int = 7) -> pd.DataFrame:
    """A walk-forward forecast table to take a baseline and a spread from."""
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet((4.0, 3.0, 3.0), size=n)
    outcomes = [list(CLASSES)[index] for index in rng.integers(0, 3, size=n)]
    return pd.DataFrame(
        {
            "forecaster": [MODEL] * n,
            "prob_home": draws[:, 0],
            "prob_draw": draws[:, 1],
            "prob_away": draws[:, 2],
            "result": outcomes,
        }
    )


# ---- the two names this package had to declare for itself ---------------------


def test_the_archive_and_the_model_cannot_disagree_about_column_names() -> None:
    """CI holds `src/evaluation` to importing nothing but the canonical schema,
    the metrics and utils — so these three names are declared locally rather
    than imported from the model package that also declares them. Declared
    twice, they would eventually differ, and the symptom would be a drift
    report scored on somebody else's columns.
    """
    assert FORECAST_COLUMNS == ensemble.FORECAST_COLUMNS
    assert KEY_COLUMN == ensemble.KEY_COLUMN
    assert list(FORECAST_COLUMNS) == [column for column in log.COLUMNS if column.startswith("prob")]


# ---- what counts as one forecast ---------------------------------------------


def test_a_fixture_priced_three_times_is_one_forecast_not_three() -> None:
    """The service caches and the log does not deduplicate — deliberately, the
    timestamps are what it is for. Counting them three times would shrink every
    error bar downstream by a factor the evidence does not support."""
    repeated = archive(
        served("a", at="2026-09-01T12:00:00Z"),
        served("a", at="2026-09-01T12:00:01Z"),
        served("a", at="2026-09-01T12:00:02Z"),
    )
    assert len(latest(repeated)) == 1


def test_the_surviving_row_is_the_last_one_served() -> None:
    """A forecast revised before kick-off is the forecast that was standing."""
    revised = archive(
        served("a", at="2026-09-01T09:00:00Z", probabilities=(0.5, 0.3, 0.2)),
        served("a", at="2026-09-04T09:00:00Z", probabilities=(0.8, 0.1, 0.1)),
    )
    assert float(latest(revised)["prob_home"].iloc[0]) == pytest.approx(0.8)


def test_two_versions_of_the_model_are_two_forecasts_about_one_match() -> None:
    """A redeploy means the same fixture was priced by two artefacts, and the
    question "did the served model get worse" is a question about a version."""
    both = archive(served("a"), served("a", model_version="0.14.0"))
    assert len(latest(both)) == 2


def test_an_empty_archive_stays_empty_rather_than_raising() -> None:
    assert latest(pd.DataFrame()).empty


# ---- what may be scored ------------------------------------------------------


def test_a_match_with_no_result_yet_is_kept_and_counted_not_dropped() -> None:
    """The difference between "we served nothing" and "we served plenty and
    none of it can be scored yet" is an outage against a Tuesday."""
    joined = resolved(archive(served("a"), served("b")), played(("a", HOME)))
    assert len(joined) == 2
    assert len(scorable(joined)) == 1


def test_an_in_sample_forecast_is_excluded_from_the_score() -> None:
    """The shipped artefact is fitted on the whole history, so a request for a
    match inside it is answered by a model that trained on it. Scoring that
    against the walk-forward folds would report memorisation as drift."""
    joined = resolved(
        archive(served("a", in_sample=True), served("b")), played(("a", HOME), ("b", HOME))
    )
    assert list(scorable(joined)["match_id"]) == ["b"]


def test_a_match_the_table_has_never_heard_of_is_unresolved_not_an_error() -> None:
    joined = resolved(archive(served("z")), played(("a", HOME)))
    assert joined["result"].isna().all()
    assert scorable(joined).empty


def test_an_empty_archive_joins_to_a_result_column_that_is_simply_empty() -> None:
    """So that a caller has one shape to handle rather than two."""
    joined = resolved(pd.DataFrame(), played(("a", HOME)))
    assert joined.empty
    # And filtering it asks for no column the empty frame does not have.
    assert scorable(joined).empty


# ---- the sample-size guard, which is the milestone ----------------------------


def test_nothing_is_distinguishable_in_an_empty_archive() -> None:
    assert detectable(SPREAD, 0) == float("inf")


def test_the_detectable_shift_shrinks_with_the_square_root_of_the_archive() -> None:
    """Quadrupling the archive halves the smallest shift it can see, which is
    why "wait for more data" is a real answer with a real timetable."""
    assert detectable(SPREAD, 400) == pytest.approx(detectable(SPREAD, 100) / 2)


def test_an_unknown_spread_gives_no_threshold_rather_than_a_confident_zero() -> None:
    assert np.isnan(detectable(float("nan"), 100))


def test_the_two_directions_agree_with_each_other() -> None:
    """`needed` is `detectable` solved for n, and a rounding that disagreed
    would put a page and a command at odds about how much archive is left."""
    rows = needed(SPREAD, 0.05)
    assert detectable(SPREAD, rows) <= 0.05
    assert detectable(SPREAD, rows - 1) > 0.05


def test_the_published_horizons_are_the_ones_the_documents_quote() -> None:
    """The numbers in README, CHANGELOG and the plan come from here."""
    assert needed(SPREAD, 0.05) == 243
    assert needed(SPREAD, 0.0163) == 2286
    assert needed(SPREAD, 0.01) == 6073


def test_a_difference_that_could_not_be_measured_is_refused_rather_than_returned() -> None:
    with pytest.raises(ValueError):
        needed(SPREAD, 0.0)


def test_the_horizon_table_is_one_row_per_difference() -> None:
    table = horizon(SPREAD, (0.05, 0.01))
    assert list(table["difference"]) == [0.05, 0.01]
    assert list(table["needed"]) == [243, 6073]


# ---- the baseline ------------------------------------------------------------


def test_the_baseline_and_the_spread_come_from_one_population() -> None:
    """Two numbers taken from two frames would eventually describe two things.
    The mean is what a served figure is compared against and the spread is what
    says whether the comparison could mean anything."""
    table = folds()
    measured = reference(table)
    scored = terms(table[["prob_home", "prob_draw", "prob_away"]].to_numpy(float), table["result"])
    assert measured.n == len(table)
    assert measured.log_loss == pytest.approx(scored["log_loss"].mean())
    assert measured.spread == pytest.approx(scored["log_loss"].std(ddof=1))


def test_one_forecast_has_no_spread_and_says_so_rather_than_reporting_zero() -> None:
    """Zero would make every difference detectable, which is the failure this
    whole module is arranged against."""
    measured = reference(folds(n=1))
    assert np.isnan(measured.spread)


def test_a_frame_with_no_forecaster_column_is_measured_whole() -> None:
    """The archive's own rows have no forecaster column, and a caller may ask
    for their spread."""
    table = folds().drop(columns=["forecaster"])
    assert reference(table).n == len(table)


# ---- the report --------------------------------------------------------------


def test_an_empty_archive_reports_no_versions_with_the_right_columns() -> None:
    """A service nobody has called yet is the ordinary state of a fresh
    deployment, not a failure to report."""
    report = summarise(pd.DataFrame(), played(("a", HOME)), spread=SPREAD, baseline=1.0)
    assert report.empty
    assert list(report.columns) == list(SUMMARY_COLUMNS)


def test_the_report_shows_what_was_dropped_beside_what_was_kept() -> None:
    """A mean over two matches next to one over sixty thousand invites them to
    be read as comparable, so every denominator is on the row."""
    report = summarise(
        archive(
            served("a"),
            served("a", at="2026-09-01T12:00:05Z"),
            served("b", in_sample=True),
            served("c"),
            served("d"),
        ),
        played(("a", HOME), ("b", HOME), ("c", AWAY)),
        spread=SPREAD,
        baseline=1.0,
    )
    (row,) = report.to_dict("records")
    assert row["logged"] == 4  # the repeat collapsed
    assert row["in_sample"] == 1
    assert row["unresolved"] == 1  # d has not been played
    assert row["n"] == 2  # a and c


def test_a_small_archive_reports_its_difference_as_indistinguishable() -> None:
    """The finding this milestone is mostly about. Two scored forecasts cannot
    tell a drifted model from a lucky Saturday, and the report has to say that
    in a column rather than leave it to a reader to work out."""
    report = summarise(
        archive(served("a"), served("c")),
        played(("a", HOME), ("c", HOME)),
        spread=SPREAD,
        baseline=1.0,
    )
    (row,) = report.to_dict("records")
    assert abs(row["drift"]) < row["detectable"]
    assert row["distinguishable"] is False


def test_a_model_that_is_wrong_on_everything_is_distinguishable_even_when_small() -> None:
    """The guard is a threshold, not a refusal: a served model that has fallen
    over produces a difference no sample size can explain away."""
    report = summarise(
        archive(*(served(str(index), probabilities=(0.001, 0.001, 0.998)) for index in range(20))),
        played(*((str(index), HOME) for index in range(20))),
        spread=SPREAD,
        baseline=1.0,
    )
    (row,) = report.to_dict("records")
    assert row["drift"] > row["detectable"]
    assert row["distinguishable"] is True


def test_without_a_baseline_there_is_no_drift_figure_rather_than_a_zero() -> None:
    """`make ensemble` may not have run. A missing comparison is null, and null
    is not distinguishable from anything."""
    report = summarise(archive(served("a")), played(("a", HOME)), spread=SPREAD, baseline=None)
    (row,) = report.to_dict("records")
    assert np.isnan(row["drift"])
    assert np.isnan(row["baseline"])
    assert row["distinguishable"] is False


def test_an_archive_with_nothing_scorable_still_reports_what_it_holds() -> None:
    """The realistic state of a young deployment: rows logged, none scored."""
    report = summarise(
        archive(served("a", in_sample=True)), played(("a", HOME)), spread=SPREAD, baseline=1.0
    )
    (row,) = report.to_dict("records")
    assert row["logged"] == 1
    assert row["n"] == 0
    assert np.isnan(row["log_loss"])
    assert row["detectable"] == float("inf")
    assert row["distinguishable"] is False


def test_each_served_version_is_its_own_row_newest_first() -> None:
    report = summarise(
        archive(
            served("a", at="2026-09-01T12:00:00Z"),
            served("b", at="2026-09-09T12:00:00Z", model_version="0.14.0"),
        ),
        played(("a", HOME), ("b", HOME)),
        spread=SPREAD,
        baseline=1.0,
    )
    assert list(report["model_version"]) == ["0.14.0", VERSION]
    assert list(report["n"]) == [1, 1]


def test_the_first_and_last_time_a_version_answered_are_recorded() -> None:
    """When a version was in service, which is what makes a drift figure
    attributable to a deployment rather than to a month."""
    report = summarise(
        archive(
            served("a", at="2026-09-01T12:00:00Z"),
            served("b", at="2026-09-06T18:30:00Z"),
        ),
        played(("a", HOME), ("b", HOME)),
        spread=SPREAD,
        baseline=1.0,
    )
    (row,) = report.to_dict("records")
    assert row["first_served"] < row["last_served"]


def test_the_spread_that_produced_the_threshold_is_carried_on_the_row() -> None:
    """So a page can price the archive it still needs without recomputing a σ
    from a table it does not have."""
    report = summarise(archive(served("a")), played(("a", HOME)), spread=SPREAD, baseline=1.0)
    assert float(report["spread"].iloc[0]) == pytest.approx(SPREAD)
