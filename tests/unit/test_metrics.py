"""The scoring rules, and what they refuse to score.

The arithmetic is checked against hand-computable cases rather than against
another library's output: a metric that agrees with scikit-learn is only as
right as the argument order it was called with, and the order is the mistake
this module is most exposed to.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    CLASSES,
    MetricError,
    Scores,
    one_hot,
    score,
    terms,
)

CERTAIN_HOME = np.array([[1.0, 0.0, 0.0]])
UNIFORM = np.full((3, 3), 1 / 3)


def test_the_classes_are_ordered_home_draw_away() -> None:
    """RPS accumulates across them, so the order is arithmetic, not convention."""
    assert CLASSES == ("H", "D", "A")


# ---- log loss ---------------------------------------------------------------


def test_a_uniform_forecast_scores_the_log_of_three() -> None:
    assert score(UNIFORM, ["H", "D", "A"]).log_loss == pytest.approx(math.log(3))


def test_a_correct_certainty_scores_zero() -> None:
    assert score(CERTAIN_HOME, ["H"]).log_loss == pytest.approx(0.0)


def test_ruling_out_what_happened_scores_infinity() -> None:
    """Not clipped. A forecast that said it could not happen was infinitely wrong."""
    assert score(CERTAIN_HOME, ["A"]).log_loss == math.inf


def test_the_class_prior_scores_its_own_entropy() -> None:
    """The measured base rates, scored on matches in exactly those proportions."""
    prior = np.array([0.45, 0.27, 0.28])
    outcomes = ["H"] * 45 + ["D"] * 27 + ["A"] * 28
    expected = -float(prior @ np.log(prior))
    assert score(np.tile(prior, (100, 1)), outcomes).log_loss == pytest.approx(expected)


# ---- RPS --------------------------------------------------------------------


def test_rps_knows_the_classes_are_ordered() -> None:
    """Predicting a draw when the away side won beats predicting a home win.

    This is the whole reason RPS is here: log loss gives both forecasts the
    same infinite score, because both ruled out what happened.
    """
    draw = np.array([[0.0, 1.0, 0.0]])
    home = np.array([[1.0, 0.0, 0.0]])
    assert score(draw, ["A"]).rps < score(home, ["A"]).rps


def test_a_correct_certainty_scores_zero_rps() -> None:
    assert score(CERTAIN_HOME, ["H"]).rps == pytest.approx(0.0)


def test_the_worst_possible_rps_is_one() -> None:
    """Certain of a home win, and the away side won: both thresholds maximally wrong."""
    assert score(CERTAIN_HOME, ["A"]).rps == pytest.approx(1.0)


def test_rps_is_computed_over_the_two_informative_thresholds() -> None:
    """A hand-computed case, so a change to the divisor is visible."""
    forecast = np.array([[0.5, 0.3, 0.2]])
    # Cumulative forecast (0.5, 0.8) against a draw's (0.0, 1.0).
    expected = (0.5**2 + 0.2**2) / 2
    assert score(forecast, ["D"]).rps == pytest.approx(expected)


# ---- accuracy ---------------------------------------------------------------


def test_accuracy_counts_the_most_likely_class() -> None:
    forecast = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    assert score(forecast, ["H", "A"]).accuracy == pytest.approx(1.0)
    assert score(forecast, ["A", "H"]).accuracy == pytest.approx(0.0)


def test_a_tie_goes_to_the_earlier_class() -> None:
    """Not a property worth relying on, and worth pinning so it cannot drift."""
    assert score(UNIFORM[:1], ["H"]).accuracy == pytest.approx(1.0)


# ---- what it refuses --------------------------------------------------------


def test_a_transposed_array_is_refused() -> None:
    """The mistake this module is most exposed to, and it would score fine."""
    with pytest.raises(MetricError, match=r"must be \(n, 3\)"):
        score(UNIFORM[:2].T, ["H", "D", "A"])


def test_a_one_dimensional_array_is_refused() -> None:
    with pytest.raises(MetricError, match=r"must be \(n, 3\)"):
        score(np.array([0.4, 0.3, 0.3]), ["H"])


def test_nulls_are_refused_rather_than_scored() -> None:
    """An unpriced match is dropped by the caller, never filled in here."""
    with pytest.raises(MetricError, match="nulls"):
        score(np.array([[np.nan, np.nan, np.nan]]), ["H"])


def test_probabilities_outside_the_unit_interval_are_refused() -> None:
    with pytest.raises(MetricError, match=r"outside \[0, 1\]"):
        score(np.array([[1.5, -0.5, 0.0]]), ["H"])


def test_a_forecast_that_forgot_to_normalise_is_refused() -> None:
    with pytest.raises(MetricError, match="sum to one"):
        score(np.array([[0.5, 0.5, 0.5]]), ["H"])


def test_a_length_mismatch_is_refused() -> None:
    with pytest.raises(MetricError, match="2 forecasts against 1 outcomes"):
        score(UNIFORM[:2], ["H"])


def test_an_unknown_outcome_is_refused() -> None:
    with pytest.raises(MetricError, match=r"not outcomes: \['X'\]"):
        one_hot(["H", "X"])


def test_a_match_with_no_result_is_refused() -> None:
    with pytest.raises(MetricError, match="nulls"):
        one_hot(["H", None])  # type: ignore[list-item]


# ---- shape ------------------------------------------------------------------


def test_the_metrics_are_the_means_of_the_per_match_terms() -> None:
    """The property the whole reporting layer rests on.

    Scoring once and grouping afterwards has to give what scoring each group
    would have given, or every per-competition figure in the backtest is wrong.
    """
    forecast = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.3, 0.4, 0.3]])
    outcomes = ["H", "A", "D"]
    contributions = terms(forecast, outcomes)
    computed = score(forecast, outcomes)
    assert computed.log_loss == pytest.approx(contributions["log_loss"].mean())
    assert computed.rps == pytest.approx(contributions["rps"].mean())
    assert computed.accuracy == pytest.approx(contributions["hit"].mean())


def test_a_summary_states_the_count_it_was_computed_over() -> None:
    summary = Scores(n=1234, log_loss=1.0, rps=0.2, accuracy=0.45).summary()
    assert "1,234 matches" in summary
    assert "45.0%" in summary
