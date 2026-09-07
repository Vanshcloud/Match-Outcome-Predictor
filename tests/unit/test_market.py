"""The closing line as a forecast, and what disagreeing with it is worth.

Milestone 17. Two questions, and they are the same subject from opposite ends:
does the de-vig produce the probabilities the benchmark was scored against, and
does the disagreement table say what the milestone claims it says.

The direction of that table is the finding, so it is asserted as a direction —
a test that pinned five means would go red on any change to the split and tell
nobody anything about whether the claim still held.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.market import (
    DISAGREEMENT_LABELS,
    MARKET_COLUMNS,
    band,
    disagreement,
    distance,
    implied_probabilities,
    overround,
)
from src.ingestion.base import ODDS_COLUMNS
from src.models.baselines import Bookmaker
from src.pipelines.report import market_comparison

EVEN = [3.0, 3.0, 3.0]
"""Three equal prices: a book with no margin at all, and a third each."""


# ---- turning a price into a probability --------------------------------------


def test_a_fair_book_implies_the_probabilities_it_looks_like() -> None:
    stated = implied_probabilities(np.array([EVEN]))
    assert stated[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert overround(np.array([EVEN])) == pytest.approx([0.0])


def test_the_margin_is_removed_and_reported() -> None:
    """The removal is proportional and the size of what was removed is
    published beside it, because the removal is an assumption about how the
    margin is spread and a reader has to be able to judge it."""
    odds = np.array([[2.0, 3.5, 4.0]])
    assert implied_probabilities(odds).sum() == pytest.approx(1.0)
    assert overround(odds) == pytest.approx([0.5 + 1 / 3.5 + 0.25 - 1.0])


def test_a_shorter_price_implies_a_higher_probability() -> None:
    stated = implied_probabilities(np.array([[1.5, 4.0, 7.0]]))[0]
    assert stated[0] > stated[1] > stated[2]


@pytest.mark.parametrize(
    "odds",
    [
        [np.nan, 3.0, 3.0],
        [2.0, 3.0, np.inf],
        [1.0, 3.0, 3.0],
        [0.5, 3.0, 3.0],
    ],
    ids=["missing", "infinite", "certainty", "below-one"],
)
def test_a_row_that_is_not_three_prices_is_not_a_forecast(odds: list[float]) -> None:
    """All three or none. Two of three prices normalise to something that sums
    to one and is a forecast of nothing — which would be scored as a valid
    answer that happens to be poor."""
    assert np.isnan(implied_probabilities(np.array([odds]))).all()
    assert np.isnan(overround(np.array([odds]))).all()


def test_the_benchmark_and_the_page_use_one_implementation() -> None:
    """Three callers need "what did the market say", and the number a reader
    sees has to be the number the model was scored against."""
    frame = pd.DataFrame([[2.0, 3.5, 4.0]], columns=list(ODDS_COLUMNS))
    assert Bookmaker().forecast(frame, frame) == pytest.approx(
        implied_probabilities(frame[list(MARKET_COLUMNS)].to_numpy(dtype=float))
    )


def test_the_odds_block_and_the_market_order_cannot_drift() -> None:
    """The schema's odds columns are a dict, whose order is incidental. This
    order is arithmetic — it is the class order every metric accumulates over."""
    assert set(MARKET_COLUMNS) == set(ODDS_COLUMNS)
    assert MARKET_COLUMNS == ("odds_home", "odds_draw", "odds_away")


# ---- how far apart two forecasts are -----------------------------------------


def test_two_identical_forecasts_are_no_distance_apart() -> None:
    same = np.array([[0.5, 0.3, 0.2]])
    assert distance(same, same) == pytest.approx([0.0])


def test_opposite_certainties_are_the_whole_distance() -> None:
    assert distance(np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]])) == pytest.approx(
        [1.0]
    )


def test_the_distance_reads_as_a_percentage_point_gap() -> None:
    """Seven points moved from home to away is a gap of seven points, not
    fourteen — which is why it is half the sum of absolute differences."""
    apart = distance(np.array([[0.50, 0.25, 0.25]]), np.array([[0.43, 0.25, 0.32]]))
    assert apart == pytest.approx([0.07])


@pytest.mark.parametrize(
    ("gap", "expected"),
    [(0.0, "<2%"), (0.02, "<2%"), (0.021, "2-5%"), (0.2, "10-20%"), (0.9, ">20%")],
)
def test_a_gap_finds_its_own_band(gap: float, expected: str) -> None:
    """How one fixture on a page selects its row of the table."""
    assert band(gap) == expected


def test_a_gap_that_is_not_a_number_has_no_band() -> None:
    assert band(float("nan")) is None


# ---- the measurement ---------------------------------------------------------


def frames(rows: int = 8000) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """A market that is right and a model that drifts from it as it disagrees.

    The forecasts are built rather than sampled — the model here is
    deliberately wrong in proportion to how far it strays — but the *outcomes*
    have to be drawn, and a log loss over a few dozen drawn outcomes is noise.
    Eight thousand rows is enough that the narrowest band holds hundreds, which
    is what makes the direction below a property rather than a seed.
    """
    generator = np.random.default_rng(17)
    truth = np.tile([0.5, 0.25, 0.25], (rows, 1))
    drift = np.linspace(0.0, 0.4, rows)
    model = np.clip(truth + np.column_stack([-drift, drift / 2, drift / 2]), 1e-6, 1)
    model /= model.sum(axis=1, keepdims=True)
    outcomes = pd.Series(generator.choice(["H", "D", "A"], size=rows, p=[0.5, 0.25, 0.25]))
    return model, truth, outcomes


def test_the_table_has_one_row_per_band_that_holds_matches() -> None:
    model, market, outcomes = frames()
    table = disagreement(model, market, outcomes)
    assert list(table.columns) == [
        "band",
        "n",
        "model",
        "market",
        "model_minus_market",
        "model_better",
    ]
    assert set(table["band"]).issubset(DISAGREEMENT_LABELS)
    assert table["n"].sum() == len(model)


def test_the_deficit_grows_with_the_disagreement() -> None:
    """Milestone 17's finding, as the shape of the answer rather than as five
    pinned means: a model that strays further from a well-calibrated market
    pays for it, and this is the function that has to show it."""
    model, market, outcomes = frames()
    table = disagreement(model, market, outcomes)
    assert table["model_minus_market"].is_monotonic_increasing


def test_a_match_neither_side_priced_is_in_no_band() -> None:
    """A band whose model column is computed over more matches than its market
    column compares two different questions."""
    model = np.array([[0.5, 0.25, 0.25], [np.nan, np.nan, np.nan], [0.4, 0.3, 0.3]])
    market = np.array([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25], [np.nan, np.nan, np.nan]])
    table = disagreement(model, market, pd.Series(["H", "H", "H"]))
    assert table["n"].sum() == 1


def test_nothing_priced_is_an_empty_table_rather_than_a_failure() -> None:
    """The clean-checkout shape. A caller renders "not measured yet"."""
    empty = disagreement(np.empty((0, 3)), np.empty((0, 3)), pd.Series([], dtype=str))
    assert empty.empty
    assert "model_minus_market" in empty.columns


# ---- the report table --------------------------------------------------------


def forecasts_with(match_ids: list[str], probabilities: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": 0,
            "match": np.arange(len(match_ids)),
            "match_id": match_ids,
            "competition_id": "ENG_1",
            "prob_home": [row[0] for row in probabilities],
            "prob_draw": [row[1] for row in probabilities],
            "prob_away": [row[2] for row in probabilities],
            "forecaster": "ensemble-calibrated",
            "result": "H",
        }
    )


def test_the_comparison_joins_on_the_fixture_rather_than_on_a_position() -> None:
    """Milestone 17 put ``match_id`` on every forecast for exactly this. The
    matches here are deliberately in the other order."""
    forecasts = forecasts_with(["a", "b"], [[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]])
    matches = pd.DataFrame(
        {
            "match_id": ["b", "a"],
            "odds_home": [1.5, 3.0],
            "odds_draw": [4.0, 3.0],
            "odds_away": [7.0, 3.0],
        }
    )
    assert (
        disagreement(
            forecasts[["prob_home", "prob_draw", "prob_away"]].to_numpy(),
            implied_probabilities(
                matches.set_index("match_id").loc[["a", "b"], list(MARKET_COLUMNS)].to_numpy(float)
            ),
            forecasts["result"],
        )["n"].sum()
        == market_comparison(forecasts, matches)["n"].sum()
    )


def test_a_table_with_no_odds_column_is_reported_rather_than_guessed_at() -> None:
    forecasts = forecasts_with(["a"], [[0.5, 0.3, 0.2]])
    assert market_comparison(forecasts, pd.DataFrame({"match_id": ["a"]})).empty


def test_a_forecaster_that_is_not_in_the_frame_compares_nothing() -> None:
    forecasts = forecasts_with(["a"], [[0.5, 0.3, 0.2]])
    matches = pd.DataFrame(
        {"match_id": ["a"], "odds_home": [2.0], "odds_draw": [3.5], "odds_away": [4.0]}
    )
    assert market_comparison(forecasts, matches, name="nobody").empty


def test_only_the_named_forecaster_is_compared() -> None:
    """The frame can hold several; the card is about one."""
    forecasts = forecasts_with(["a", "b"], [[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]])
    other = forecasts.assign(forecaster="bookmaker")
    matches = pd.DataFrame(
        {
            "match_id": ["a", "b"],
            "odds_home": [2.0, 2.0],
            "odds_draw": [3.5, 3.5],
            "odds_away": [4.0, 4.0],
        }
    )
    assert market_comparison(pd.concat([forecasts, other]), matches)["n"].sum() == 2
