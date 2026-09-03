"""Elo.

The tests fall into two halves. The arithmetic half pins the formulas against
values that can be checked by hand — a 400-point gap is 10:1 odds, a two-goal
margin counts one and a half times. The causal half is the one that matters:
the row emitted for a match must be the state that match was played from, and
that is proved by the probes in :mod:`src.validation.temporal` rather than by
reading the loop.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.ratings.base import ELO_COLUMNS, KEY_COLUMN, RatingError, RatingModel
from src.ratings.elo import (
    DEFAULT,
    EloParameters,
    EloRatings,
    actual_scores,
    expected_score,
    fit,
    margin_multiplier,
    mean_squared_error,
)
from src.validation.temporal import outcome_independence, prefix_invariance
from tests.factories import canonical_frame, league_frame

SEASON = league_frame(seasons=["2020-21"])
TWO_SEASONS = league_frame(seasons=["2020-21", "2021-22"])


def two_matches(*, first: tuple[int, int], second: tuple[int, int]) -> pd.DataFrame:
    """The same two clubs meeting twice, so a rating change is traceable."""
    rows = []
    for index, (home_goals, away_goals) in enumerate((first, second)):
        rows.append(
            {
                "match_id": f"m{index}",
                "provider": "stub",
                "competition_id": "ENG_1",
                "country": "England",
                "competition": "Premier League",
                "tier": 1,
                "season": "2020-21",
                "date": pd.Timestamp("2020-08-01") + pd.Timedelta(index * 7, "D"),
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_team_id": "eng:alpha",
                "away_team_id": "eng:beta",
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": (
                    "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
                ),
            }
        )
    return canonical_frame(rows)


# ---- arithmetic -------------------------------------------------------------


def test_equal_ratings_favour_the_home_side() -> None:
    """45% of matches in the ingest are home wins against 28% away. A model
    without this term is wrong in one direction for every fixture."""
    assert expected_score(1500.0, 1500.0, DEFAULT) > 0.5


def test_home_advantage_can_be_switched_off() -> None:
    neutral = replace(DEFAULT, home_advantage=0.0)
    assert expected_score(1500.0, 1500.0, neutral) == pytest.approx(0.5)


def test_a_four_hundred_point_gap_is_ten_to_one() -> None:
    """The definition of the scale, checkable by hand."""
    neutral = replace(DEFAULT, home_advantage=0.0)
    assert expected_score(1900.0, 1500.0, neutral) == pytest.approx(10 / 11, abs=1e-9)


def test_the_expectation_is_monotone_in_the_rating_gap() -> None:
    scores = [expected_score(1500.0 + gap, 1500.0, DEFAULT) for gap in (-200, -50, 0, 50, 200)]
    assert scores == sorted(scores)


def test_the_expectation_stays_inside_zero_and_one() -> None:
    assert 0.0 < expected_score(1000.0, 3000.0, DEFAULT) < 1.0
    assert 0.0 < expected_score(3000.0, 1000.0, DEFAULT) < 1.0


@pytest.mark.parametrize(
    ("margin", "expected"),
    [(0, 1.0), (1, 1.0), (2, 1.5), (3, 1.75), (5, 2.0)],
)
def test_the_margin_scale_is_the_world_football_elo_one(margin: int, expected: float) -> None:
    """Flat to one goal, then growing slowly, so one thrashing cannot rewrite a
    season."""
    undamped = replace(DEFAULT, damping=1e12)
    assert margin_multiplier(margin, 0.0, undamped) == pytest.approx(expected, rel=1e-6)


def test_a_favourite_winning_counts_for_less_than_an_underdog_winning() -> None:
    """The autocorrelation correction. Strong teams beat weak teams heavily, so
    an undamped multiplier inflates the already-strong."""
    favourite = margin_multiplier(3, 300.0, DEFAULT)
    even = margin_multiplier(3, 0.0, DEFAULT)
    underdog = margin_multiplier(3, -300.0, DEFAULT)
    assert favourite < even < underdog


def test_a_draw_is_never_scaled_up() -> None:
    assert margin_multiplier(0, 0.0, DEFAULT) == pytest.approx(1.0)


# ---- the walk ---------------------------------------------------------------


def test_the_first_appearance_carries_the_prior_and_no_evidence() -> None:
    """1500 because we have never seen them, and 1500 because they are exactly
    average, are different facts — the played count is what separates them."""
    rated = EloRatings().rate(two_matches(first=(1, 0), second=(0, 0)))
    assert rated.iloc[0]["elo_home"] == DEFAULT.initial
    assert rated.iloc[0]["elo_away"] == DEFAULT.initial
    assert rated.iloc[0]["elo_home_played"] == 0
    assert rated.iloc[1]["elo_home_played"] == 1


def test_a_win_raises_the_winner_and_lowers_the_loser() -> None:
    rated = EloRatings().rate(two_matches(first=(3, 0), second=(0, 0)))
    assert rated.iloc[1]["elo_home"] > DEFAULT.initial
    assert rated.iloc[1]["elo_away"] < DEFAULT.initial


def test_the_pool_mean_is_conserved() -> None:
    """The update is zero-sum, which is what lets a rating be read as 'points
    above or below an average team' rather than as an arbitrary index."""
    rated = EloRatings().rate(TWO_SEASONS)
    total = rated["elo_home"].astype("float64") + rated["elo_away"].astype("float64")
    assert total.mean() == pytest.approx(2 * DEFAULT.initial, abs=1.0)


def test_a_draw_costs_the_favourite() -> None:
    """A draw against a weaker side is evidence against the favourite, and a
    rating that ignored it would drift upward on every fixture it failed to
    win."""
    strong_win = two_matches(first=(4, 0), second=(0, 0))
    rated = EloRatings().rate(strong_win)
    after_draw = EloRatings().rate(
        pd.concat([strong_win, strong_win.tail(1).assign(match_id="m2")], ignore_index=True)
    )
    assert after_draw.iloc[2]["elo_home"] < rated.iloc[1]["elo_home"]


def test_a_bigger_win_moves_the_rating_further() -> None:
    narrow = EloRatings().rate(two_matches(first=(1, 0), second=(0, 0)))
    thrashing = EloRatings().rate(two_matches(first=(5, 0), second=(0, 0)))
    assert thrashing.iloc[1]["elo_home"] > narrow.iloc[1]["elo_home"]


def test_season_carry_over_regresses_toward_the_prior() -> None:
    """Squads turn over in the summer, so a club that was strong two years ago
    starts closer to average than to its own peak."""
    matches = TWO_SEASONS
    strong = EloRatings(replace(DEFAULT, season_carry=1.0)).rate(matches)
    regressed = EloRatings(replace(DEFAULT, season_carry=0.5)).rate(matches)
    second_season = matches["season"].to_numpy() == "2021-22"
    gap_strong = (strong["elo_home"][second_season] - DEFAULT.initial).abs().mean()
    gap_regressed = (regressed["elo_home"][second_season] - DEFAULT.initial).abs().mean()
    assert gap_regressed < gap_strong


def test_carry_over_is_applied_once_per_team_per_season() -> None:
    """Detected from each team's own previous match rather than from a date,
    because a country's league and cup seasons do not change label on the same
    day — and a regression applied twice is invisible in the output."""
    matches = TWO_SEASONS
    rated = EloRatings(replace(DEFAULT, season_carry=0.0)).rate(matches)
    first_of_second_season = matches["season"].to_numpy() == "2021-22"
    opening = rated[first_of_second_season].head(1)
    assert float(opening["elo_home"].iloc[0]) == pytest.approx(DEFAULT.initial)


# ---- the contract -----------------------------------------------------------


def test_elo_satisfies_the_rating_protocol() -> None:
    assert isinstance(EloRatings(), RatingModel)


def test_the_output_shape_matches_the_schema() -> None:
    rated = EloRatings().rate(SEASON)
    assert list(rated.columns) == [KEY_COLUMN, *ELO_COLUMNS]
    assert {c: str(d) for c, d in rated.dtypes.items()} == {"match_id": "string", **ELO_COLUMNS}
    assert len(rated) == len(SEASON)
    assert list(rated[KEY_COLUMN]) == list(SEASON[KEY_COLUMN])


def test_an_unsorted_input_is_refused() -> None:
    """Sorting silently would hide that a caller's pipeline lost its ordering,
    and the ratings would look plausible and be computed from the future."""
    shuffled = SEASON.sort_values("match_id", ascending=False)
    with pytest.raises(RatingError, match="sorted by date"):
        EloRatings().rate(shuffled)


def test_an_empty_table_produces_an_empty_frame() -> None:
    rated = EloRatings().rate(canonical_frame([]))
    assert rated.empty
    assert list(rated.columns) == [KEY_COLUMN, *ELO_COLUMNS]


@pytest.mark.parametrize(
    "bad",
    [{"k": 0.0}, {"scale": -1.0}, {"damping": 0.0}, {"season_carry": 1.5}, {"season_carry": -0.1}],
)
def test_impossible_parameters_are_refused(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        EloParameters(**bad)  # type: ignore[arg-type]


# ---- causality --------------------------------------------------------------


def test_elo_is_prefix_invariant() -> None:
    """Truncate the history and the surviving ratings must not move."""
    assert prefix_invariance(EloRatings().rate, TWO_SEASONS, name="elo").ok


def test_elo_does_not_read_a_match_it_is_rating() -> None:
    """The probe prefix invariance cannot make: rewrite one scoreline and the
    row for that very match must be unchanged."""
    assert outcome_independence(EloRatings().rate, TWO_SEASONS, name="elo").ok


def test_rewriting_a_result_does_change_later_ratings() -> None:
    """The other half of the previous test. A rating that ignored results
    entirely would pass outcome independence trivially."""
    rewritten = TWO_SEASONS.copy()
    position = 10
    rewritten.iloc[position, rewritten.columns.get_loc("home_goals")] = 9
    rewritten.iloc[position, rewritten.columns.get_loc("result")] = "H"
    before = EloRatings().rate(TWO_SEASONS)
    after = EloRatings().rate(rewritten)
    assert not before.tail(1).equals(after.tail(1))


# ---- scoring and fitting ----------------------------------------------------


def test_actual_scores_are_one_half_and_zero() -> None:
    scores = actual_scores(two_matches(first=(2, 0), second=(1, 1)))
    assert list(scores) == [1.0, 0.5]


def test_home_advantage_beats_no_home_advantage_on_real_shaped_data() -> None:
    """The measurement that justifies the term, in miniature."""
    with_advantage = mean_squared_error(TWO_SEASONS, EloRatings())
    without = mean_squared_error(TWO_SEASONS, EloRatings(replace(DEFAULT, home_advantage=0.0)))
    assert with_advantage < without


def test_fit_returns_the_best_of_the_grid_it_was_given() -> None:
    parameters, error = fit(
        TWO_SEASONS, k_values=[10.0, 20.0], home_advantage_values=[80.0], season_carry_values=[1.0]
    )
    assert parameters.k in {10.0, 20.0}
    assert parameters.home_advantage == 80.0
    assert error == pytest.approx(mean_squared_error(TWO_SEASONS, EloRatings(parameters)))


def test_fit_is_deterministic() -> None:
    """A grid rather than an optimiser, so the answer is reproducible to the
    digit rather than to a random seed."""
    first = fit(SEASON, k_values=[10.0, 20.0, 30.0], season_carry_values=[1.0])
    second = fit(SEASON, k_values=[10.0, 20.0, 30.0], season_carry_values=[1.0])
    assert first == second


def test_fit_refuses_an_unsorted_window() -> None:
    with pytest.raises(RatingError):
        fit(SEASON.sort_values("match_id", ascending=False))
