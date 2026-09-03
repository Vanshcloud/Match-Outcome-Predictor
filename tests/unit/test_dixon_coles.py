"""Dixon-Coles.

Three things are worth testing here and the third is the point.

The arithmetic — the low-score correction and the probability grid — is checked
against values derivable by hand. The *estimation* is checked by simulating
matches from known strengths and asking whether the fit recovers them; a
likelihood with a sign error still optimises happily and produces confident
nonsense, and only a recovery test notices.

The causality is checked with the probes, and with one case they do not reach:
two matches on the same day must not inform each other. A full Saturday
programme is one round, and a model fitted on the 3pm results to predict the
5.30 kick-off would look excellent and be useless.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.ratings.base import DIXON_COLES_COLUMNS, KEY_COLUMN, RatingError, RatingModel
from src.ratings.dixon_coles import (
    DixonColesParameters,
    DixonColesRatings,
    Strengths,
    fit_window,
    outcome_probabilities,
    tau,
)
from src.validation.temporal import outcome_independence, prefix_invariance
from tests.factories import canonical_frame, league_frame

# Small windows and an early first fit, so a test exercises the refit machinery
# in seconds rather than in the minutes a real competition takes.
FAST = DixonColesParameters(window_days=540, refit_days=60, min_matches=60, min_teams=6)

# Recovery is a question about the estimator, not about the decay policy. With
# the shipped half-life of ~107 days a six-season simulation puts almost all
# its weight on the last few months, so the estimate is noisy for a reason that
# has nothing to do with whether the likelihood is right.
UNWEIGHTED = replace(FAST, decay=0.0, window_days=100_000)

LEAGUE = league_frame(seasons=["2018-19", "2019-20", "2020-21", "2021-22"], teams=10)


def simulate(
    *,
    seasons: int = 6,
    teams: int = 10,
    home_advantage: float = 0.3,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """Matches drawn from known strengths, for the recovery test.

    Independent Poissons, deliberately: the estimator has to find the strengths
    that generated the data even when the low-score correction it also fits has
    nothing to find. A generator that used the correction too would let a sign
    error in one term cancel a sign error in the other.
    """
    rng = np.random.default_rng(seed)
    names = [f"Team {index:02d}" for index in range(teams)]
    attack = {
        name: float(value)
        for name, value in zip(names, np.linspace(0.35, -0.35, teams), strict=True)
    }
    defence = {
        name: float(value)
        for name, value in zip(names, np.linspace(0.25, -0.25, teams), strict=True)
    }

    records = []
    day = pd.Timestamp("2015-08-01")
    for season in range(seasons):
        label = f"{2015 + season}-{(2016 + season) % 100:02d}"
        fixtures = [(h, a) for h in names for a in names if h != a]
        rng.shuffle(fixtures)  # type: ignore[arg-type]
        for index, (home, away) in enumerate(fixtures):
            home_rate = float(np.exp(attack[home] + defence[away] + home_advantage))
            away_rate = float(np.exp(attack[away] + defence[home]))
            home_goals = int(rng.poisson(home_rate))
            away_goals = int(rng.poisson(away_rate))
            records.append(
                {
                    "match_id": f"s{season}-{index:04d}",
                    "provider": "sim",
                    "competition_id": "ENG_1",
                    "country": "England",
                    "competition": "Premier League",
                    "tier": 1,
                    "season": label,
                    "date": day + pd.Timedelta(season * 300 + index * 300 // len(fixtures), "D"),
                    "home_team": home,
                    "away_team": away,
                    "home_team_id": f"eng:{home.lower().replace(' ', '-')}",
                    "away_team_id": f"eng:{away.lower().replace(' ', '-')}",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": (
                        "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
                    ),
                }
            )
    frame = canonical_frame(records).sort_values("date", kind="stable").reset_index(drop=True)
    keyed = {f"eng:{name.lower().replace(' ', '-')}": value for name, value in attack.items()}
    keyed_defence = {
        f"eng:{name.lower().replace(' ', '-')}": value for name, value in defence.items()
    }
    return frame, keyed, keyed_defence


# ---- the low-score correction -----------------------------------------------


def test_a_zero_rho_corrects_nothing() -> None:
    goals = np.array([0, 0, 1, 1, 3])
    rates = np.full(5, 1.4)
    assert np.allclose(tau(goals, goals[::-1], rates, rates, 0.0), 1.0)


def test_only_the_four_low_scorelines_are_touched() -> None:
    """One parameter rather than a full joint distribution, which is the whole
    reason the correction is affordable."""
    home = np.array([0, 0, 1, 1, 2, 3, 0])
    away = np.array([0, 1, 0, 1, 2, 0, 4])
    rates = np.full(len(home), 1.5)
    corrected = tau(home, away, rates, rates, 0.1)
    assert np.allclose(corrected[4:], 1.0)
    assert not np.allclose(corrected[:4], 1.0)


@pytest.mark.parametrize(
    ("home_goals", "away_goals", "expected"),
    [(0, 0, 1 - 2.0 * 1.5 * 0.1), (0, 1, 1 + 2.0 * 0.1), (1, 0, 1 + 1.5 * 0.1), (1, 1, 1 - 0.1)],
)
def test_each_corrected_cell_matches_the_paper(
    home_goals: int, away_goals: int, expected: float
) -> None:
    value = tau(
        np.array([home_goals]), np.array([away_goals]), np.array([2.0]), np.array([1.5]), 0.1
    )
    assert value[0] == pytest.approx(expected)


def test_the_correction_never_goes_non_positive() -> None:
    """An optimiser that walks into log(0) reports success on whatever it was
    holding at the time."""
    corrected = tau(np.array([0]), np.array([0]), np.array([9.0]), np.array([9.0]), 0.2)
    assert corrected[0] > 0


# ---- the probability grid ---------------------------------------------------


def test_the_three_probabilities_are_a_distribution() -> None:
    for home_rate, away_rate in ((1.5, 1.2), (0.4, 3.0), (2.8, 2.8)):
        probabilities = outcome_probabilities(home_rate, away_rate, -0.1)
        assert sum(probabilities) == pytest.approx(1.0)
        assert all(0.0 < value < 1.0 for value in probabilities)


def test_equal_rates_give_equal_win_probabilities() -> None:
    home_win, _, away_win = outcome_probabilities(1.4, 1.4, 0.0)
    assert home_win == pytest.approx(away_win)


def test_a_higher_home_rate_raises_the_home_probability() -> None:
    modest = outcome_probabilities(1.4, 1.4, 0.0)[0]
    strong = outcome_probabilities(2.4, 1.4, 0.0)[0]
    assert strong > modest


def test_a_negative_rho_produces_more_draws() -> None:
    """The direction the correction exists to fix: independent Poissons
    under-count 0-0 and 1-1, and the fitted rho is negative on real football."""
    plain = outcome_probabilities(1.3, 1.1, 0.0)[1]
    corrected = outcome_probabilities(1.3, 1.1, -0.12)[1]
    assert corrected > plain


def test_the_grid_is_renormalised() -> None:
    """The correction does not preserve mass and the truncation loses a little
    more, so without this the three numbers would only nearly be a
    distribution."""
    assert sum(outcome_probabilities(1.5, 1.5, -0.15, max_goals=3)) == pytest.approx(1.0)


# ---- estimation -------------------------------------------------------------


def test_the_fit_recovers_the_strengths_that_generated_the_data() -> None:
    """The test that catches a likelihood with a sign error, which optimises
    just as happily as a correct one."""
    matches, attack, defence = simulate()
    fitted = fit_window(matches, matches["date"].max() + pd.Timedelta(1, "D"), UNWEIGHTED)
    assert fitted is not None

    teams = sorted(attack)
    truth = [attack[team] for team in teams]
    estimate = [fitted.attack[team] for team in teams]
    assert np.corrcoef(truth, estimate)[0, 1] > 0.9

    truth_defence = [defence[team] for team in teams]
    estimate_defence = [fitted.defence[team] for team in teams]
    assert np.corrcoef(truth_defence, estimate_defence)[0, 1] > 0.9


def test_the_fit_recovers_home_advantage() -> None:
    matches, _, _ = simulate(home_advantage=0.3)
    fitted = fit_window(matches, matches["date"].max() + pd.Timedelta(1, "D"), UNWEIGHTED)
    assert fitted is not None
    assert fitted.home_advantage == pytest.approx(0.3, abs=0.12)


def test_time_decay_prefers_the_recent_past() -> None:
    """A fit that weighs a match from three years ago as heavily as last week's
    is describing a squad that no longer exists."""
    matches, _, _ = simulate(seasons=6)
    as_of = matches["date"].max() + pd.Timedelta(1, "D")
    flat = fit_window(matches, as_of, UNWEIGHTED)
    decayed = fit_window(matches, as_of, replace(UNWEIGHTED, decay=0.02))
    assert flat is not None and decayed is not None
    assert flat.attack != decayed.attack


def test_a_window_with_too_few_matches_produces_nothing() -> None:
    """Below this the window cannot identify two strengths per team, and the
    fit reports confident nonsense rather than failing."""
    matches, _, _ = simulate(seasons=1)
    assert fit_window(matches.head(20), matches["date"].max(), FAST) is None


def test_a_window_with_too_few_teams_produces_nothing() -> None:
    matches, _, _ = simulate(teams=10)
    # Both sides, not just the home one: a filter on home_team_id alone leaves
    # every away club in the window and the count is unchanged.
    few = {"eng:team-00", "eng:team-01", "eng:team-02"}
    small = matches[matches["home_team_id"].isin(few) & matches["away_team_id"].isin(few)]
    assert fit_window(small, matches["date"].max(), replace(UNWEIGHTED, min_matches=1)) is None


def test_a_warm_start_lands_in_the_same_place() -> None:
    """Warm starting is what makes a refit every thirty days affordable; it
    would not be worth having if it changed the answer."""
    matches, _, _ = simulate()
    as_of = matches["date"].max() + pd.Timedelta(1, "D")
    cold = fit_window(matches, as_of, UNWEIGHTED)
    assert cold is not None
    warm = fit_window(matches, as_of, UNWEIGHTED, previous=cold)
    assert warm is not None
    for team in cold.attack:
        assert warm.attack[team] == pytest.approx(cold.attack[team], abs=0.05)


def test_rates_are_unavailable_for_an_unseen_team() -> None:
    strengths = Strengths(
        attack={"a": 0.1}, defence={"a": 0.0}, home_advantage=0.3, rho=-0.1, matches=200
    )
    assert strengths.rates("a", "b") is None
    assert strengths.rates("a", "a") is not None


@pytest.mark.parametrize(
    "bad", [{"window_days": 0}, {"refit_days": -1}, {"decay": -0.1}, {"max_rho": 1.0}]
)
def test_impossible_parameters_are_refused(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        DixonColesParameters(**bad)  # type: ignore[arg-type]


# ---- the walk ---------------------------------------------------------------


def test_dixon_coles_satisfies_the_rating_protocol() -> None:
    assert isinstance(DixonColesRatings(), RatingModel)


def test_the_output_shape_matches_the_schema() -> None:
    rated = DixonColesRatings(FAST).rate(LEAGUE)
    assert list(rated.columns) == [KEY_COLUMN, *DIXON_COLES_COLUMNS]
    assert {c: str(d) for c, d in rated.dtypes.items()} == {
        "match_id": "string",
        **DIXON_COLES_COLUMNS,
    }
    assert list(rated[KEY_COLUMN]) == list(LEAGUE[KEY_COLUMN])


def test_matches_before_the_first_viable_fit_are_null() -> None:
    """A number nobody could have had is worse than no number."""
    rated = DixonColesRatings(FAST).rate(LEAGUE)
    assert rated["dc_prob_home"].head(20).isna().all()
    assert rated["dc_prob_home"].notna().any()


def test_probabilities_sum_to_one_wherever_they_exist() -> None:
    rated = DixonColesRatings(FAST).rate(LEAGUE).dropna()
    total = (
        rated["dc_prob_home"].astype("float64")
        + rated["dc_prob_draw"].astype("float64")
        + rated["dc_prob_away"].astype("float64")
    )
    assert np.allclose(total.to_numpy(), 1.0)


def test_competitions_are_fitted_independently() -> None:
    """Attack and defence are only identifiable among teams that play each
    other, and two leagues that never meet share no scale."""
    first = league_frame(seasons=["2018-19", "2019-20", "2020-21"], teams=10)
    second = league_frame(
        competition_id="ESP_1",
        country="Spain",
        name="La Liga",
        seasons=["2018-19", "2019-20", "2020-21"],
        teams=10,
    )
    together = (
        pd.concat([first, second], ignore_index=True)
        .sort_values(["date", "competition_id", "match_id"], kind="stable")
        .reset_index(drop=True)
    )
    combined = DixonColesRatings(FAST).rate(together).set_index(KEY_COLUMN)
    alone = DixonColesRatings(FAST).rate(first).set_index(KEY_COLUMN)
    assert combined.loc[alone.index].equals(alone)


def test_the_row_order_follows_the_input_not_the_grouping() -> None:
    """Grouping by competition is an implementation detail of the fitting, and
    a rating table whose order depended on it would silently stop lining up
    with the canonical table."""
    first = league_frame(seasons=["2018-19", "2019-20"], teams=8)
    second = league_frame(
        competition_id="ESP_1",
        country="Spain",
        name="La Liga",
        seasons=["2018-19", "2019-20"],
        teams=8,
    )
    together = (
        pd.concat([first, second], ignore_index=True)
        .sort_values(["date", "competition_id", "match_id"], kind="stable")
        .reset_index(drop=True)
    )
    rated = DixonColesRatings(FAST).rate(together)
    assert list(rated[KEY_COLUMN]) == list(together[KEY_COLUMN])


def test_an_unsorted_input_is_refused() -> None:
    with pytest.raises(RatingError, match="sorted by date"):
        DixonColesRatings(FAST).rate(LEAGUE.sort_values("match_id", ascending=False))


def test_an_empty_table_produces_an_empty_frame() -> None:
    rated = DixonColesRatings(FAST).rate(canonical_frame([]))
    assert rated.empty
    assert list(rated.columns) == [KEY_COLUMN, *DIXON_COLES_COLUMNS]


# ---- causality --------------------------------------------------------------


def test_dixon_coles_is_prefix_invariant() -> None:
    assert prefix_invariance(DixonColesRatings(FAST).rate, LEAGUE, name="dixon-coles").ok


def test_dixon_coles_does_not_read_a_match_it_is_rating() -> None:
    assert outcome_independence(DixonColesRatings(FAST).rate, LEAGUE, name="dixon-coles").ok


def test_matches_on_the_same_day_do_not_inform_each_other() -> None:
    """The case the generic probes cannot reach.

    A full Saturday programme is one round. A refit window that used ``<=`` its
    own date would fit on the 3pm results and predict the 5.30 kick-off from
    them — a leak that is invisible to truncation, because both matches survive
    or neither does.
    """
    matches = LEAGUE.copy()
    round_day = matches["date"].iloc[-6]
    matches.loc[matches.index[-6:], "date"] = round_day
    matches = matches.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)

    baseline = DixonColesRatings(FAST).rate(matches).set_index(KEY_COLUMN)

    rewritten = matches.copy()
    rewritten.loc[rewritten.index[-6], "home_goals"] = 9
    rewritten.loc[rewritten.index[-6], "result"] = "H"
    after = DixonColesRatings(FAST).rate(rewritten).set_index(KEY_COLUMN)

    same_day = matches[matches["date"] == round_day][KEY_COLUMN]
    assert baseline.loc[same_day].equals(after.loc[same_day])
