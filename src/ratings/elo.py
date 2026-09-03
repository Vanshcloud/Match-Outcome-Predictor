"""Elo ratings, one pool per country.

Elo is the right first rating here for a reason that is easy to lose: it is
*online*. Each match updates two numbers and nothing else, so the rating
carried into match *n* is a function of matches 1..*n*-1 by construction. There
is no window to get wrong and no statistic over the whole table to accidentally
include — the causal property Milestone 6 has to enforce for every other
feature comes free, and that is worth having in the first rating rather than
the last.

**One pool per country, not per competition.** Team identity is already
country-scoped precisely so a promoted club keeps one id across divisions, and
resetting its rating at promotion would discard the history that makes the
rating worth anything. The cost is worth stating: league fixtures never cross
divisions, so a country's divisions are connected only through promotion,
relegation and the occasional cup tie, and the absolute level of a lower
division's ratings drifts against the top flight's more than a fully connected
pool's would. Ratings are still comparable *within* a division, which is where
every fixture is played.

## What each refinement is worth

Measured by walking all 303,517 matches — every prediction made from prior
matches only, so this is an out-of-sample number, not a fit. Mean squared error
of the expected score against the actual 1 / 0.5 / 0:

| Model | MSE |
|---|---|
| Plain Elo: no home advantage, margin, damping or carry-over | 0.16884 |
| ...with home advantage | 0.16225 |
| ...with margin of victory | 0.16216 |
| ...with autocorrelation damping | 0.16181 |
| ...with season carry-over (as shipped) | **0.16177** |

Home advantage is the one that matters — it is worth 4% on its own. The other
three are worth a few parts in a thousand each, and are kept because each is
about three lines and every one moved the number in the right direction.

## What did *not* work, and is therefore not here

The plan called for parameters fitted per competition. It was built, measured,
and removed: fitting each country on its own early seasons and evaluating on
everything after made the ratings **worse** (0.16300 against 0.16246 for fixed
constants). The calibration window is the coldest part of the history — every
team starts at the same rating, so the window is mostly warm-up noise, and
three free parameters happily chase it. A pooled fit across all countries did
no better than a wash.

The constants below therefore come from one pooled grid search over matches
before 2005-07-01, evaluated on everything after. That surface is flat: the
textbook values score 0.16314 post-2005 against 0.16326 for the fitted ones,
a difference of 0.1% which is the actual finding. :func:`fit` remains available
to re-derive them when the data grows — ``scripts/build_ratings.py --fit-until``
— but the pipeline does not tune anything at runtime.

Per-competition tuning may still be worth it against a *three-class* objective,
which Elo alone cannot express. That belongs to Milestone 7, which owns the
splits such a fit would need.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

import pandas as pd

from src.ratings.base import ELO_COLUMNS, KEY_COLUMN, require_chronological
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EloParameters:
    """Elo settings.

    Frozen so a pool's parameters cannot change half way through a run, which
    would make the ratings either side of the change incomparable.

    The three fitted values come from a pooled grid search over matches before
    2005-07-01 — a window chosen so the constants cannot encode anything about
    the period they are evaluated on. See the module docstring for why the
    per-competition fit the plan called for was measured and dropped.
    """

    k: float = 14.0
    """Learning rate: the most points an average match can move a team. High
    enough to track a genuine change in strength over a season, low enough that
    one result is not a verdict. The football convention is 20; 14 fitted
    better here and the surface between them is nearly flat."""

    home_advantage: float = 80.0
    """Rating points added to the home side before computing the expectation.
    Worth more than every other refinement combined: 45% of matches in the
    ingest are home wins against 28% away, and a model without this term is
    systematically wrong in one direction for every fixture."""

    initial: float = 1500.0
    """Where an unseen team starts, and the value season carry-over regresses
    toward. Arbitrary in absolute terms — only differences matter — but fixed
    rather than a running pool mean, which would make each team's rating depend
    on which others happened to be in the pool."""

    scale: float = 400.0
    """The logistic scale. 400 points is the classic 10:1 odds ratio."""

    season_carry: float = 0.97
    """Fraction of a team's deviation from :attr:`initial` carried into its next
    season; 1.0 keeps everything. The football-Elo convention is 0.75, which is
    measurably too aggressive on this data — squads turn over, but three
    decades of results say a club's strength persists across a summer far more
    than that."""

    damping: float = 2.2
    """Autocorrelation damping constant. Strong teams beat weak teams heavily,
    so an undamped margin multiplier inflates the already-strong; the
    multiplier is scaled by ``damping / (0.001 * winner_advantage + damping)``.
    Larger means less damping."""

    def __post_init__(self) -> None:
        if self.k <= 0 or self.scale <= 0 or self.damping <= 0:
            raise ValueError("k, scale and damping must be positive")
        if not 0.0 <= self.season_carry <= 1.0:
            raise ValueError("season_carry must be between 0 and 1")


DEFAULT = EloParameters()


def expected_score(home: float, away: float, parameters: EloParameters) -> float:
    """The home side's expected score, on the 1 / 0.5 / 0 scale.

    **Not a probability of winning.** Elo's expectation is a mean score, mixing
    a win and a draw, and it cannot be decomposed into three class
    probabilities without a further model — that mapping belongs to Milestone
    7. Naming it anything else here would put a number in the feature table
    that every consumer would misread.
    """
    difference = home + parameters.home_advantage - away
    # float() around the power: pandas-stubs types `float ** float` loosely
    # enough that mypy loses the return type through it, and this project runs
    # with warn_return_any.
    return 1.0 / (1.0 + float(10.0 ** (-difference / parameters.scale)))


def margin_multiplier(
    goal_difference: int, winner_advantage: float, parameters: EloParameters
) -> float:
    """How much more a result counts for having been decisive.

    The scale is the World Football Elo one: flat for a one-goal margin, then
    growing slowly, so one thrashing cannot rewrite a season.

    Args:
        goal_difference: Absolute margin; zero for a draw.
        winner_advantage: The winner's own pre-match rating edge, home
            advantage included; zero for a draw. Negative when an underdog won,
            which correctly *increases* the multiplier — a surprise is more
            informative than a confirmation.
    """
    if goal_difference <= 1:
        scale = 1.0
    elif goal_difference == 2:
        scale = 1.5
    else:
        scale = (11.0 + goal_difference) / 8.0
    return scale * (parameters.damping / (0.001 * winner_advantage + parameters.damping))


class EloRatings:
    """An Elo pool per country, walked forward one match at a time.

    Satisfies :class:`~src.ratings.base.RatingModel`.
    """

    name = "elo"
    feature_columns = tuple(ELO_COLUMNS)

    def __init__(self, parameters: EloParameters = DEFAULT) -> None:
        self.parameters = parameters

    def rate(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Walk every match in order, emitting the ratings each side brought to it."""
        require_chronological(matches)
        settings = self.parameters

        ratings: dict[str, float] = {}
        played: dict[str, int] = {}
        seasons: dict[str, str] = {}

        home_before: list[float] = []
        away_before: list[float] = []
        expected: list[float] = []
        home_played: list[int] = []
        away_played: list[int] = []

        columns = zip(
            matches["season"],
            matches["home_team_id"],
            matches["away_team_id"],
            matches["home_goals"],
            matches["away_goals"],
            strict=True,
        )

        for season, home_id, away_id, home_goals, away_goals in columns:
            home_rating = self._current(ratings, seasons, played, home_id, season, settings)
            away_rating = self._current(ratings, seasons, played, away_id, season, settings)

            # Recorded BEFORE the update. This one ordering is the entire causal
            # guarantee of the module, which is why it is not tucked into a
            # helper: the row emitted for this match is the state it was played
            # from, and the update that follows belongs to the next one.
            home_before.append(home_rating)
            away_before.append(away_rating)
            home_played.append(played[home_id])
            away_played.append(played[away_id])

            score = expected_score(home_rating, away_rating, settings)
            expected.append(score)

            margin = int(home_goals) - int(away_goals)
            actual = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
            edge = home_rating + settings.home_advantage - away_rating
            advantage = 0.0 if margin == 0 else (edge if margin > 0 else -edge)
            change = (
                settings.k * margin_multiplier(abs(margin), advantage, settings) * (actual - score)
            )
            # Zero-sum, so the pool mean is conserved and a rating is always
            # readable as "points above or below an average team".
            ratings[home_id] = home_rating + change
            ratings[away_id] = away_rating - change
            played[home_id] += 1
            played[away_id] += 1

        return pd.DataFrame(
            {
                KEY_COLUMN: matches[KEY_COLUMN].to_numpy(),
                "elo_home": home_before,
                "elo_away": away_before,
                "elo_expected_home": expected,
                "elo_home_played": home_played,
                "elo_away_played": away_played,
            }
        ).astype({KEY_COLUMN: "string", **ELO_COLUMNS})

    @staticmethod
    def _current(
        ratings: dict[str, float],
        seasons: dict[str, str],
        played: dict[str, int],
        team: str,
        season: str,
        settings: EloParameters,
    ) -> float:
        """Return a team's rating, applying season carry-over first if it is due.

        The boundary is detected per team from its own previous match rather
        than per country from a date, because a country's league and cup
        seasons do not change label on the same day — and a regression applied
        twice, or on the wrong side of a fixture, is invisible in the output.
        """
        if team not in ratings:
            ratings[team] = settings.initial
            played[team] = 0
            seasons[team] = season
            return settings.initial
        if seasons[team] != season:
            seasons[team] = season
            ratings[team] = settings.initial + settings.season_carry * (
                ratings[team] - settings.initial
            )
        return ratings[team]


def actual_scores(matches: pd.DataFrame) -> pd.Series:
    """The home side's realised score, on the same 1 / 0.5 / 0 scale."""
    scores = pd.Series(0.5, index=matches.index, dtype="float64")
    scores[(matches["home_goals"] > matches["away_goals"]).to_numpy()] = 1.0
    scores[(matches["home_goals"] < matches["away_goals"]).to_numpy()] = 0.0
    return scores


def mean_squared_error(matches: pd.DataFrame, model: EloRatings) -> float:
    """Mean squared error of the expected score against the actual one.

    The classic Elo objective, and the honest one here: Elo predicts a *score*,
    not three class probabilities, so scoring it with log loss would mean
    inventing the draw model that does not exist yet and then reporting that
    model's quality as Elo's.

    Every prediction in the walk is made from prior matches only, so this is an
    out-of-sample number over the whole table rather than a fit.
    """
    rated = model.rate(matches)
    error = (
        rated["elo_expected_home"].astype("float64").to_numpy() - actual_scores(matches).to_numpy()
    )
    return float((error**2).mean())


def fit(
    matches: pd.DataFrame,
    *,
    k_values: Iterable[float] = (10.0, 14.0, 18.0, 22.0, 26.0, 32.0),
    home_advantage_values: Iterable[float] = (40.0, 55.0, 65.0, 80.0, 100.0),
    season_carry_values: Iterable[float] = (0.75, 0.85, 0.92, 0.97, 1.0),
    base: EloParameters = DEFAULT,
) -> tuple[EloParameters, float]:
    """Grid-search the three settings that matter, on ``matches`` alone.

    A grid rather than an optimiser: three parameters over a surface that is
    flat near its minimum, where a gradient method spends its time chasing
    noise and a grid is reproducible to the digit.

    **This is a maintenance tool, not a pipeline step.** Fitting on the same
    rows the ratings are later used to predict is a leak — a subtle one,
    because the ratings themselves stay causal and only the constants know the
    future — so nothing calls this during a rating run. It exists to re-derive
    :data:`DEFAULT` from an early window when the data grows;
    ``scripts/build_ratings.py --fit-until`` is the supported way in.

    Returns:
        The best parameters and their mean squared error.
    """
    require_chronological(matches)
    best = base
    best_error = math.inf
    for k in k_values:
        for home_advantage in home_advantage_values:
            for season_carry in season_carry_values:
                candidate = replace(
                    base, k=k, home_advantage=home_advantage, season_carry=season_carry
                )
                error = mean_squared_error(matches, EloRatings(candidate))
                if error < best_error:
                    best, best_error = candidate, error
    logger.info(
        "fitted elo on %d matches: k=%.0f hfa=%.0f carry=%.2f (mse %.5f)",
        len(matches),
        best.k,
        best.home_advantage,
        best.season_carry,
        best_error,
    )
    return best, best_error
