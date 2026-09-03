"""Dixon-Coles: a bivariate Poisson goal model, refitted as the season moves.

Elo answers "who is stronger". This answers "how many goals, to whom, with what
probability" — and unlike Elo it produces a genuine H/D/A distribution, which
makes it the first thing in this project that can be scored with log loss and
compared against a bookmaker.

The model, from Dixon and Coles (1997):

    lambda = exp(attack_home + defence_away + home_advantage)
    mu     = exp(attack_away + defence_home)

Home goals are Poisson(lambda) and away goals Poisson(mu), *almost*
independently. Two departures from a plain double Poisson do the real work:

**The low-score correction.** Independent Poissons get 0-0, 1-0, 0-1 and 1-1
wrong — there are more of them in football than independence predicts, because
a match that is level late stops being two teams scoring at fixed rates. A
single parameter ``rho`` reweights exactly those four cells and leaves the rest
alone. It is worth having precisely because those four are a large share of all
football scorelines.

**Time decay.** A fit that weighs a match from three years ago as heavily as
one from last week is describing a squad that no longer exists. Each match in
the window is weighted ``exp(-decay * days_before_the_fit)``.

## Causality

The expensive property, and the one that shapes the whole design. Strengths are
refitted every :attr:`~DixonColesParameters.refit_days`, on a window that ends
**strictly before** the date of the match that triggered the refit. Matches on
the same day therefore cannot inform each other, which matters more than it
sounds: a full Saturday programme is one round, and a model that fit on the
3pm results to predict the 5.30 kick-off would look excellent and be useless.

Everything before a competition's first viable fit is null. A team promoted
into a division mid-window is null until a refit has seen it. Both are honest
answers rather than a number nobody could have had.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from src.ratings.base import DIXON_COLES_COLUMNS, KEY_COLUMN, require_chronological
from src.utils.logging import get_logger

logger = get_logger(__name__)

MAX_GOALS = 10
"""Goals per side in the probability grid. The ingest's maximum is 13 and the
tail beyond 10 carries less than 1e-6 of the mass at any plausible rate; the
grid is renormalised afterwards, so truncation costs nothing but arithmetic."""

_TAU_FLOOR = 1e-10
"""The low-score correction can drive a cell to zero or below at extreme rho.
Clamped so the log-likelihood stays finite — an optimiser that walks into a
NaN reports success on whatever it was holding at the time."""

_CENTRING_PENALTY = 1e3
"""The likelihood is invariant to adding a constant to every attack and
subtracting it from every defence, which leaves one flat direction for the
optimiser to wander along. A small penalty on the mean attack pins it without
measurably distorting the fit — cheaper and far less error-prone than
re-parameterising to T-1 free strengths and reconstructing the last."""


@dataclass(frozen=True, slots=True)
class DixonColesParameters:
    """How the model is fitted, and how often."""

    window_days: int = 1095
    """How much history each fit sees. Three seasons rather than the two the
    literature usually uses: measured, a longer window keeps winning, because
    the time decay already handles recency and a wider window is what makes the
    strengths of rarely-meeting teams identifiable. Beyond three seasons the
    gain is 0.0006 of log loss for a third more arithmetic."""

    refit_days: int = 60
    """How often strengths are refitted. Measured against 14, 30 and 90; the
    differences are within noise on subsets that differ by 2%, and this halves
    the number of fits. When a metric cannot tell two options apart, runtime
    can."""

    decay: float = 0.002
    """Exponential time-decay rate per day: a half-life of 347 days, about one
    season. Measured against 0 (no decay), 0.003, 0.0065 and 0.012. The
    conventional Dixon-Coles half-life of roughly half a year is too fast for
    this data — and *some* decay beats none, so the parameter earns its
    place."""

    min_matches: int = 150
    """Below this the window cannot identify two strengths per team, and the
    fit reports confident nonsense rather than failing."""

    min_teams: int = 6
    max_rho: float = 0.2
    """Bound on the low-score correction. Beyond this the corrected cells go
    negative at ordinary scoring rates."""

    max_iterations: int = 200

    def __post_init__(self) -> None:
        if self.window_days <= 0 or self.refit_days <= 0:
            raise ValueError("window_days and refit_days must be positive")
        if self.decay < 0:
            raise ValueError("decay must not be negative")
        if not 0.0 < self.max_rho < 1.0:
            raise ValueError("max_rho must lie strictly between 0 and 1")


DEFAULT = DixonColesParameters()


@dataclass(frozen=True, slots=True)
class Strengths:
    """One competition's fitted state, as of a date."""

    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    rho: float

    def rates(self, home: str, away: str) -> tuple[float, float] | None:
        """Expected goals for each side, or ``None`` if either team is unseen."""
        if home not in self.attack or away not in self.attack:
            return None
        home_rate = np.exp(self.attack[home] + self.defence[away] + self.home_advantage)
        away_rate = np.exp(self.attack[away] + self.defence[home])
        return float(home_rate), float(away_rate)


def tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_rate: np.ndarray,
    away_rate: np.ndarray,
    rho: float,
) -> np.ndarray:
    """The Dixon-Coles low-score correction, over four scorelines only.

    Independent Poissons under-count 0-0, 1-0, 0-1 and 1-1. This reweights
    exactly those and leaves every other scoreline at 1, which is what makes it
    one parameter rather than a full joint distribution.
    """
    correction = np.ones_like(home_rate)
    goalless = (home_goals == 0) & (away_goals == 0)
    away_one = (home_goals == 0) & (away_goals == 1)
    home_one = (home_goals == 1) & (away_goals == 0)
    one_each = (home_goals == 1) & (away_goals == 1)
    correction[goalless] = 1.0 - home_rate[goalless] * away_rate[goalless] * rho
    correction[away_one] = 1.0 + home_rate[away_one] * rho
    correction[home_one] = 1.0 + away_rate[home_one] * rho
    correction[one_each] = 1.0 - rho
    clamped: np.ndarray = np.clip(correction, _TAU_FLOOR, None)
    return clamped


def outcome_probabilities(
    home_rate: float, away_rate: float, rho: float, *, max_goals: int = MAX_GOALS
) -> tuple[float, float, float]:
    """Home / draw / away probabilities from a pair of rates and the correction.

    The grid is renormalised: the correction does not preserve total mass, and
    truncating at ``max_goals`` drops a little more. Renormalising is what makes
    the three numbers a distribution rather than three numbers that nearly are.
    """
    goals = np.arange(max_goals + 1)
    home_pmf = np.exp(goals * np.log(home_rate) - home_rate - gammaln(goals + 1))
    away_pmf = np.exp(goals * np.log(away_rate) - away_rate - gammaln(goals + 1))
    joint = np.outer(home_pmf, away_pmf)

    joint[0, 0] *= 1.0 - home_rate * away_rate * rho
    joint[0, 1] *= 1.0 + home_rate * rho
    joint[1, 0] *= 1.0 + away_rate * rho
    joint[1, 1] *= 1.0 - rho
    joint = np.clip(joint, 0.0, None)
    joint /= joint.sum()

    home_win = float(np.tril(joint, -1).sum())
    draw = float(np.trace(joint))
    return home_win, draw, float(1.0 - home_win - draw)


@dataclass
class _Window:
    """The arrays one fit works over. Built once per refit, not per iteration."""

    home_index: np.ndarray
    away_index: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    weights: np.ndarray
    teams: list[str]
    log_factorial_home: np.ndarray = field(init=False)
    log_factorial_away: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # Constant across every optimiser iteration, so computed once. With
        # ~40 free parameters a numeric gradient costs 41 evaluations per step,
        # and these two terms would otherwise be recomputed in all of them.
        self.log_factorial_home = gammaln(self.home_goals + 1)
        self.log_factorial_away = gammaln(self.away_goals + 1)


def _negative_log_likelihood(theta: np.ndarray, window: _Window) -> float:
    """Weighted negative log-likelihood of one window under ``theta``."""
    count = len(window.teams)
    attack = theta[:count]
    defence = theta[count : 2 * count]
    home_advantage = theta[-2]
    rho = float(theta[-1])

    log_home = attack[window.home_index] + defence[window.away_index] + home_advantage
    log_away = attack[window.away_index] + defence[window.home_index]
    home_rate = np.exp(log_home)
    away_rate = np.exp(log_away)

    log_likelihood = (
        window.home_goals * log_home
        - home_rate
        - window.log_factorial_home
        + window.away_goals * log_away
        - away_rate
        - window.log_factorial_away
        + np.log(tau(window.home_goals, window.away_goals, home_rate, away_rate, rho))
    )
    penalty = _CENTRING_PENALTY * float(attack.mean()) ** 2
    return -float((window.weights * log_likelihood).sum()) + penalty


def fit_window(
    window: pd.DataFrame,
    as_of: pd.Timestamp,
    parameters: DixonColesParameters = DEFAULT,
    previous: Strengths | None = None,
) -> Strengths | None:
    """Fit strengths to ``window``, or return ``None`` if it is too thin.

    Args:
        window: Matches strictly before ``as_of``. The caller owns that
            guarantee; this function does not re-check it, because the slicing
            that establishes it is the same slicing that defines the window.
        as_of: The date the fit is made for. Time decay is measured back from
            here.
        parameters: Window, decay and bounds.
        previous: The last fit for this competition, used as the starting
            point. Warm starting is what makes a refit every 30 days
            affordable — the optimiser lands near the answer rather than
            walking to it from zero.
    """
    teams = sorted(set(window["home_team_id"]) | set(window["away_team_id"]))
    if len(window) < parameters.min_matches or len(teams) < parameters.min_teams:
        return None

    position = {team: index for index, team in enumerate(teams)}
    days_before = (as_of - window["date"]).dt.days.to_numpy(dtype="float64")

    built = _Window(
        home_index=window["home_team_id"].map(position).to_numpy(dtype="int64"),
        away_index=window["away_team_id"].map(position).to_numpy(dtype="int64"),
        home_goals=window["home_goals"].to_numpy(dtype="float64"),
        away_goals=window["away_goals"].to_numpy(dtype="float64"),
        weights=np.exp(-parameters.decay * days_before),
        teams=teams,
    )

    start = np.zeros(2 * len(teams) + 2)
    start[-2] = 0.25  # a sensible home-advantage prior, in log-goals
    if previous is not None:
        for team, index in position.items():
            start[index] = previous.attack.get(team, 0.0)
            start[len(teams) + index] = previous.defence.get(team, 0.0)
        start[-2] = previous.home_advantage
        start[-1] = previous.rho

    bounds = (
        [(-3.0, 3.0)] * len(teams)
        + [(-3.0, 3.0)] * len(teams)
        + [(-1.0, 1.0), (-parameters.max_rho, parameters.max_rho)]
    )
    result = minimize(
        _negative_log_likelihood,
        start,
        args=(built,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": parameters.max_iterations},
    )
    # `result.success` is False on hitting the iteration cap, which a warm
    # start reaches routinely while already sitting on a good answer. The
    # parameters are still the best seen, so they are used; a fit that had not
    # moved at all would show up as flat strengths, not as silence.
    theta = np.asarray(result.x, dtype="float64")
    count = len(teams)
    return Strengths(
        attack={team: float(theta[position[team]]) for team in teams},
        defence={team: float(theta[count + position[team]]) for team in teams},
        home_advantage=float(theta[-2]),
        rho=float(theta[-1]),
    )


class DixonColesRatings:
    """Per-competition strengths, refitted on a rolling window.

    Satisfies :class:`~src.ratings.base.RatingModel`.
    """

    name = "dixon_coles"
    feature_columns = tuple(DIXON_COLES_COLUMNS)

    def __init__(self, parameters: DixonColesParameters = DEFAULT) -> None:
        self.parameters = parameters

    def rate(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Rate every match, refitting per competition as the dates advance."""
        require_chronological(matches)

        columns = [KEY_COLUMN, *DIXON_COLES_COLUMNS]
        if matches.empty:
            return pd.DataFrame({column: [] for column in columns}).astype(
                {KEY_COLUMN: "string", **DIXON_COLES_COLUMNS}
            )

        rated: list[pd.DataFrame] = []
        for competition_id, group in matches.groupby("competition_id", observed=True, sort=False):
            rated.append(self._rate_competition(str(competition_id), group))

        # Reassembled in the caller's order. Grouping is an implementation
        # detail of the fitting, and a rating table whose row order depended on
        # it would silently stop lining up with the canonical table.
        combined = pd.concat(rated, ignore_index=True).set_index(KEY_COLUMN)
        return (
            combined.reindex(matches[KEY_COLUMN])
            .reset_index()
            .astype({KEY_COLUMN: "string", **DIXON_COLES_COLUMNS})
        )

    def _rate_competition(self, competition_id: str, group: pd.DataFrame) -> pd.DataFrame:
        settings = self.parameters
        window = pd.Timedelta(settings.window_days, "D")

        strengths: Strengths | None = None
        next_refit: pd.Timestamp | None = None
        rows: list[tuple[float | None, ...]] = []
        fits = 0

        dates = group["date"].to_numpy()
        for date, home, away in zip(
            group["date"], group["home_team_id"], group["away_team_id"], strict=True
        ):
            if next_refit is None or date >= next_refit:
                # `dates < date`, strictly: a full Saturday programme is one
                # round, and a model fitted on the 3pm results to predict the
                # 5.30 kick-off would look excellent and be useless.
                history = group.iloc[: int(np.searchsorted(dates, np.datetime64(date), "left"))]
                candidate = fit_window(
                    history[history["date"] >= date - window],
                    date,
                    settings,
                    previous=strengths,
                )
                if candidate is not None:
                    strengths = candidate
                    fits += 1
                next_refit = date + pd.Timedelta(settings.refit_days, "D")

            # `current` rather than reaching through `strengths` again: the
            # None check above narrows this local, and the attribute access
            # below would otherwise be on an optional the checker cannot see
            # through.
            current = strengths
            rates = current.rates(home, away) if current is not None else None
            if current is None or rates is None:
                rows.append((None, None, None, None, None))
                continue
            home_rate, away_rate = rates
            home_win, draw, away_win = outcome_probabilities(home_rate, away_rate, current.rho)
            rows.append((home_rate, away_rate, home_win, draw, away_win))

        logger.debug("%s: %d fits over %d matches", competition_id, fits, len(group))
        frame = pd.DataFrame(rows, columns=list(DIXON_COLES_COLUMNS))
        frame.insert(0, KEY_COLUMN, group[KEY_COLUMN].to_numpy())
        return frame
