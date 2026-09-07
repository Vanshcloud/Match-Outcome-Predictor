"""The closing line as a forecast, and what disagreeing with it is worth.

Milestone 17. Two things live here and they are the same subject from opposite
ends: turning decimal odds into probabilities, and measuring what happens when
the model's probabilities differ from them.

**The de-vig is here rather than in the baseline that used to hold it.** Three
callers now need "what did the market say" — the backtest's benchmark, the
disagreement table below, and the dashboard's match page — and three
implementations of removing an overround is three chances to publish a number
that is not the one the benchmark was scored against.

**Why the measurement is the milestone rather than a value detector.** The
obvious reading of "the model says 45%, the market says 38%" is that there is
value in the difference. That reading is testable, this project has the data to
test it, and it does not survive: over the walk-forward folds the model's
deficit against the closing line *grows* with the size of the disagreement, and
on the matches where the two differ most the market gets sharper rather than
weaker. So the difference is not an edge over the price; it is the best
available estimate of how wrong this model is about a particular fixture.
:func:`disagreement` is what produces that table, so the claim is a
measurement anyone can re-run rather than a sentence in a document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.evaluation.metrics import terms
from src.ingestion.base import ODDS_COLUMNS

if TYPE_CHECKING:
    from src.evaluation.metrics import Outcomes

MARKET_COLUMNS: tuple[str, str, str] = ("odds_home", "odds_draw", "odds_away")
"""The three closing prices, in home-draw-away order.

Named here as a tuple rather than read from :data:`~src.ingestion.base.ODDS_COLUMNS`
directly because a ``dict`` has an order that is incidental and this order is
load-bearing — it is the class order every metric in this package accumulates
over. The assertion below is what keeps the two from drifting.
"""

assert set(MARKET_COLUMNS) == set(ODDS_COLUMNS), "the odds block and the market order disagree"

MIN_DECIMAL_ODDS = 1.0
"""A decimal price is a multiplier on the stake and cannot be below one.

At exactly one the implied probability is certainty, which no bookmaker offers
and no feed should contain, so the bound is strict.
"""

DISAGREEMENT_EDGES: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.20, 1.0)
"""Bucket boundaries for :func:`disagreement`, in total-variation distance.

Five bands rather than deciles. The question is whether the relationship is
monotone, and a reader checks five rows by eye; ten rows of a hundred matches
each would be noise presented as resolution.
"""

DISAGREEMENT_LABELS: tuple[str, ...] = ("<2%", "2-5%", "5-10%", "10-20%", ">20%")
"""How each band of :data:`DISAGREEMENT_EDGES` is named in a report and a page."""


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """Decimal odds as probabilities, with the overround removed proportionally.

    Decimal odds imply probabilities that sum to more than one — the overround,
    around 8% in this feed — and the standard removal is to divide each by the
    total. That is not the only way to do it (the favourite carries more of the
    margin than an equal share) but it is the transparent one, and a cleverer
    de-vigging would make this benchmark a modelling choice rather than a
    measurement.

    Args:
        odds: ``(n, 3)`` decimal prices, home-draw-away.

    Returns:
        ``(n, 3)`` probabilities summing to one, or a row of ``nan`` where the
        price was missing or not a price. A partially filled row is *all* nan:
        two of three prices normalise to something that sums to one and is not
        a forecast of anything.
    """
    priced = np.asarray(odds, dtype=float)
    usable = np.isfinite(priced).all(axis=1) & (priced > MIN_DECIMAL_ODDS).all(axis=1)

    stated = np.full(priced.shape, np.nan)
    implied = np.reciprocal(priced, where=usable[:, None], out=np.full(priced.shape, np.nan))
    stated[usable] = implied[usable] / implied[usable].sum(axis=1, keepdims=True)
    return stated


def overround(odds: np.ndarray) -> np.ndarray:
    """How much more than certainty each row of prices adds up to.

    The bookmaker's margin, as a fraction: ``0.06`` is a book paying out on
    106% of the stake. Reported beside a de-vigged probability because the
    removal above is an assumption about *how* that margin is spread, and a
    reader who can see its size can judge how much the assumption matters.
    """
    priced = np.asarray(odds, dtype=float)
    usable = np.isfinite(priced).all(axis=1) & (priced > MIN_DECIMAL_ODDS).all(axis=1)
    total = np.reciprocal(priced, where=usable[:, None], out=np.full(priced.shape, np.nan)).sum(
        axis=1
    )
    return np.where(usable, total - 1.0, np.nan)


def distance(model: np.ndarray, market: np.ndarray) -> np.ndarray:
    """Total-variation distance between two forecasts of the same match.

    Half the sum of absolute differences, which for three classes is the
    largest probability either forecast assigns to any *set* of outcomes that
    the other does not. It reads as a percentage-point gap — 0.07 is "seven
    points apart" — which is what makes it the right axis to bucket on: a
    reader looking at two rows of percentages is already doing this subtraction
    by eye.
    """
    apart = np.abs(np.asarray(model, dtype=float) - np.asarray(market, dtype=float))
    total: np.ndarray = apart.sum(axis=1) / 2
    return total


def band(gap: float) -> str | None:
    """Which :data:`DISAGREEMENT_LABELS` band a single distance falls in.

    How one fixture on a page finds its own row of the table. ``None`` for a
    gap that is not a number, which is what an unpriced match produces.
    """
    if not np.isfinite(gap):
        return None
    # Every edge but the last. The last band is open above rather than closed
    # at 1.0: total variation cannot exceed one, so a closing comparison there
    # would be a branch nothing can reach, and the fall-through says the same
    # thing without one.
    for label, upper in zip(DISAGREEMENT_LABELS[:-1], DISAGREEMENT_EDGES[1:-1], strict=True):
        if gap <= upper:
            return label
    return DISAGREEMENT_LABELS[-1]


def disagreement(
    model: np.ndarray,
    market: np.ndarray,
    outcomes: Outcomes,
    *,
    edges: tuple[float, ...] = DISAGREEMENT_EDGES,
    labels: tuple[str, ...] = DISAGREEMENT_LABELS,
) -> pd.DataFrame:
    """What each forecaster scores, grouped by how far apart the two were.

    The measurement Milestone 17 rests on. Rows where either forecaster has
    nothing to say are dropped before anything is grouped: a band whose model
    column is computed over more matches than its market column would compare
    two different questions, which is the same reason
    :mod:`src.pipelines.backtest` reports a common subset.

    Args:
        model: ``(n, 3)`` probabilities from the model under test.
        market: ``(n, 3)`` probabilities from the closing line, already
            de-vigged by :func:`implied_probabilities`.
        outcomes: What actually happened, one per row.

    Returns:
        One row per band, with the count, both log losses, the model's deficit,
        and the share of matches the model scored better on. Ordered by band
        rather than by any column, because the whole claim is about the
        *direction* down the table.
    """
    stated = np.asarray(model, dtype=float)
    priced = np.asarray(market, dtype=float)
    scored = pd.DataFrame(
        {
            "gap": distance(stated, priced),
            "model": terms(np.nan_to_num(stated, nan=1 / 3), outcomes)["log_loss"].to_numpy(),
            "market": terms(np.nan_to_num(priced, nan=1 / 3), outcomes)["log_loss"].to_numpy(),
            "usable": np.isfinite(stated).all(axis=1) & np.isfinite(priced).all(axis=1),
        }
    )
    scored = scored[scored["usable"]].drop(columns="usable")
    scored["band"] = pd.cut(scored["gap"], list(edges), labels=list(labels), include_lowest=True)
    # A column rather than a lambda over each group: the mean of a boolean is
    # the share, `agg` computes it in the same pass as the two log losses, and
    # a `groupby.apply` returns a frame rather than a series when there are no
    # groups at all — which is the clean-checkout case, not an exotic one.
    scored["better"] = scored["model"] < scored["market"]

    table = scored.groupby("band", observed=True).agg(
        n=("gap", "size"),
        model=("model", "mean"),
        market=("market", "mean"),
        model_better=("better", "mean"),
    )
    table["model_minus_market"] = table["model"] - table["market"]
    return table.reset_index()[
        ["band", "n", "model", "market", "model_minus_market", "model_better"]
    ]


__all__ = [
    "DISAGREEMENT_EDGES",
    "DISAGREEMENT_LABELS",
    "MARKET_COLUMNS",
    "MIN_DECIMAL_ODDS",
    "band",
    "disagreement",
    "distance",
    "implied_probabilities",
    "overround",
]
