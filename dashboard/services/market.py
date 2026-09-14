"""The market's price, the model's goal rates, and what a gap between them means.

The market service layer. Three cached reads and one lookup, over
:class:`~dashboard.providers.historical.HistoricalOdds`, the ratings table and
the disagreement table `make card` writes.

**Nothing here is computed.** The de-vig is
:func:`src.evaluation.market.implied_probabilities` — the same function the
backtest's ``bookmaker`` benchmark uses, so the percentages a reader sees are
the percentages the model was scored against. The verdict is a row of
``market.parquet``, produced by :func:`src.pipelines.report.market_comparison`
over 61,889 out-of-sample forecasts. A dashboard that recomputed either would
be a second number to reconcile, which is the rule
:mod:`dashboard.services.reports` states and this follows.

**The gap is the model's, not the market's.** That is the finding, and it is
the reason this module exists rather than a value detector: over the folds, the
model's deficit against the closing line *grows* with the size of the
disagreement, from 0.0009 where the two are within two points of each other to
0.1835 where they are more than twenty apart. A page that presented the same
gap as an edge would be contradicting the measurement three directories away.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.domain.match import ExpectedGoals, MarketPrice, Prediction
from dashboard.providers.base import OddsProvider
from dashboard.providers.historical import expected_goals
from src.evaluation.market import band, distance
from src.pipelines.tables import read_ratings

BAND_COLUMN = "band"
BUILD_COMMAND = "make card"
"""What writes the disagreement table, named in the empty state."""


@st.cache_data(show_spinner="reading the ratings table…")
def _ratings(ratings_path: str) -> pd.DataFrame:
    """The ratings table, cached and keyed on the path as a string.

    Keyed on a string for the reason every read in
    :mod:`dashboard.services.history` is: a ``Path`` is not what Streamlit
    hashes cheaply. Empty rather than ``None`` when the table is absent, so the
    filter below has one shape to handle.
    """
    frame = read_ratings(Path(ratings_path))
    return pd.DataFrame() if frame is None else frame


def goals(ratings_path: str, match_id: str) -> ExpectedGoals | None:
    """A fixture's fitted goal rates, or ``None`` when the model has none for it.

    ``None`` is the honest gap rather than a failure. Dixon-Coles refits per
    competition on a rolling window and has nothing to say about a match before
    its first fit, so a null means exactly that — see
    :class:`~dashboard.domain.match.ExpectedGoals` for what these numbers are
    and, more importantly, what they are not.
    """
    return expected_goals(_ratings(ratings_path), match_id)


def price(odds: OddsProvider, match_id: str) -> MarketPrice | None:
    """The closing line for one fixture, or ``None`` when there is not one.

    Not cached here, and deliberately: the provider is the cache key a
    ``st.cache_data`` would need and it is a dataclass holding a path, so the
    read is one narrow scan of a Parquet file per fixture a reader opens.
    Caching per ``(path, match_id)`` would hold one entry per match ever
    viewed to save 190 ms on a page nobody revisits twice in a session.
    """
    return odds.price(match_id)


def gap(prediction: Prediction, quoted: MarketPrice) -> float:
    """How far apart the model and the market are, in the units the table buckets on.

    Total-variation distance, which for three outcomes reads as a
    percentage-point gap: 0.07 is "seven points apart". The same function the
    measurement uses, so the number on the page and the row it selects below
    cannot come from two different definitions of "apart".
    """
    # Ordered by the market's own keys rather than by a constant, so the two
    # rows are aligned by outcome name and a source that ever answered in a
    # different order could not silently produce a gap of zero.
    stated = np.array([[prediction.probabilities.get(key, 0.0) for key in quoted.probabilities]])
    priced = np.array([list(quoted.probabilities.values())])
    return float(distance(stated, priced)[0])


def verdict(market: pd.DataFrame | None, apart: float) -> pd.Series | None:
    """The row of the disagreement table this fixture falls in, if there is one.

    ``None`` when `make card` has not written the table, which is the
    clean-checkout state and the state before the run that produces it — the
    page then shows the two forecasts and says which command measures what the
    difference is worth, rather than guessing.
    """
    if market is None or market.empty:
        return None
    label = band(apart)
    found = market[market[BAND_COLUMN].astype(str) == label]
    return None if found.empty else found.iloc[0]


__all__ = ["BUILD_COMMAND", "gap", "goals", "price", "verdict"]
