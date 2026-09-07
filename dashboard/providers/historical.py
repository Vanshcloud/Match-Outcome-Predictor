"""Results and closing prices, from the table this project's pipelines wrote.

The providers that work today: the only source of a scoreline anywhere in this
application, and — since Milestone 17 — the only source of a bookmaker's price.
Both read the same canonical table, because the closing odds are three columns
of it; two modules would be two opens of one file. It reads the canonical match table through
:func:`src.pipelines.tables.read_matches` — through the storage layer, like
everything else in this project that reads a table.

**Why not from the API.** ``/predict`` deliberately returns no scoreline: a
prediction endpoint that handed back the result beside its forecast would be
answering a different question, and ``docs/API.md`` says so. Results are not
predictions, and they come from where results come from.

**Nothing here is a feature.** This provider filters and projects rows. It
computes no rolling window, no rating and nothing else the model reads, and it
is not allowed to start — a dashboard that computed a model input would be an
unprobed second copy of the layer the leakage suite exists to guard.

**No cache and no Streamlit.** Caching is the service layer's job, in
:mod:`dashboard.services.history`. A provider that cached would be one a test
had to reset between cases, and one a second front end would have to disable.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashboard.domain.match import OUTCOMES, ExpectedGoals, Fixture, MarketPrice, MatchStatus
from src.evaluation.market import MARKET_COLUMNS, implied_probabilities, overround
from src.pipelines.tables import KEY_COLUMN, RESULT_COLUMNS, read_matches
from src.utils.logging import get_logger

logger = get_logger(__name__)

LAMBDA_COLUMNS: tuple[str, str] = ("dc_home_lambda", "dc_away_lambda")
"""Dixon-Coles' two Poisson rates, home first.

Named here rather than imported from :mod:`src.ratings.base`: CI confines this
package to the pipelines, the models, the metrics and the canonical schema, and
the rating internals are on the far side of that line. Two strings is the
cheaper half of that trade.
"""


@dataclass(frozen=True, slots=True)
class HistoricalResults:
    """Finished matches, out of the canonical table.

    ``available`` is a property rather than a stored flag because the table
    appears the moment `make data` finishes, and a provider constructed at
    import time would go on reporting the empty state it was built in.
    """

    path: Path
    name: str = "canonical-table"

    @property
    def available(self) -> bool:
        return self.path.is_file()

    def frame(
        self,
        *,
        competitions: Sequence[str] | None = None,
        since: dt.date | None = None,
        until: dt.date | None = None,
    ) -> pd.DataFrame:
        """The rows themselves, for the readers that want a table.

        Head-to-head, form and a season's fixture list are all filters over a
        frame, and turning three hundred thousand rows into dataclasses to
        filter six of them would be the most expensive way to answer any of
        them. :meth:`results` is the same read as fixtures, for the readers
        that render cards.
        """
        if not self.available:
            logger.info("no match table at %s", self.path)
            return empty()
        frame = read_matches(
            self.path,
            competitions=competitions,
            since=since.isoformat() if since else None,
            until=until.isoformat() if until else None,
        )
        return empty() if frame is None else frame

    def results(
        self,
        *,
        competitions: Sequence[str] | None = None,
        since: dt.date | None = None,
        until: dt.date | None = None,
        limit: int | None = None,
    ) -> list[Fixture]:
        """Finished matches in the window, most recent first."""
        frame = self.frame(competitions=competitions, since=since, until=until)
        fixtures = to_fixtures(frame)
        return fixtures if limit is None else fixtures[:limit]

    def latest(self) -> dt.date | None:
        """The most recent date the table holds, read as one column.

        Its own narrow read. The alternative is loading three hundred thousand
        rows to look at the last one.
        """
        if not self.available:
            return None
        frame = read_matches(self.path, columns=("date",))
        if frame is None or frame.empty:
            return None
        return pd.Timestamp(frame["date"].max()).date()


@dataclass(frozen=True, slots=True)
class HistoricalOdds:
    """The bookmaker's closing line, out of the same canonical table.

    **The odds are read here and nowhere else in this package.** They are
    deliberately not in :data:`~src.pipelines.tables.RESULT_COLUMNS` — Milestone
    12 kept them off the reader's columns on the grounds that a page showing
    both invites the model/market comparison to be made without the folds that
    make it meaningful. Milestone 17 makes that comparison properly instead of
    hiding it, and this is the narrow read that feeds it: three columns and a
    key, never joined to the twelve a result is displayed by.

    **The de-vig is not done here either.** It is
    :func:`src.evaluation.market.implied_probabilities`, which is what the
    backtest's ``bookmaker`` benchmark uses, so the percentages this returns
    are the percentages the model was scored against.
    """

    path: Path
    name: str = "canonical-table"

    @property
    def available(self) -> bool:
        return self.path.is_file()

    def price(self, match_id: str) -> MarketPrice | None:
        """One fixture's closing line, or ``None`` when there is not one.

        A read per fixture rather than a cached frame of all three hundred
        thousand: this answers one match on one page, the store pushes the
        filter into the query, and the service layer above caches the result.
        """
        found = self._row(match_id)
        if found is None:
            return None
        # Off the one-row *frame*, never off a row. A row of this table is a
        # mixed Series — the key is a string — so selecting three nullable
        # `Float64` cells out of it yields an object array, and a missing price
        # in one is a `NAType` that `float()` refuses. Roughly a fifth of this
        # table has no price, so that is the ordinary path rather than an edge.
        odds = found[list(MARKET_COLUMNS)].to_numpy(dtype=float)
        stated = implied_probabilities(odds)[0]
        if not np.isfinite(stated).all():
            return None
        return MarketPrice(
            match_id=match_id,
            odds=dict(zip(OUTCOMES, (float(value) for value in odds[0]), strict=True)),
            probabilities=dict(zip(OUTCOMES, (float(value) for value in stated), strict=True)),
            overround=float(overround(odds)[0]),
        )

    def _row(self, match_id: str) -> pd.DataFrame | None:
        """The fixture's odds as a one-row frame, or ``None`` when there is none.

        A frame rather than a Series, for the dtype reason above. No
        ``available`` check in front of the read either: it would be the same
        ``is_file`` this class's property does and the same one
        :func:`~src.pipelines.tables.read_matches` does before it logs the
        absent table — three checks of one fact, two of which no test could
        reach.
        """
        frame = read_matches(self.path, columns=(KEY_COLUMN, *MARKET_COLUMNS))
        if frame is None:
            return None
        found = frame[frame[KEY_COLUMN].astype(str) == match_id]
        return None if found.empty else found.head(1)


def expected_goals(ratings: pd.DataFrame, match_id: str) -> ExpectedGoals | None:
    """A fixture's Dixon-Coles goal rates, or ``None`` when it has none.

    Not a provider method, and not on :class:`HistoricalOdds`, because these
    are not somebody else's data about football — they are *this project's own
    fitted model*, in the table `make ratings` wrote. The same category as the
    report tables :mod:`dashboard.services.reports` reads, and read the same
    way: the service caches the narrow frame and this filters it.

    ``None`` covers the honest gap rather than a failure: Dixon-Coles refits
    per competition on a rolling window and has nothing to say about a match
    before its first fit, and a null there means exactly that.
    """
    if ratings.empty or LAMBDA_COLUMNS[0] not in ratings:
        return None
    found = ratings[ratings[KEY_COLUMN].astype(str) == match_id]
    if found.empty:
        return None
    row = found.iloc[0]
    if any(pd.isna(row.get(column)) for column in LAMBDA_COLUMNS):
        return None
    return ExpectedGoals(
        match_id=match_id,
        home=float(row[LAMBDA_COLUMNS[0]]),
        away=float(row[LAMBDA_COLUMNS[1]]),
    )


def empty() -> pd.DataFrame:
    """The shape every reader here expects, with no rows in it."""
    return pd.DataFrame(columns=list(RESULT_COLUMNS))


def to_fixtures(frame: pd.DataFrame) -> list[Fixture]:
    """Rows of the canonical table as fixtures, newest first.

    Every one is :attr:`~dashboard.domain.match.MatchStatus.FINISHED`. The
    table holds results and nothing else — the ingestion layer will not write a
    row for a match with no scoreline — so this is a fact about the source
    rather than a guess about a row.
    """
    if frame.empty:
        return []
    ordered = frame.sort_values("date", ascending=False)
    return [
        Fixture(
            match_id=str(row["match_id"]),
            competition_id=str(row["competition_id"]),
            date=as_date(row["date"]),
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
            status=MatchStatus.FINISHED,
            competition=maybe(row.get("competition")),
            country=maybe(row.get("country")),
            kickoff=maybe(row.get("kickoff")),
            home_goals=as_int(row.get("home_goals")),
            away_goals=as_int(row.get("away_goals")),
        )
        for _, row in ordered.iterrows()
    ]


def as_date(value: Any) -> dt.date:
    return pd.Timestamp(str(value)).date()


def maybe(value: Any) -> str | None:
    """A cell as text, or ``None`` when it is absent or NaN."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def as_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
