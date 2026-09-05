"""Results, from the table this project's pipelines wrote.

The provider that works today, and the only source of a scoreline anywhere in
this application. It reads the canonical match table through
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

import pandas as pd

from dashboard.domain.match import Fixture, MatchStatus
from src.pipelines.tables import RESULT_COLUMNS, read_matches
from src.utils.logging import get_logger

logger = get_logger(__name__)


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
