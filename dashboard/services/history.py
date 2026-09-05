"""Match history: what happened, cached, and the small tables built from it.

The service over :class:`~dashboard.providers.historical.HistoricalResults`.
The provider does the reading; this adds the cache Streamlit needs and the
three derived tables a fixture page asks for — form, head-to-head, and the
record between two clubs.

**None of these is a feature.** A head-to-head list is rows filtered by two club
names; a form string is the last five results. Neither is a rolling window, a
rating or anything the model reads, and neither is allowed to become one — a
dashboard that computed a model input would be an unprobed second copy of the
layer the leakage suite exists to guard. Everything the *model* is described by
comes from :mod:`dashboard.services.reports`, which recomputes nothing at all.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.domain.match import Fixture
from dashboard.providers.historical import HistoricalResults, empty, to_fixtures

RECENT_DAYS = 21
"""How far back the home page's results section reaches.

Three weeks: long enough that a mid-week round and two weekends are always on
the page, short enough that the read stays a few thousand rows.
"""

FORM_MATCHES = 5
HEAD_TO_HEAD_MATCHES = 10


# ---- the cached reads --------------------------------------------------------
#
# Every one is keyed on the path as a string, because that is what Streamlit
# can hash cheaply — a Path or a provider instance is not.


@st.cache_data(show_spinner="reading the match table…")
def latest_date(matches_path: str) -> dt.date | None:
    """The most recent date the table holds, or ``None`` when there is no table."""
    return HistoricalResults(Path(matches_path)).latest()


@st.cache_data(show_spinner="reading the match table…")
def recent(matches_path: str, *, days: int = RECENT_DAYS) -> pd.DataFrame:
    """Every match in the last ``days`` of the table, all competitions.

    Bounded relative to the **latest match in the table**, not to today. The
    table is rebuilt by `make data`, and a reader who has not run it this month
    should still see the last round that was ingested — "nothing has happened
    in three weeks" is a statement about stale data that this function has no
    business making.
    """
    provider = HistoricalResults(Path(matches_path))
    latest = provider.latest()
    if latest is None:
        return empty()
    return provider.frame(since=latest - dt.timedelta(days=days))


@st.cache_data(show_spinner="reading the match table…")
def all_matches(matches_path: str) -> pd.DataFrame:
    """The whole match table, display columns only.

    One cached frame that the detail page filters several ways — the fixture's
    own row, both clubs' form, the head-to-head — rather than one cached read
    per question, which is the same three hundred thousand rows held three
    times.

    ponytail: twelve columns of a table that is thirty-five wide, held once per
    Streamlit process. If a fixture provider makes this a per-request cost, the
    upgrade is a team predicate pushed into ``DuckDBStore.read_matches``, not
    an index here.
    """
    return HistoricalResults(Path(matches_path)).frame()


@st.cache_data(show_spinner="reading the match table…")
def for_competition(matches_path: str, competition_id: str, *, seasons: int = 1) -> pd.DataFrame:
    """One competition's matches, most recent seasons first.

    ``seasons`` is a count of season labels rather than a date bound, because
    seasons do not line up across the thirty-nine competitions — Brazil runs on
    calendar years and England does not — and "this season" is the question a
    league page is asked.
    """
    frame = HistoricalResults(Path(matches_path)).frame(competitions=[competition_id])
    if frame.empty:
        return frame
    labels = sorted(frame["season"].dropna().unique(), reverse=True)[:seasons]
    return frame[frame["season"].isin(labels)].sort_values("date", ascending=False)


def as_fixtures(frame: pd.DataFrame, *, limit: int | None = None) -> list[Fixture]:
    """Rows a view has narrowed, as the cards it is about to render.

    Here rather than imported straight from the provider by the view. A view
    that named a provider module would be a view to revisit the day a second
    source of results exists, and the point of this layer is that there is
    exactly one place to revisit.
    """
    found = to_fixtures(frame)
    return found if limit is None else found[:limit]


# ---- the tables a fixture page asks for --------------------------------------


def by_id(frame: pd.DataFrame, match_id: str) -> pd.Series | None:
    """One fixture's row, or ``None`` when the table does not hold it."""
    if frame.empty:
        return None
    found = frame[frame["match_id"] == match_id]
    return None if found.empty else found.iloc[0]


def for_team(frame: pd.DataFrame, team: str, *, limit: int = 200) -> pd.DataFrame:
    """One club's matches, most recent first. A filter, not a read."""
    if frame.empty:
        return frame
    played = frame[(frame["home_team"] == team) | (frame["away_team"] == team)]
    return played.sort_values("date", ascending=False).head(limit)


def head_to_head(
    frame: pd.DataFrame, home: str, away: str, *, limit: int = HEAD_TO_HEAD_MATCHES
) -> pd.DataFrame:
    """Meetings between two clubs, either way round, most recent first."""
    if frame.empty:
        return frame
    pairs = (frame["home_team"] == home) & (frame["away_team"] == away)
    reverse = (frame["home_team"] == away) & (frame["away_team"] == home)
    return frame[pairs | reverse].sort_values("date", ascending=False).head(limit)


def form(frame: pd.DataFrame, team: str, *, matches: int = FORM_MATCHES) -> list[str]:
    """The club's last results as ``W``/``D``/``L``, **oldest first**.

    Oldest first because that is the direction a form string is read — the
    rightmost square is the most recent match — and reversing it at the point
    of rendering is how half the callers end up reading it backwards.
    """
    if frame.empty:
        return []
    played = for_team(frame, team, limit=matches)
    return [_letter(row, team) for _, row in played.iloc[::-1].iterrows()]


def _letter(row: pd.Series, team: str) -> str:
    """One match as it went for ``team``. ``D`` covers a row with no result."""
    result = str(row.get("result", ""))
    if result == "D":
        return "D"
    won_at_home = result == "H" and row["home_team"] == team
    won_away = result == "A" and row["away_team"] == team
    if won_at_home or won_away:
        return "W"
    return "L" if result in ("H", "A") else "D"


def record(frame: pd.DataFrame, home: str) -> tuple[int, int, int]:
    """A head-to-head as wins, draws and losses — from ``home``'s point of view.

    One club rather than two: the other side of a two-club head-to-head is
    every row that is not this club's, so naming it would be a second argument
    that can disagree with the frame.
    """
    home_wins = draws = away_wins = 0
    for _, row in frame.iterrows():
        letter = _letter(row, home)
        if letter == "W":
            home_wins += 1
        elif letter == "D":
            draws += 1
        else:
            away_wins += 1
    return home_wins, draws, away_wins


def teams_in(frame: pd.DataFrame) -> list[str]:
    """Every club appearing in a frame, sorted. What a search box completes on."""
    if frame.empty:
        return []
    names = pd.concat([frame["home_team"], frame["away_team"]]).dropna().unique()
    return sorted(str(name) for name in names)


def follow(frame: pd.DataFrame, teams: Sequence[str]) -> pd.DataFrame:
    """The rows involving any of ``teams``. All of them when the list is empty."""
    if not teams or frame.empty:
        return frame
    wanted = set(teams)
    return frame[frame["home_team"].isin(wanted) | frame["away_team"].isin(wanted)]
