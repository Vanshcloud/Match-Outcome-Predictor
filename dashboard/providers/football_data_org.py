"""Fixtures and live scores, from football-data.org.

Milestone 13, and the whole of it: one class satisfying
:class:`~dashboard.providers.base.FixtureProvider`, one entry in
:data:`~dashboard.providers.FIXTURE_PROVIDERS`, one environment variable. No
view moved and no card changed to add it, which was the claim Milestone 12's
shape was making.

**A second provider, not a second ingestion source.** Nothing here is written
to a table, joined to one, or read by a model. This feed answers "what is on
tonight" — the one question the results feed this project ingests cannot answer
— and its rows live for as long as a page render. The moment a fixture is
played, the *canonical* row for it comes from `make data` like every other
result, through :mod:`dashboard.providers.historical`.

That is why the ids are prefixed ``fdorg-`` rather than built with
:func:`src.ingestion.base.make_match_id`. A match id in this project is a hash
of the natural key *including the team names as its provider spells them*, and
this feed spells them differently ("Manchester United FC" against "Man
United"). An id built here would look like a canonical one and match nothing,
which is worse than an id that plainly says where it came from. The
consequence is visible and deliberate: a card from this feed opens a match page
that has neither a table row nor a forecast, and that page already says so —
the model in this repository is fitted on finished matches and cannot price a
fixture it has no history for.

**Two clocks, and only one of them reaches the rest of the application.**
The feed indexes by UTC and stamps every kick-off ``utcDate``; the dashboard's
callers ask about *their* today — :func:`datetime.date.today` on the host — and
those are different days for most of the world. Mixing them is not a rounding
error: on a machine at UTC+05:30, a match kicking off at 21:00 UTC is "tonight"
to the reader and *yesterday* to the feed, so a live centre asking for the
local date would go blank exactly during Saturday evening in Europe. So every
:class:`~dashboard.domain.match.Fixture` this module returns carries **host
local** date and kick-off, UTC is confined to the wire, and the windows sent to
the feed are widened by a day at each end to cover the offset.

**Free tier, so requests are the scarce resource.** Ten calls a minute, and
Streamlit reruns the whole script on every click. Answers are therefore memoised
for :data:`CACHE_SECONDS` at module level rather than per instance: the context
builds a fresh provider on every rerun, so an instance cache would be a cache
that is empty every time it is read.

**Every failure is an empty answer with a reason**, as with every other
provider here. A feed that rejects an expired key must not raise into a page
whose other five sections are about something else — and the reason has to be
the feed's own words, because this one answers a bare ``400`` with an empty
HTTP reason phrase and puts the whole explanation in the body.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import requests

from dashboard.domain.match import Fixture, MatchStatus
from src.utils.http import HttpClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

API_KEY_ENV = "FOOTBALL_DATA_API_KEY"
"""The key, from the environment and never from ``configs/config.yaml``.

Same rule as ``PREDICTION_LOG_DSN``: that file is committed, so a credential
that can only arrive by environment is one nobody can commit by accident.
"""

BASE_URL = "https://api.football-data.org/v4"
TIMEOUT_SECONDS = 10.0
CACHE_SECONDS = 60
"""How long an answer is reused. A minute is under the free tier's rate limit
at any plausible click rate, and it is the resolution a live score is worth —
the section is repainted on the next rerun after that."""

MAX_WINDOW_DAYS = 10
"""Whole days one request can cover. The feed's own rule, measured on it:
``dateTo - dateFrom`` of eleven days is answered ``400 Specified period must
not exceed 10 days``; ten is answered. A longer ask is split into consecutive
requests rather than clamped — a page showing six days of an ask for fourteen
would be missing fixtures with nothing on the screen saying so.

ponytail: a very long window is one request per ten days against a limit of ten
a minute. The shipped dashboard asks for seven days, so it is one request; a
caller that wants a season wants a prefetch, not a wider loop here.
"""

# The feed's window is [dateFrom T00:00Z, dateTo T00:00Z] and it includes both
# instants, so `dateTo` is the morning *after* the last day a caller wants: a
# request for 09-05..09-05 answers nothing at all, and 09-05..09-06 answers the
# whole of the 5th plus anything kicking off at exactly 09-06T00:00Z. Measured,
# because it is the difference between a live centre that works and one that is
# permanently empty. Every date in this module is an inclusive day, and this is
# the single place that converts.

MAX_RETRIES = 2
"""Fewer than :class:`~src.utils.http.HttpClient` defaults to. This runs inside
a page render: five retries with backoff is a page that hangs for fifteen
seconds rather than a section that says the feed is not answering."""

COMPETITION_CODES: Mapping[str, str] = {
    "ENG_1": "PL",
    "ENG_2": "ELC",
    "GER_1": "BL1",
    "ITA_1": "SA",
    "ESP_1": "PD",
    "FRA_1": "FL1",
    "NED_1": "DED",
    "POR_1": "PPL",
    "BRA_1": "BSA",
}
"""This project's competition ids against football-data.org's codes.

Here rather than in ``configs/leagues.yaml``, which is otherwise where a
competition is added to this project. The registry describes what is
*ingested*: a feed, a division code and a season layout, all of them
football-data.co.uk's. This is one consumer's coverage of nine of those
thirty-nine competitions, it changes when a subscription changes rather than
when a league is added, and putting it in the registry would mean every
competition carrying a column about a feed that has nothing to do with it.

Nine, because that is what the free tier serves. A competition this table
does not name is dropped from the filter rather than sent — not defensively:
this project's ids are its own, and there is simply no code here to send for
one. Measured against the live API, a paid competition left in a filter is
ignored rather than refused (``competitions=PL,BL2`` answers 200 with the PL
rows), so the drop costs nothing and the alternative would be an unfiltered
request for competitions the reader did not ask about.
"""

OUR_ID: Mapping[str, str] = {code: ours for ours, code in COMPETITION_CODES.items()}

STATUSES: Mapping[str, MatchStatus] = {
    "SCHEDULED": MatchStatus.SCHEDULED,
    "TIMED": MatchStatus.SCHEDULED,
    "IN_PLAY": MatchStatus.LIVE,
    "PAUSED": MatchStatus.LIVE,
    "FINISHED": MatchStatus.FINISHED,
}
"""The feed's status vocabulary, as this application's three states.

``POSTPONED``, ``SUSPENDED``, ``CANCELLED`` and ``AWARDED`` are deliberately
absent: they fall through to
:attr:`~dashboard.domain.match.MatchStatus.UNKNOWN` and are then dropped from
both answers by :meth:`FootballDataOrgFixtures.scheduled`. A postponed match
rendered as scheduled is a kick-off time on the screen that nobody is playing
to, which is the one thing the null provider exists to refuse.
"""

UPCOMING = (MatchStatus.SCHEDULED, MatchStatus.LIVE)

ONE_DAY = dt.timedelta(days=1)
"""The most the feed's calendar and the host's can differ. Every real offset is
under 24 hours, so a window widened by this at each end contains every match
the caller's local dates name."""


@dataclass(slots=True)
class FootballDataOrgFixtures:
    """football-data.org's ``/v4/matches``, as a fixture provider.

    Mutable, like :class:`~dashboard.providers.api.ApiPredictions` and for the
    same reason: :attr:`error` records what the last call failed with and the
    views render that sentence instead of the section.
    """

    api_key: str = field(default_factory=lambda: os.environ.get(API_KEY_ENV, ""))
    base_url: str = BASE_URL
    http: HttpClient | None = None
    """Injected by tests, which mount a stub adapter so no test reaches the
    network. An ``HttpClient`` rather than a session, for the trap
    ``dashboard/client.py`` documents: this class builds its own client, and a
    session handed in would have its adapter silently replaced."""

    name: str = "football-data.org"
    error: str | None = None

    @property
    def available(self) -> bool:
        """Whether there is a key **and** the feed answered.

        The second half costs nothing: it is the same window request
        :meth:`live` makes, and the memo means the two are one call. Without
        it a feed rejecting the key renders "no fixtures scheduled in the next
        week", which is a statement about football rather than about a key.
        """
        if not self.api_key:
            self.error = (
                "No football-data.org key. Register a free one at "
                "football-data.org/client/register and set "
                f"`{API_KEY_ENV}`, or set `DASHBOARD_FIXTURE_PROVIDER=none`."
            )
            return False
        self._now_window(None)
        return self.error is None

    def scheduled(
        self,
        *,
        since: dt.date,
        until: dt.date,
        competitions: Sequence[str] | None = None,
    ) -> list[Fixture]:
        """Matches kicking off in the window, earliest first.

        ``since`` and ``until`` are the caller's own calendar — the dashboard
        passes :func:`datetime.date.today` — so the request is widened by a day
        at each end to cover the offset from the feed's UTC calendar, and the
        answer is narrowed back to the days that were actually asked for.

        In-play matches are included: a match kicking off at three o'clock is
        one of today's fixtures at four whether or not it is still running. A
        finished one is not — that is the results provider's row, with the
        scoreline this project actually ingested.
        """
        found = self._window(since - ONE_DAY, until + ONE_DAY, competitions)
        return sorted(
            (one for one in found if one.status in UPCOMING and since <= one.date <= until),
            key=_order,
        )

    def live(self, *, competitions: Sequence[str] | None = None) -> list[Fixture]:
        """Matches in play right now, with the minute where the feed gives one.

        Filtered by **status**, never by date: the feed's own status is what
        "in play" means, and a date filter is how this method would go blank at
        22:00 UTC for every reader east of Greenwich.

        That status is passed through rather than second-guessed, including
        when it lags — the live API was observed calling a 16:45 kick-off
        ``IN_PLAY`` at 22:40 with a current ``lastUpdated``. Inferring "this
        must have finished" from the clock would be this dashboard inventing a
        result, which is the one thing the whole provider layer refuses.

        (This plan does not populate ``minute`` — measured, not assumed: no row
        the live API returned carried the field. The card then reads ``live``
        rather than ``63'``, which is :func:`dashboard.ui.match_card`'s existing
        fallback and needs nothing here.)
        """
        return [one for one in self._now_window(competitions) if one.status is MatchStatus.LIVE]

    def _now_window(self, competitions: Sequence[str] | None) -> tuple[Fixture, ...]:
        """Everything the feed has around *now*, in the feed's own calendar.

        Yesterday and today in **UTC**, which is the smallest window certain to
        contain every match currently in play whatever the host's timezone.
        """
        utc_today = dt.datetime.now(dt.UTC).date()
        return self._window(utc_today - ONE_DAY, utc_today, competitions)

    def _window(
        self,
        since: dt.date,
        until: dt.date,
        competitions: Sequence[str] | None,
    ) -> tuple[Fixture, ...]:
        """The window, as inclusive days: one memoised request per ten of them,
        merged, with any failure kept in :attr:`error` rather than raised."""
        codes = _codes(competitions)
        if codes is None:
            self.error = (
                "None of the competitions you follow are on this feed's plan. "
                f"It covers {', '.join(sorted(COMPETITION_CODES))}."
            )
            return ()
        found: dict[str, Fixture] = {}
        for start, end in _chunks(since, until):
            params: tuple[tuple[str, str], ...] = (
                ("dateFrom", start.isoformat()),
                ("dateTo", (end + ONE_DAY).isoformat()),
            )
            if codes:
                params += (("competitions", codes),)
            try:
                answered = _fetch(self.base_url, self.api_key, params, self.http, _bucket())
            except (requests.RequestException, ValueError) as failure:
                self.error = _describe(failure)
                logger.info("football-data.org did not answer: %s", self.error)
                return ()
            # Keyed rather than concatenated: consecutive chunks meet at an
            # instant the feed includes in both, so a midnight kick-off — every
            # Brazilian evening match is one — would otherwise be two cards.
            for one in answered:
                found.setdefault(one.match_id, one)
        self.error = None
        return tuple(found.values())


# ---- the request -------------------------------------------------------------


@lru_cache(maxsize=32)
def _fetch(
    base_url: str,
    api_key: str,
    params: tuple[tuple[str, str], ...],
    http: HttpClient | None,
    _bucket: int,
) -> tuple[Fixture, ...]:
    """``GET /v4/matches``, parsed. Cached for one :data:`CACHE_SECONDS` bucket.

    A time bucket rather than an expiry check, because ``lru_cache`` already
    holds the entries, bounds them and evicts them — the whole of a TTL cache
    that would otherwise be a dict, a clock and a prune nobody tests.
    """
    with _client(http) as client:
        response = client.get(
            f"{base_url}/matches",
            params=dict(params),
            headers={"X-Auth-Token": api_key},
        )
    payload = response.json()
    return to_fixtures(payload.get("matches", []) if isinstance(payload, Mapping) else [])


def _bucket() -> int:
    """Which :data:`CACHE_SECONDS` window we are in. Monotonic, so a clock
    adjustment cannot send the cache backwards."""
    return int(time.monotonic() // CACHE_SECONDS)


@contextmanager
def _client(http: HttpClient | None) -> Iterator[HttpClient]:
    """The injected client, or one owned and closed here."""
    if http is not None:
        yield http
        return
    with HttpClient(timeout_seconds=TIMEOUT_SECONDS, max_retries=MAX_RETRIES) as made:
        yield made


def _codes(competitions: Sequence[str] | None) -> str | None:
    """The feed's codes for the competitions asked for.

    Three answers, and they are all different: ``""`` is "no filter, everything
    the plan covers", a comma-joined string is the filter, and ``None`` is
    "every competition asked for is outside this plan" — which is a sentence
    for the page rather than an unfiltered request for competitions the reader
    did not ask about.
    """
    if not competitions:
        return ""
    codes = [COMPETITION_CODES[one] for one in competitions if one in COMPETITION_CODES]
    return ",".join(codes) if codes else None


def _chunks(since: dt.date, until: dt.date) -> list[tuple[dt.date, dt.date]]:
    """The window, as inclusive day ranges the feed will accept one at a time.

    One entry for anything up to :data:`MAX_WINDOW_DAYS` days, which is every
    ask the shipped dashboard makes. A wider ask becomes consecutive windows rather
    than a truncated one: dropping the tail would be a page that is missing
    fixtures with nothing on it saying so, which is the failure mode this whole
    provider layer is arranged against.

    An inverted window is no requests at all — the feed would answer nothing
    and the loop below would not terminate.
    """
    if until < since:
        return []
    span = dt.timedelta(days=MAX_WINDOW_DAYS - 1)
    windows: list[tuple[dt.date, dt.date]] = []
    start = since
    while start <= until:
        end = min(start + span, until)
        windows.append((start, end))
        start = end + dt.timedelta(days=1)
    return windows


def _describe(failure: Exception) -> str:
    """Why the feed did not answer, in a sentence a reader can act on.

    **The body, not the status line.** This feed sends an empty HTTP reason
    phrase and puts the whole explanation in a JSON ``message`` — an invalid
    token and a window that is too wide are both a bare ``400``, and the
    difference between them is the only useful half. Reading ``reason`` alone
    produced ``"football-data.org answered 400: "``, which is what a reader
    would have had to debug from.
    """
    response = getattr(failure, "response", None)
    if response is not None:
        return f"football-data.org answered {response.status_code}: {_message(response)}"
    return f"football-data.org could not be reached: {failure}"


def _message(response: Any) -> str:
    """The feed's own explanation, or the status line when it sent none."""
    try:
        said = response.json()
    except ValueError:
        said = None
    if isinstance(said, Mapping) and said.get("message"):
        return str(said["message"])
    return str(getattr(response, "reason", "") or "no detail given")


# ---- the wire, as domain types -----------------------------------------------


def to_fixtures(rows: Sequence[Mapping[str, Any]]) -> tuple[Fixture, ...]:
    """Every row this application can render, in the order the feed gave them."""
    made = (as_fixture(row) for row in rows if isinstance(row, Mapping))
    return tuple(one for one in made if one is not None)


def as_fixture(row: Mapping[str, Any]) -> Fixture | None:
    """One ``/v4/matches`` row, or ``None`` when it is not a match this can show.

    Dropped rather than rendered: a competition this project has no id for
    (the feed's plan includes internationals and cups the registry does not
    carry), a row with no parseable kick-off, and a fixture whose teams are not
    both known — a cup round drawn but not decided has null teams, and "TBD vs
    TBD" on a card is furniture, not information.
    """
    competition = _mapping(row.get("competition"))
    ours = OUR_ID.get(str(competition.get("code", "")))
    when = _local(row.get("utcDate"))
    home, away = _mapping(row.get("homeTeam")), _mapping(row.get("awayTeam"))
    if ours is None or when is None or not (_team(home) and _team(away)):
        return None
    score = _mapping(_mapping(row.get("score")).get("fullTime"))
    return Fixture(
        match_id=f"fdorg-{row.get('id')}",
        competition_id=ours,
        date=when.date(),
        home_team=_team(home),
        away_team=_team(away),
        status=STATUSES.get(str(row.get("status")), MatchStatus.UNKNOWN),
        competition=str(competition.get("name")) or None,
        country=_text(_mapping(row.get("area")).get("name")),
        kickoff=_kickoff(when),
        home_goals=_as_int(score.get("home")),
        away_goals=_as_int(score.get("away")),
        minute=_as_int(row.get("minute")),
        home_crest_url=_crest(home),
        away_crest_url=_crest(away),
    )


def _order(fixture: Fixture) -> tuple[dt.date, str]:
    """Earliest first, by kick-off within a day. ``"HH:MM"`` sorts as text."""
    return fixture.date, fixture.kickoff or ""


def _mapping(value: Any) -> Mapping[str, Any]:
    """A nested object, or an empty one. Every branch of this feed's tree is
    nullable — an unfinished match has ``score.fullTime.home = null`` and an
    undrawn tie has a null team — so reaching two levels in means checking."""
    return value if isinstance(value, Mapping) else {}


def _team(team: Mapping[str, Any]) -> str:
    """The club as this feed names it, preferring its own short form.

    ``shortName`` is "Man United" where ``name`` is "Manchester United FC" —
    closer to the canonical table's spelling, and the form that fits a card.
    No alias table: nothing here is joined to an ingested row, so a mapping
    between the two vocabularies would be a guess with no reader.
    """
    return _text(team.get("shortName")) or _text(team.get("name")) or ""


def _text(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _crest(team: Mapping[str, Any]) -> str | None:
    """The club badge, if the feed gave an ``https`` one.

    The scheme is checked because this is a trust boundary: the URL is written
    into an ``<img src>`` by :func:`dashboard.ui.crest_html`, and a value that
    arrived over the network deciding what a browser fetches is a decision this
    module gets to make once, here.
    """
    crest = _text(team.get("crest"))
    return crest if crest.startswith("https://") else None


def _local(value: Any) -> dt.datetime | None:
    """``"2026-09-06T15:00:00Z"`` in the host's timezone, or ``None``.

    Converted here and nowhere else, so that a ``Fixture``'s ``date`` is in the
    same calendar as the ``date.today()`` its caller filtered on. A stamp that
    arrives without an offset is read as UTC rather than as local time: this
    field is named ``utcDate``, and assuming the host's zone for it would put a
    match on the wrong evening in the one case the feed changed its mind about
    formatting.
    """
    text = _text(value)
    try:
        when = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.UTC)
    return when.astimezone()


def _kickoff(when: dt.datetime) -> str:
    """``"20:30 IST"``. The zone is on the card because the time is the
    *host's*, not the reader's — a deployed dashboard renders server-side, and
    an unlabelled clock is the one a reader assumes is their own."""
    return f"{when:%H:%M} {when:%Z}".strip()


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
