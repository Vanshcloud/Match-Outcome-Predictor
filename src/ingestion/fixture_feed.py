"""Upcoming fixtures from football-data.org, rewritten in the results provider's spellings.

:mod:`src.ingestion.fixtures` reads the results provider's own ``fixtures.csv``,
which is the right source and a short one: it is published a few days at a
time, so on a Monday it can hold nothing past Monday night while the dashboard's
live feed already lists the whole week. A fixture that is on the home page and
in no design table is a card with no forecast.

This module fills that gap without a second identity. It turns each feed match
into a row **shaped exactly like a line of** ``fixtures.csv`` — the ``Div`` code,
the UK calendar date, the UK kick-off time, and the club names as the results
provider spells them — and hands those rows to :func:`fixtures.to_frame`, which
builds the match id, the season and the team ids the same way it does for the
provider's own file. So a forecast made from a feed row joins to the played
match next week like any other.

**The club name is the part that can be wrong, so it is never guessed.** The feed
says "Club Atlético de Madrid" where the table says "Ath Madrid". A name
resolves by, in order: :data:`ALIASES`; every word of a table club being the
start of a word in the feed's short name ("Hull" in "Hull City"); every word of
the feed's short name being the start of a word of a table club ("PSV" in "PSV
Eindhoven"). Only a single candidate counts. The full name is never matched on
words: "RCD Espanyol de Barcelona" contains Barcelona, which is a different club.
A club that resolves to none, or to two, drops its fixture and is named in the log — a
fixture with a misspelled club would get a design row with no history, which is
a wrong forecast rather than a missing one.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from src.ingestion.fixtures import fixture_divisions
from src.ingestion.registry import Registry
from src.utils.http import HttpClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.football-data.org/v4"
API_KEY_ENV = "FOOTBALL_DATA_API_KEY"
UK = ZoneInfo("Europe/London")
"""The results provider files dates and kick-offs in UK time."""

UPCOMING_STATUSES = frozenset({"SCHEDULED", "TIMED"})
DOCUMENTED_STATUSES = frozenset(
    {*UPCOMING_STATUSES, "IN_PLAY", "PAUSED", "LIVE", "FINISHED"}
    | {"POSTPONED", "SUSPENDED", "CANCELLED", "AWARDED"}
)
"""Every status the feed documents. Anything else is the feed misreporting one:
measured on 2026-09-14, some responses carry each Brazilian match's kick-off
time in ``status`` in place of ``TIMED``, the same request answering correctly
seconds later. An unscored match with such a status is taken as upcoming."""

MAX_WINDOW_DAYS = 10
"""The longest ``dateFrom``..``dateTo`` range the feed answers in one request."""

COMPETITIONS: Mapping[str, str] = {
    "PL": "ENG_1",
    "ELC": "ENG_2",
    "BL1": "GER_1",
    "SA": "ITA_1",
    "PD": "ESP_1",
    "FL1": "FRA_1",
    "DED": "NED_1",
    "PPL": "POR_1",
    "BSA": "BRA_1",
}
"""The feed's codes for the free-plan competitions this project has a history for."""

ALIASES: Mapping[str, str] = {
    "Nottingham Forest FC": "Nott'm Forest",
    "Wolverhampton Wanderers FC": "Wolves",
    "Sheffield United FC": "Sheffield United",
    "FC Bayern München": "Bayern Munich",
    "Hamburger SV": "Hamburg",
    "FC Internazionale Milano": "Inter",
    "Athletic Club": "Ath Bilbao",
    "Club Atlético de Madrid": "Ath Madrid",
    "RCD Espanyol de Barcelona": "Espanol",
    "FC Barcelona": "Barcelona",
    "RC Deportivo La Coruña": "La Coruna",
    "Stade Rennais FC 1901": "Rennes",
    "Paris Saint-Germain FC": "Paris SG",
    "NEC": "Nijmegen",
    "Sporting Clube de Portugal": "Sp Lisbon",
    "Vitória SC": "Guimaraes",
    "CF Estrela da Amadora": "Estrela",
    "CA Mineiro": "Atletico-MG",
    "CA Paranaense": "Athletico-PR",
}
"""The feed's full club name against the results provider's spelling, where no
word rule finds it. Every club of the nine competitions' current rosters was
resolved and read by eye; these are the ones the rules could not reach. For
Brazil, last round's finished matches rebuilt the exact ids the table holds."""


def fetch(
    client: HttpClient, api_key: str, *, since: dt.date, days: int
) -> list[Mapping[str, object]]:
    """Every match the feed lists from ``since`` for ``days`` days, in those competitions."""
    found: list[Mapping[str, object]] = []
    start = since
    end = since + dt.timedelta(days=days)
    while start <= end:
        stop = min(end, start + dt.timedelta(days=MAX_WINDOW_DAYS - 1))
        response = client.get(
            f"{BASE_URL}/matches",
            headers={"X-Auth-Token": api_key},
            params={
                "dateFrom": start.isoformat(),
                "dateTo": stop.isoformat(),
                "competitions": ",".join(COMPETITIONS),
            },
        )
        payload = response.json()
        matches = payload.get("matches", []) if isinstance(payload, Mapping) else []
        found.extend(one for one in matches if isinstance(one, Mapping))
        start = stop + dt.timedelta(days=1)
    return found


def to_rows(
    matches: Iterable[Mapping[str, object]],
    registry: Registry,
    clubs: Mapping[str, Sequence[str]],
) -> list[dict[str, str]]:
    """Feed matches as ``fixtures.csv`` rows, dropping any club that does not resolve.

    Args:
        matches: What :func:`fetch` returned.
        registry: Supplies each competition's ``Div`` code.
        clubs: The table's club spellings by competition id — see :func:`recent_clubs`.
    """
    divisions = {one.id: code for code, one in fixture_divisions(registry).items()}
    rows: list[dict[str, str]] = []
    unresolved: set[str] = set()
    for match in matches:
        if not _upcoming(match):
            continue
        competition = _mapping(match.get("competition")).get("code")
        competition_id = COMPETITIONS.get(str(competition))
        kickoff = _uk_time(match.get("utcDate"))
        if competition_id not in divisions or kickoff is None:
            continue
        known = clubs.get(str(competition_id), ())
        sides = [_mapping(match.get("homeTeam")), _mapping(match.get("awayTeam"))]
        names = [resolve(side, known) for side in sides]
        if None in names:
            unresolved.update(
                str(side.get("name")) for side, name in zip(sides, names, strict=True) if not name
            )
            continue
        rows.append(
            {
                "Div": divisions[str(competition_id)],
                "Date": kickoff.strftime("%d/%m/%Y"),
                "Time": kickoff.strftime("%H:%M"),
                "HomeTeam": str(names[0]),
                "AwayTeam": str(names[1]),
            }
        )
    if unresolved:
        logger.warning(
            "%d club name(s) from the feed matched no single table club, their "
            "fixtures skipped (add them to ALIASES): %s",
            len(unresolved),
            ", ".join(sorted(unresolved)),
        )
    return rows


def resolve(team: Mapping[str, object], clubs: Sequence[str]) -> str | None:
    """The table's spelling of a feed club, or ``None`` rather than a guess."""
    full = str(team.get("name") or "")
    short = str(team.get("shortName") or "")
    if full in ALIASES:
        return ALIASES[full] if ALIASES[full] in clubs else None
    words = _words(short)
    if not words:
        return None
    inside = [club for club in clubs if _starts(_words(club), set(words))]
    if len(inside) == 1:
        return inside[0]
    around = [club for club in clubs if _starts(words, set(_words(club)))]
    return around[0] if len(around) == 1 else None


def recent_clubs(matches: pd.DataFrame, *, days: int = 400) -> dict[str, list[str]]:
    """Every club that played in each feed competition's country lately, by competition.

    The whole country rather than the one division, and a year and a month rather
    than this season: a newly promoted club's recent matches are in the division
    below, and the first round of a season is not yet in the table at all.
    """
    if matches.empty:
        return {}
    recent = matches[matches["date"] >= matches["date"].max() - pd.Timedelta(days, "D")]
    found: dict[str, list[str]] = {}
    for competition_id in COMPETITIONS.values():
        country = recent[recent["competition_id"].str.startswith(competition_id.split("_")[0])]
        found[competition_id] = sorted(set(country["home_team"]) | set(country["away_team"]))
    return found


def _upcoming(match: Mapping[str, object]) -> bool:
    status = str(match.get("status"))
    if status in UPCOMING_STATUSES:
        return True
    score = _mapping(_mapping(match.get("score")).get("fullTime"))
    return status not in DOCUMENTED_STATUSES and score.get("home") is None


def _starts(words: Sequence[str], within: set[str]) -> bool:
    return bool(words) and all(any(one.startswith(word) for one in within) for word in words)


def _words(name: str) -> list[str]:
    """Lower case, accents folded, apostrophes closed up: "Nott'm" is one word."""
    decomposed = unicodedata.normalize("NFKD", name.lower().replace("'", ""))
    folded = "".join(char for char in decomposed if not unicodedata.combining(char))
    return [word for word in re.split(r"[^a-z0-9]+", folded) if word]


def _uk_time(value: object) -> dt.datetime | None:
    try:
        when = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return when.astimezone(UK) if when.tzinfo is not None else None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
