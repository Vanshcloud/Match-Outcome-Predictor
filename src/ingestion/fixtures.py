"""Matches that have not been played, from the provider that publishes the results.

This project ingests *results*: :mod:`src.ingestion.football_data` drops a row
with no scoreline on sight, because a row with no scoreline is not a match this
project can train on. That rule is right for the canonical table, and without
this module it would leave the service able to price only played fixtures —
every forecast in the served archive would be for a match inside the
artefact's own training window, with **nothing** to score.

The same provider publishes ``/fixtures.csv``: the next week or so of
kick-offs, in the same columns as a season file with the result columns empty.
That is the whole of the source, and choosing it over the dashboard's live
feed is the one decision in this module worth arguing:

**The team names already match.** ``dashboard/providers/football_data_org.py``
spells clubs "Manchester United FC" where the canonical table says "Man
United", and the squad panel needs a whole matching function to join the two.
This file is written by the same hand as the tables, so "Man United" is "Man
United" and there is no matching function here at all. A fixture whose team
names do not resolve is a fixture that gets a design row full of nulls, which
is a wrong forecast rather than a missing one — so the cheapest way to be right
is to not have the problem.

**The identifier has to survive the match being played.** A forecast is joined
to its outcome on ``match_id`` and nothing else, so the id built for a fixture
here must be byte-identical to the one :meth:`FootballDataProvider._to_canonical`
builds when the result lands next week. Same provider, same competition id,
same date, same spelling: everything in the natural key is already shared. The
one part that is not published in this file is the **season**, and that is what
:func:`season_for` exists for.

**Nothing here is written to the canonical table.** These rows have no result
and never will — the *result* arrives through `make data` like every other
match, with the same id. This module hands its frame to
:mod:`src.pipelines.fixtures`, which builds design rows from it and throws the
rows themselves away.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ingestion.base import CANONICAL_SCHEMA, make_match_id
from src.ingestion.csv_reader import ProviderFileError, looks_like_csv, read_provider_csv
from src.ingestion.football_data import (
    BASE_URL,
    MAIN_COLUMNS,
    PROVIDER_NAME,
    parse_date,
    parse_int,
)
from src.ingestion.registry import Competition, Feed, Registry
from src.ingestion.teams import team_id
from src.utils.http import HttpClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

FIXTURES_FILENAME = "fixtures.csv"
"""The published name, and the local one. One file for every division at once."""

DIVISION_COLUMN = "Div"

SEASON_BREAK_DAYS = 30
"""How long a competition must have been idle before a fixture starts a new season.

See :func:`season_for`. Measured rather than chosen: at 30 days the rule
reproduces the season label of 270,829 of the 270,848 ingested matches.
"""


class FixtureFeedError(RuntimeError):
    """The fixture list could not be fetched or is not a fixture list."""


@dataclass(frozen=True, slots=True)
class Played:
    """The most recent match a competition has in the canonical table.

    Two fields because :func:`season_for` needs both: the season is the answer
    it carries forward, and the date is what says whether carrying it forward
    is still sane.
    """

    date: pd.Timestamp
    season: str


def fixtures_url(base_url: str = BASE_URL) -> str:
    return f"{base_url}/{FIXTURES_FILENAME}"


def local_path(raw_dir: Path) -> Path:
    """Where a fetched fixture list lives, beside the season files.

    Exported rather than composed by the caller, because the provider's name is
    in it: CI holds provider details inside this package, and a command that
    joined the same three path parts would be that boundary dissolving one
    string literal at a time.
    """
    return raw_dir / PROVIDER_NAME / FIXTURES_FILENAME


def download(client: HttpClient, raw_dir: Path, *, base_url: str = BASE_URL) -> Path:
    """Fetch the fixture list to ``raw_dir`` and return where it landed.

    Unconditional — no ``ETag``, no cache entry, unlike every other file this
    package fetches. Those are season files, which stop changing once a season
    is settled; this one is rewritten as matches are played and a conditional
    request that answered 304 would hand back last week's fixtures as though
    they were next week's.

    Raises:
        FixtureFeedError: If the provider answered with something that is not a
            CSV. This is the same HTML-with-a-200 failure
            :meth:`FootballDataProvider.ensure_file` guards against, and it is
            checked here for the same reason: a bad file is removed before it
            can be read back as a cached copy.
    """
    destination = local_path(raw_dir)
    url = fixtures_url(base_url)
    client.download(url, destination)
    if not looks_like_csv(destination.read_bytes()):
        destination.unlink(missing_ok=True)
        raise FixtureFeedError(f"{url} answered with a page, not a fixture list")
    logger.info("fetched %s", url)
    return destination


def read(path: Path) -> list[dict[str, str]]:
    """The fixture file as rows, or an empty list when there is no file yet.

    Raises:
        FixtureFeedError: If the file exists and cannot be read as CSV.
    """
    if not path.is_file():
        return []
    try:
        return read_provider_csv(path)
    except ProviderFileError as error:
        raise FixtureFeedError(f"{path}: {error}") from error


def season_for(when: pd.Timestamp, played: Played | None) -> str:
    """Which season label a fixture on ``when`` will be filed under.

    The one part of the natural key this file does not publish, and getting it
    wrong is not a visible failure: the forecast is served and logged with an
    id that the played match will never carry, so it sits in the archive as
    permanently unresolved. So the answer is taken from the data wherever the
    data has one.

    Two rules, in order:

    1. **The competition is mid-season** — it has played inside the last
       :data:`SEASON_BREAK_DAYS` — so this fixture belongs to the season those
       matches belong to. This is the case for almost every fixture, and it is
       the rule that survives a season that does not end when it should: the
       2019-20 season ran into August 2020 across fifteen competitions, and
       every one of those July fixtures is correctly filed by *asking* rather
       than by counting months.
    2. **Otherwise the competition is between seasons**, and the fixture starts
       the one named by its own date. July is the boundary rather than August
       because Ligue 1 has opened in July since 1993 and the provider files
       those matches under the season that is starting.

    Measured over every ingested match by handing each one its predecessor:
    19 of 270,848 labels disagree, and all nineteen are a competition resuming
    after a break longer than a month — the Romanian and Argentine seasons,
    whose structure this project already treats as unusual.
    """
    if played is not None and (when - played.date).days <= SEASON_BREAK_DAYS:
        return played.season
    if played is not None and "-" not in played.season:
        # A calendar-year league (Brazil files "2026"): a new season is the year.
        return str(when.year)
    start = when.year if when.month >= 7 else when.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def fixture_divisions(registry: Registry) -> dict[str, Competition]:
    """Competitions by the ``Div`` code a fixture row carries.

    Every main-feed division, plus each per-country competition that is the only
    one in its file — its country code names it without a league filter. Brazil
    is one; Argentina, whose file mixes a league and a cup, is not.
    """
    divisions = {competition.code: competition for competition in registry.for_feed(Feed.MAIN)}
    for competition in registry.for_feed(Feed.EXTRA):
        if competition.league_filter is None:
            divisions.setdefault(competition.code, competition)
    return divisions


def to_frame(
    rows: list[dict[str, str]],
    registry: Registry,
    *,
    played: Mapping[str, Played] | None = None,
    since: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Provider rows as canonical-shaped fixtures, without a result.

    Every canonical column is present so that the frame concatenates onto the
    match table without widening it — ``home_goals``, ``away_goals`` and
    ``result`` are null, and null is what makes these rows fixtures rather than
    matches. The columns a played match fills afterwards are null too: a
    fixture has no shots and no referee, and inventing a zero for either would
    put a value into a feature window that the match itself is going to
    contradict.

    The odds columns are null on purpose even though this file usually carries
    prices. They are the benchmark this project measures itself against and
    they belong to :mod:`src.evaluation.market`, which reads the closing line
    off the *played* row; a pre-match price copied in here would be a different
    number wearing the same column name.

    Args:
        rows: What :func:`read` returned.
        registry: Used to turn the ``Div`` code into a competition. Divisions
            the registry does not carry are skipped and counted — the provider
            publishes fixtures for divisions this project does not ingest, and
            a fixture for a competition with no history is one the model has
            nothing to say about anyway.
        played: The last match each competition has, by competition id. See
            :func:`season_for`.
        since: Fixtures before this date are dropped. The published file keeps
            the last few days of already-played matches in it, and those are
            not fixtures — the canonical row for them is on its way through
            `make data`.
    """
    divisions = fixture_divisions(registry)
    known = played or {}

    records: list[dict[str, object]] = []
    unknown_divisions: set[str] = set()
    skipped_played = 0
    skipped_past = 0
    skipped_invalid = 0

    for row in rows:
        competition = divisions.get(row.get(DIVISION_COLUMN, "").strip())
        if competition is None:
            code = row.get(DIVISION_COLUMN, "").strip()
            if code:
                unknown_divisions.add(code)
            continue

        mapped = {canonical: row.get(source, "") for source, canonical in MAIN_COLUMNS.items()}
        when = parse_date(mapped["date"])
        home_team = mapped["home_team"].strip()
        away_team = mapped["away_team"].strip()

        if when is None or not home_team or not away_team or home_team == away_team:
            skipped_invalid += 1
            continue
        # A scoreline means the file has caught up with this one. It is a
        # match now, and `make data` is where matches come from.
        if parse_int(mapped["home_goals"]) is not None:
            skipped_played += 1
            continue
        if since is not None and when < since:
            skipped_past += 1
            continue

        records.append(_record(competition, when, home_team, away_team, mapped, known))

    if unknown_divisions:
        logger.info(
            "%d division(s) not in the registry, skipped: %s",
            len(unknown_divisions),
            ", ".join(sorted(unknown_divisions)),
        )
    logger.info(
        "%d fixture(s); skipped %d already played, %d before the window, %d unusable",
        len(records),
        skipped_played,
        skipped_past,
        skipped_invalid,
    )
    frame = pd.DataFrame.from_records(records).reindex(columns=list(CANONICAL_SCHEMA))
    return frame.astype(CANONICAL_SCHEMA).sort_values("date", kind="stable", ignore_index=True)


def _record(
    competition: Competition,
    when: pd.Timestamp,
    home_team: str,
    away_team: str,
    mapped: Mapping[str, str],
    played: Mapping[str, Played],
) -> dict[str, object]:
    """One fixture, keyed exactly as the played match will be keyed.

    Every argument to :func:`~src.ingestion.base.make_match_id` is the one
    :meth:`FootballDataProvider._to_canonical` will pass for the same match
    next week — that identity is the join the whole archive rests on, and it is
    why this builds the id the long way instead of borrowing a shorter key.
    """
    season = season_for(when, played.get(competition.id))
    return {
        "match_id": make_match_id(
            PROVIDER_NAME,
            competition.id,
            season,
            when.strftime("%Y-%m-%d"),
            home_team,
            away_team,
        ),
        "provider": PROVIDER_NAME,
        "competition_id": competition.id,
        "country": competition.country,
        "competition": competition.name,
        "tier": competition.tier,
        "season": season,
        "date": when,
        "kickoff": mapped.get("kickoff", "").strip() or None,
        "home_team": home_team,
        "away_team": away_team,
        "home_team_id": team_id(competition.id, home_team),
        "away_team_id": team_id(competition.id, away_team),
    }


def latest_played(matches: pd.DataFrame) -> dict[str, Played]:
    """The most recent match each competition has, for :func:`season_for`.

    Computed from the canonical frame the caller already loaded rather than
    from a store, which keeps this module a mapper over rows: it is handed
    football and returns football, and it opens nothing.
    """
    if matches.empty:
        return {}
    newest = matches.sort_values("date", kind="stable").drop_duplicates(
        "competition_id", keep="last"
    )
    # Column by column rather than by row. `itertuples` hands the checker a
    # union of every scalar a frame can hold, and `pd.Timestamp` does not
    # accept all of it — three typed series is the same loop without the cast.
    return {
        str(competition): Played(date=when, season=str(season))
        for competition, when, season in zip(
            newest["competition_id"],
            pd.to_datetime(newest["date"]),
            newest["season"],
            strict=True,
        )
    }
