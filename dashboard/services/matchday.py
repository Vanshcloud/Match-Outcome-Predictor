"""What goes on the home page, assembled once.

Two sections — in play, and coming up with forecasts — each looking forward, each a
question put to a provider and narrowed by what this reader follows. The
assembly is here rather than in :mod:`dashboard.views.home` so that the view is
a list of sections and this is a function a test can call with stub
providers and no browser.

**A section is a list of fixtures and a reason it might be empty.** Those are
different things and both have to reach the page: "nothing is being played
right now" and "no fixture feed is connected" are the same empty list and
completely different sentences, and a view handed only the list would have to
guess which one to print.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from dashboard.domain import competition as catalogue
from dashboard.domain.match import Fixture
from dashboard.providers.base import FixtureProvider, PredictionProvider
from dashboard.providers.football_data_org import COMPETITION_CODES, FEED_ONLY_CODES, club_words

COVERED = tuple(COMPETITION_CODES)
"""The registry competitions the live feed serves and the model is trained on."""

FOLLOWABLE = (*COVERED, *FEED_ONLY_CODES)
"""What a reader can follow: those leagues, plus the feed-only Champions League."""

FEED_ONLY_NAMES = {"UEFA_CL": "UEFA Champions League"}


def label(competition_id: str) -> str:
    """``"England · Premier League"``, or the feed-only competition's own name."""
    if competition_id in FEED_ONLY_NAMES:
        return f"Europe · {FEED_ONLY_NAMES[competition_id]}"
    return catalogue.label(competition_id)


def short_label(competition_id: str) -> str:
    """What a card or the sidebar shows, likewise."""
    return FEED_ONLY_NAMES.get(competition_id) or catalogue.short_label(competition_id)


UPCOMING_DAYS = 7


@dataclass(frozen=True, slots=True)
class Section:
    """One strip of the home page: what to show, or why there is nothing.

    ``unavailable`` is the provider-is-missing case and ``empty`` is the
    provider-answered-nothing case. Keeping them apart is the whole reason this
    type exists: a page that printed "no matches today" when the truth is "no
    fixture feed is connected" would be telling a reader that football has
    stopped.
    """

    title: str
    note: str = ""
    fixtures: tuple[Fixture, ...] = ()
    unavailable: str | None = None
    empty: str = "Nothing to show here."
    probabilities: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    """Home / draw / away by match id, for the cards the model has priced."""

    @property
    def has_fixtures(self) -> bool:
        return bool(self.fixtures)


@dataclass(frozen=True, slots=True)
class HomePage:
    """Every section, in the order the reader reads them."""

    live: Section
    upcoming: Section

    @property
    def sections(self) -> tuple[Section, ...]:
        return (self.live, self.upcoming)


def home_page(
    *,
    fixtures: FixtureProvider,
    predictions: PredictionProvider,
    competitions: Sequence[str] | None = None,
) -> HomePage:
    """The feed's sections, the upcoming cards wearing the model's forecasts.

    A service that is not answering costs the bars, not the cards, and says so
    beside the section title.
    """
    upcoming = upcoming_section(fixtures, competitions)
    if upcoming.has_fixtures and not predictions.available:
        upcoming = replace(upcoming, note=f"{upcoming.note} · no forecasts: {reason(predictions)}")
    return HomePage(
        live=live_section(fixtures, competitions),
        upcoming=with_forecasts(upcoming, predictions),
    )


def live_section(
    provider: FixtureProvider,
    competitions: Sequence[str] | None = None,
    *,
    fixtures: Sequence[Fixture] | None = None,
) -> Section:
    """Matches in play, or the provider's own reason there are none.

    ``fixtures`` is the answer the caller already has.
    :func:`dashboard.services.watch.since_last_look` asks the feed for what is
    in play *and* what changed since the last look in one call, and this
    renders that answer rather than asking a second time — on a free tier of
    ten requests a minute, one paint should cost one request.
    """
    note = "in play, across the competitions you follow"
    if not provider.available:
        return Section("Live now", note, unavailable=reason(provider))
    found = provider.live(competitions=competitions) if fixtures is None else fixtures
    return Section(
        "Live now",
        note,
        fixtures=tuple(found),
        empty="Nothing in play right now.",
    )


def upcoming_section(
    provider: FixtureProvider,
    competitions: Sequence[str] | None,
    *,
    today: dt.date | None = None,
    days: int = UPCOMING_DAYS,
) -> Section:
    """Today and the coming week.

    ``today`` is injectable so a test states the date rather than depending on
    the one the machine happens to be on.
    """
    note = f"the next {days} days"
    if not provider.available:
        return Section("Today and next", note, unavailable=reason(provider))
    start = today or dt.date.today()
    return Section(
        "Today and next",
        note,
        fixtures=tuple(
            provider.scheduled(
                since=start, until=start + dt.timedelta(days=days), competitions=competitions
            )
        ),
        empty="No fixtures scheduled in the next week.",
    )


def club_fixtures(
    provider: FixtureProvider,
    predictions: PredictionProvider,
    clubs: Sequence[str],
    *,
    today: dt.date | None = None,
    days: int = UPCOMING_DAYS,
) -> Section:
    """The upcoming section narrowed to fixtures involving any of ``clubs``, with forecasts.

    ``clubs`` are the match table's names, so each feed card is read through
    its table twin, whose names are the table's own ("Barça" is "Barcelona",
    and "Inter" is not Brazil's "Internacional"). A card with no twin falls
    back to an exact normalised name. The window is the one the home page asks
    for, so it is the same cached request.
    """
    section = upcoming_section(provider, None, today=today, days=days)
    found = twins(section.fixtures, predictions)
    wanted = {club_words(club) for club in clubs}

    def plays(card: Fixture) -> bool:
        twin = found.get(card.match_id, card)
        return bool(wanted & {club_words(twin.home_team), club_words(twin.away_team)})

    mine = replace(
        section,
        title="Live and upcoming",
        fixtures=tuple(one for one in section.fixtures if plays(one)),
        empty=f"No fixtures for {'this club' if len(clubs) == 1 else 'these clubs'} "
        f"in the next {days} days.",
    )
    return with_forecasts(mine, predictions, found=found)


def twins(fixtures: Sequence[Fixture], predictions: PredictionProvider) -> dict[str, Fixture]:
    """Each feed card's match-table fixture, by card id, for the cards that have one.

    One ``priceable`` call for the cards' competitions and dates, not one per
    card; each card is matched by :func:`same_match` against the service's
    fixtures in its competition within a day of it.
    """
    if not fixtures or not predictions.available:
        return {}
    day = dt.timedelta(days=1)
    candidates = predictions.priceable(
        competitions=sorted({one.competition_id for one in fixtures}),
        since=min(one.date for one in fixtures) - day,
        until=max(one.date for one in fixtures) + day,
        limit=200,
    )
    found: dict[str, Fixture] = {}
    for card in fixtures:
        nearby = [
            one
            for one in candidates
            if one.competition_id == card.competition_id and abs(one.date - card.date) <= day
        ]
        twin = same_match(card, nearby)
        if twin is not None:
            found[card.match_id] = twin
    return found


def with_forecasts(
    section: Section,
    predictions: PredictionProvider,
    *,
    found: Mapping[str, Fixture] | None = None,
) -> Section:
    """The section with each feed card's forecast, for the cards the model has priced.

    ``found`` is :func:`twins` when the caller already has it.
    """
    if not section.fixtures or not predictions.available:
        return section
    matched = twins(section.fixtures, predictions) if found is None else found
    probabilities: dict[str, Mapping[str, float]] = {}
    for card in section.fixtures:
        twin = matched.get(card.match_id)
        prediction = predictions.predict(twin.match_id) if twin is not None else None
        if prediction is not None:
            probabilities[card.match_id] = prediction.probabilities
    return replace(section, probabilities=probabilities)


def priced_twin(
    match_id: str,
    *,
    fixtures: FixtureProvider,
    predictions: PredictionProvider,
    today: dt.date | None = None,
) -> str | None:
    """The match-table id of the fixture a feed card shows, so it can be priced.

    The feed card is found again in the window the home page drew it from, then
    matched to the service's fixtures in the same competition within a day
    either side — the feed's date is the reader's, the table's is the UK's.
    ``None`` when the feed no longer has the card, the competition is not in
    the table (the Champions League), or no single fixture matches.
    """
    start = today or dt.date.today()
    card = next(
        (
            one
            for one in fixtures.scheduled(
                since=start - dt.timedelta(days=1),
                until=start + dt.timedelta(days=UPCOMING_DAYS),
            )
            if one.match_id == match_id
        ),
        None,
    )
    if card is None or card.competition_id not in COVERED:
        return None
    candidates = predictions.priceable(
        competitions=[card.competition_id],
        since=card.date - dt.timedelta(days=1),
        until=card.date + dt.timedelta(days=1),
        limit=200,
    )
    found = same_match(card, candidates)
    return found.match_id if found is not None else None


def with_crests(fixtures: Sequence[Fixture], feed: FixtureProvider) -> list[Fixture]:
    """The table's fixtures, wearing their clubs' crests where the feed has them.

    The match table carries names only. Each missing crest is looked up by name
    in the feed's club list for the fixture's competition, one list per
    competition; a club the feed does not list keeps its initials.
    """
    lists: dict[str, Mapping[str, str]] = {}
    dressed = []
    for one in fixtures:
        if one.competition_id not in lists:
            lists[one.competition_id] = crests_around(feed, one.competition_id)
        crests = lists[one.competition_id]
        dressed.append(
            replace(
                one,
                home_crest_url=one.home_crest_url or crest_for(one.home_team, crests),
                away_crest_url=one.away_crest_url or crest_for(one.away_team, crests),
            )
        )
    return dressed


def crests_around(feed: FixtureProvider, competition_id: str) -> Mapping[str, str]:
    """The competition's club list, backed by its country's other covered divisions.

    The feed lists this season's clubs, so a club relegated or promoted since
    the match was played is in the division next door; its own competition's
    list still wins where both name a club.
    """
    known = catalogue.by_id()
    country = known[competition_id].country if competition_id in known else None
    merged: dict[str, str] = {}
    for other in COVERED:
        if country and other != competition_id and known[other].country == country:
            merged.update(feed.crests(other))
    merged.update(feed.crests(competition_id))
    return merged


CREST_ALIASES: Mapping[str, str] = {
    "Athletico-PR": "Paranaense",
    "Atletico-MG": "Mineiro",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton",
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atleti",
    "Espanol": "Espanyol",
    "Rennes": "Stade Rennais",
    "Nijmegen": "NEC",
    "Guimaraes": "Vitoria",
    "Sp Lisbon": "Sporting CP",
}
"""The match table's abbreviations that share no word with the feed's name.

Measured on 2026-09-15 against every club in the covered leagues' latest
season: these eleven, and only these, found no crest by name.
"""


def crest_for(team: str, crests: Mapping[str, str]) -> str | None:
    """The crest of the one club in ``crests`` that ``team`` names, else ``None``.

    The exact name first, then :func:`same_club`; two different crests
    answering is no answer rather than a guess.
    """
    team = CREST_ALIASES.get(team, team)
    exact = crests.get(club_words(team))
    if exact:
        return exact
    found = {url for name, url in crests.items() if same_club(team, name)}
    return found.pop() if len(found) == 1 else None


def same_club(ours: str, theirs: str) -> bool:
    """Whether two spellings name one club: every word of one starts a word of the other.

    "Como" in "Como 1907", "Betis" in "Real Betis", "Acad" in "Academico".
    """
    a, b = club_words(ours).split(), club_words(theirs).split()

    def within(words: list[str], other: list[str]) -> bool:
        return all(any(one.startswith(word) for one in other) for word in words)

    return bool(a and b) and (within(a, b) or within(b, a))


def same_match(card: Fixture, candidates: Sequence[Fixture]) -> Fixture | None:
    """The one candidate that is ``card``, across two spellings of each club.

    A side matches by :func:`same_club`. Both sides matching beats one, then
    the nearer date wins; a tie is no answer. One side is enough on its own
    because a club plays once in a competition on a given day — that is what
    finds "Rio Ave v Amadora" as the table's "Rio Ave v Estrela".
    """

    def score(one: Fixture) -> tuple[int, int]:
        sides = same_club(card.home_team, one.home_team) + same_club(card.away_team, one.away_team)
        return sides, -abs((one.date - card.date).days)

    ranked = sorted(((score(one), one) for one in candidates), key=lambda pair: pair[0])
    if not ranked or ranked[-1][0][0] == 0:
        return None
    if len(ranked) > 1 and ranked[-2][0] == ranked[-1][0]:
        return None
    return ranked[-1][1]


def reason(provider: object) -> str:
    """Why a provider cannot answer, in its own words where it has any."""
    stated = getattr(provider, "reason", None) or getattr(provider, "error", None)
    return str(stated) if stated else "No provider is configured for this."
