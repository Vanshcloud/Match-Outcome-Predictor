"""What goes on the home page, assembled once.

Four sections — in play, coming up, just finished, priceable — each a question
put to a different provider and narrowed by what this reader follows. The
assembly is here rather than in :mod:`dashboard.views.home` so that the view is
a list of sections and this is a function a test can call with three stub
providers and no browser.

**A section is a list of fixtures and a reason it might be empty.** Those are
different things and both have to reach the page: "nothing is being played
right now" and "no fixture feed is connected" are the same empty list and
completely different sentences, and a view handed only the list would have to
guess which one to print.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

from dashboard.domain.match import Fixture
from dashboard.providers.base import FixtureProvider, PredictionProvider, ResultProvider

UPCOMING_DAYS = 7
RESULT_CARDS = 12
PRICEABLE_CARDS = 6


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

    @property
    def has_fixtures(self) -> bool:
        return bool(self.fixtures)


@dataclass(frozen=True, slots=True)
class HomePage:
    """Every section, in the order the reader reads them."""

    live: Section
    upcoming: Section
    finished: Section
    priceable: Section

    @property
    def sections(self) -> tuple[Section, ...]:
        return (self.live, self.upcoming, self.finished, self.priceable)


def home_page(
    *,
    fixtures: FixtureProvider,
    results: ResultProvider,
    predictions: PredictionProvider,
    competitions: Sequence[str] | None = None,
    teams: Sequence[str] | None = None,
) -> HomePage:
    """The four sections, each asked of the provider that can answer it."""
    return HomePage(
        live=live_section(fixtures, competitions),
        upcoming=upcoming_section(fixtures, competitions),
        finished=finished_section(results, competitions, teams),
        priceable=priceable_section(predictions, competitions),
    )


def live_section(provider: FixtureProvider, competitions: Sequence[str] | None) -> Section:
    """Matches in play, or the provider's own reason there are none."""
    if not provider.available:
        return Section(
            "Live now",
            "in play, across the competitions you follow",
            unavailable=_reason(provider),
        )
    return Section(
        "Live now",
        "in play, across the competitions you follow",
        fixtures=tuple(provider.live(competitions=competitions)),
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
        return Section("Today and next", note, unavailable=_reason(provider))
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


def finished_section(
    provider: ResultProvider,
    competitions: Sequence[str] | None,
    teams: Sequence[str] | None,
    *,
    days: int = 21,
    limit: int = RESULT_CARDS,
) -> Section:
    """What just finished — the section that has real data today."""
    note = f"the last {days} days of results"
    if not provider.available:
        return Section(
            "Just finished", note, unavailable="No match table yet. Run `make data` to ingest one."
        )
    latest = provider.latest()
    since = latest - dt.timedelta(days=days) if latest is not None else None
    found = provider.results(competitions=competitions, since=since)
    if teams:
        wanted = set(teams)
        found = [one for one in found if wanted & set(one.teams)]
    return Section(
        "Just finished",
        note,
        fixtures=tuple(found[:limit]),
        empty="Nothing finished recently in the competitions you follow.",
    )


def priceable_section(
    provider: PredictionProvider,
    competitions: Sequence[str] | None,
    *,
    limit: int = PRICEABLE_CARDS,
) -> Section:
    """What the service will price, asked of the service.

    Last, and failing quietly, because this is the one section that leaves the
    process: a reader with no service running should still get the three above
    it.
    """
    note = "answered by the prediction service"
    if not provider.available:
        return Section("The model can price these", note, unavailable=_reason(provider))
    return Section(
        "The model can price these",
        note,
        fixtures=tuple(provider.priceable(competitions=competitions, limit=limit)),
        empty="The service has no fixtures indexed yet.",
    )


def _reason(provider: object) -> str:
    """Why a provider cannot answer, in its own words where it has any."""
    stated = getattr(provider, "reason", None) or getattr(provider, "error", None)
    return str(stated) if stated else "No provider is configured for this."
