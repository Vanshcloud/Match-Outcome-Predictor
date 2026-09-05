"""The honest answer while no fixture feed is connected.

Not a stub that pretends. It returns nothing, reports ``available = False``,
and the views render that as a sentence naming what is missing and which
milestone connects it.

**Nothing here invents a fixture.** Plausible-looking generated matches would
put a game on the screen that is not being played, and that is the one failure
this application cannot recover from: a reader who catches it once stops
believing the real rows too. An empty section with a reason is a boundary; a
fabricated one is a lie with a countdown clock on it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

from dashboard.domain.match import Fixture


@dataclass(frozen=True, slots=True)
class NullFixtures:
    """No scheduled matches and no live ones, because there is no feed."""

    name: str = "none"
    available: bool = False

    reason: str = (
        "The data behind this project is a **results** feed — a match that has "
        "not been played is in no table here — so kick-off times, live scores "
        "and minutes need a second provider."
    )
    """Why there is nothing, in the words a view puts on the screen.

    On the provider rather than in the view, because the next provider will
    have a different reason — an expired key, a rate limit, a competition the
    plan does not cover — and a view that hard-coded this one would report it
    wrongly the day a real feed fails.
    """

    def scheduled(
        self,
        *,
        since: dt.date,
        until: dt.date,
        competitions: Sequence[str] | None = None,
    ) -> list[Fixture]:
        return []

    def live(self, *, competitions: Sequence[str] | None = None) -> list[Fixture]:
        return []
