"""What a match and a forecast *are* to this application.

Every card, every list and every detail page renders a :class:`Fixture`.
Nothing renders a dictionary from an HTTP response or a row of a DataFrame, and
that is the point: this project has three sources of matches and a page written
against one of them is a page that has to be rewritten when the second arrives.

Milestone 17 added two value types rather than a third source of matches:
:class:`MarketPrice` is what the bookmaker said about a fixture, and
:class:`ExpectedGoals` is what a fitted goal-rate model expects from each side.
Both are *about* a match rather than being one, which is why neither is a
field on :class:`Fixture` — a card has no use for either, and a type that grew
every optional fact anybody might want is a type every source has to fill.

Pure value types. No provider, no network, no file, no Streamlit — everything
that fetches lives in :mod:`dashboard.providers` and everything that renders in
:mod:`dashboard.views`, and both import this rather than the other way round.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

OUTCOMES: tuple[str, str, str] = ("home", "draw", "away")
"""The three outcomes, in one order everywhere on this dashboard.

The same order as :data:`src.evaluation.metrics.CLASSES`, and the same *names*
the service answers with — a page showing them in a different order from the
array the model returns is a page where a reader checking one against the other
reads the wrong number. Declared here rather than in the theme because it is
vocabulary before it is presentation: a provider builds a mapping keyed by
these, and the palette is one consumer of that.
"""


class MatchStatus(StrEnum):
    """Where a match is in its life.

    Three states rather than a free-text label, because the page branches on
    it: a scheduled match shows a kick-off time, a live one shows a minute, and
    a finished one shows a score.
    """

    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"

    UNKNOWN = "unknown"
    """A match the source can name but not place in time.

    This is what ``/fixtures`` returns and it is not a defect to be tidied
    away: the service answers "which matches can I price", the answer is drawn
    from a table of played matches, and it withholds the scoreline on purpose.
    Rendering that as ``FINISHED`` would put a blank score on a card and invite
    a reader to conclude the match ended 0-0.
    """


@dataclass(frozen=True, slots=True)
class Fixture:
    """One match, as a card renders it.

    Optional everywhere it can be missing, because a source that has the
    kick-off minute is a different source from the one that has the final
    score, and a page that required both would be a page neither can feed.

    ``crest_url`` is unset by every source in this repository and is read by
    :func:`dashboard.ui.crest_html`, which draws initials instead. It is the
    single field a badge provider has to start filling.
    """

    match_id: str
    competition_id: str
    date: dt.date
    home_team: str
    away_team: str
    status: MatchStatus = MatchStatus.UNKNOWN
    competition: str | None = None
    country: str | None = None
    kickoff: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    minute: int | None = None
    home_crest_url: str | None = None
    away_crest_url: str | None = None

    @property
    def has_score(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    @property
    def teams(self) -> tuple[str, str]:
        return self.home_team, self.away_team

    def involves(self, team: str) -> bool:
        """Whether ``team`` played, matched the way a person types a club name."""
        wanted = " ".join(team.split()).casefold()
        return any(" ".join(name.split()).casefold() == wanted for name in self.teams)


@dataclass(frozen=True, slots=True)
class Prediction:
    """A fixture priced, as the service answered.

    A named type rather than the response dictionary, because four views read
    it and the alternative is four places that know the wire format. Built by
    :func:`dashboard.providers.api.as_prediction`, which is the one place that
    does.

    ``in_sample`` is carried rather than dropped for the reason the API returns
    it: the served model is fitted on the whole history, so a fixture inside
    that history was trained on, and quoting its probability as the
    walk-forward number is the single most likely way a figure from this
    project ends up overstated.
    """

    match_id: str
    probabilities: Mapping[str, float]
    model: str
    model_version: str
    in_sample: bool

    @property
    def outcome(self) -> str:
        """The most likely of the three, by the name the model returns it under."""
        return max(self.probabilities, key=lambda key: self.probabilities[key])

    @property
    def stated(self) -> float:
        """How firmly the model committed: the largest of the three probabilities.

        **Not a calibrated confidence.** What this probability is *worth* is a
        measurement, and it comes from the reliability tables the model card is
        generated from — see :mod:`dashboard.services.reports`.
        """
        return float(self.probabilities[self.outcome])


@dataclass(frozen=True, slots=True)
class MarketPrice:
    """What the bookmaker said, as a page shows it beside what the model said.

    Milestone 17. Three decimal prices, the probabilities they imply once the
    overround is removed, and the overround itself — because the removal is an
    assumption about *how* the margin is spread across three outcomes, and a
    reader who can see its size can judge how much that assumption matters.

    **The de-vig is not done here.** It is
    :func:`src.evaluation.market.implied_probabilities`, the same function the
    backtest's ``bookmaker`` benchmark uses, so the percentages on this page
    are the percentages the model was scored against. A domain type that did
    its own arithmetic would be a second answer to "what did the market say".
    """

    match_id: str
    odds: Mapping[str, float]
    probabilities: Mapping[str, float]
    overround: float


@dataclass(frozen=True, slots=True)
class ExpectedGoals:
    """How many goals a fitted goal-rate model expects from each side.

    Milestone 17, and the label matters more than the numbers. These are the
    Poisson rates Milestone 4's Dixon-Coles model fits — ``dc_home_lambda`` and
    ``dc_away_lambda`` in the ratings table — not shot-quality xG. Nothing in
    this project has ever seen a shot map: the ingested feed carries shots and
    shots on target and no expected-goals column, so a panel labelled "xG"
    would be attributing a rival provider's measurement to a model that made an
    estimate.

    They are worth showing because they are *how the rating thinks*: the
    three-class probability this project reports is a sum over a Poisson grid
    built from exactly these two numbers, so a reader who wants to know why a
    forecast leans one way is looking at its inputs.
    """

    match_id: str
    home: float
    away: float

    @property
    def total(self) -> float:
        """Expected goals in the match, the number a totals market is about."""
        return self.home + self.away

    @property
    def supremacy(self) -> float:
        """Expected goal difference, positive when the home side is favoured."""
        return self.home - self.away


class EventKind(StrEnum):
    """What changed about a match since the page last looked.

    Three, because they are the three a reader would want told: it started,
    somebody scored, it finished. A change in the *minute* is not an event —
    it happens every minute of every match, and a notification that fires
    ninety times per fixture is one a person turns off.
    """

    KICK_OFF = "kick-off"
    GOAL = "goal"
    FULL_TIME = "full-time"


@dataclass(frozen=True, slots=True)
class MatchEvent:
    """One change, as a toast reads it and a webhook posts it.

    Carries the fixture rather than a copy of its fields, so a transport that
    wants the competition, the kick-off time or the crests has them without
    this type growing a column every time one is asked for.
    """

    kind: EventKind
    fixture: Fixture

    @property
    def score(self) -> str:
        """``"2-1"``, or an empty string before there is one."""
        if not self.fixture.has_score:
            return ""
        return f"{self.fixture.home_goals}-{self.fixture.away_goals}"

    @property
    def message(self) -> str:
        """One line a person reads, whatever it is delivered by.

        The same sentence in a toast and in a webhook post on purpose: two
        wordings of one event is two things to keep in step, and the first time
        they disagree is the first time somebody doubts both.
        """
        home, away = self.fixture.teams
        if self.kind is EventKind.KICK_OFF:
            return f"Kick-off: {home} v {away}"
        if self.kind is EventKind.FULL_TIME:
            return f"Full time: {home} {self.score} {away}".replace("  ", " ")
        return f"Goal: {home} {self.score} {away}"

    def as_payload(self) -> dict[str, str | None]:
        """The event as a machine reads it, for a transport that wants fields."""
        return {
            "event": str(self.kind),
            "message": self.message,
            "match_id": self.fixture.match_id,
            "competition_id": self.fixture.competition_id,
            "home_team": self.fixture.home_team,
            "away_team": self.fixture.away_team,
            "score": self.score or None,
            "kickoff": self.fixture.kickoff,
        }
