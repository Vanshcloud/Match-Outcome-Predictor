"""The interfaces a football data source implements.

This is the layer every data source plugs into, and the reason the rest of the
package does not depend on any of them. A view asks a provider
for matches; it never asks *which* provider, and it never learns whether the
answer came from a Parquet file, an HTTP feed or a websocket.

**Four protocols, one of them a transport, because they are genuinely different
questions.** Each was added without a view moving to take it.

:class:`ResultProvider`
    What has been played. Answered today by
    :class:`~dashboard.providers.historical.HistoricalResults`, over the table
    the ingestion pipeline wrote.

:class:`FixtureProvider`
    What is *going* to be played, and what is being played right now. Answered
    by :class:`~dashboard.providers.football_data_org.FootballDataOrgFixtures`
    where a key is configured, and by
    :class:`~dashboard.providers.null.NullFixtures` — which returns nothing and
    says so — where one is not. The live feed is one class satisfying this
    protocol.

:class:`Notifier`
    Where an event is sent. Answered by
    :class:`~dashboard.providers.webhook.WebhookNotifier` where one is
    configured, and by :class:`~dashboard.providers.null.NullNotifier` — which
    delivers nothing and says so — where one is not. A fourth *kind* of thing
    rather than a fourth source: the three above answer questions, this one is
    told something.

:class:`PredictionProvider`
    What the model says. Answered by
    :class:`~dashboard.providers.api.ApiPredictions`, over HTTP, because there
    must be exactly one process in this system that holds the model.

They are separate because a source that has one rarely has the others: the feed
that knows tonight's kick-off times has no forecast, and the service that has
the forecast does not know what kicked off. A single provider interface would
be an interface most implementations answered ``None`` to.

**Every implementation is responsible for its own failures.** A provider that
cannot be reached returns what it has — usually nothing — rather than raising
into a page that is mostly about something else. ``available`` is how a view
tells "no matches tonight" from "no provider", which are different sentences to
put on a screen and the reason this attribute exists at all.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from dashboard.domain.match import Fixture, MatchEvent, Prediction


@runtime_checkable
class Provider(Protocol):
    """What every provider can be asked, whatever it provides.

    Both members are declared as read-only properties rather than as
    attributes. A protocol attribute is a *settable* variable and an
    implementation whose ``available`` is a computed property does not satisfy
    one — and ``available`` should be computed: the match table appears the
    moment `make data` finishes, and a provider that stored the answer would go
    on reporting the empty state it was constructed in.
    """

    @property
    def name(self) -> str:
        """How this provider is named in the registry, in a caption and in a log."""

    @property
    def available(self) -> bool:
        """Whether it can answer at all right now.

        False is a normal state, not an error: a clean checkout has no match
        table and a fresh deployment has no fixture feed. Views render why.
        """


@runtime_checkable
class ResultProvider(Provider, Protocol):
    """Matches that have been played, with the scoreline."""

    def results(
        self,
        *,
        competitions: Sequence[str] | None = None,
        since: dt.date | None = None,
        until: dt.date | None = None,
        limit: int | None = None,
    ) -> list[Fixture]:
        """Finished matches in the window, most recent first."""

    def latest(self) -> dt.date | None:
        """The date of the most recent match this provider holds.

        How a page says "results through Sunday" without loading every row,
        and how it avoids reporting an empty week when the answer is that
        nobody has run `make data` this month.
        """


@runtime_checkable
class FixtureProvider(Provider, Protocol):
    """Matches that have not finished: scheduled, and in play."""

    def scheduled(
        self,
        *,
        since: dt.date,
        until: dt.date,
        competitions: Sequence[str] | None = None,
    ) -> list[Fixture]:
        """Matches kicking off in the window, earliest first."""

    def live(self, *, competitions: Sequence[str] | None = None) -> list[Fixture]:
        """Matches in play, with the minute where the provider gives one."""

    def crests(self, competition_id: str) -> Mapping[str, str]:
        """Club crest URLs in a competition, keyed by normalised club name; empty when none."""


@runtime_checkable
class PredictionProvider(Provider, Protocol):
    """What the model says about a fixture, and which fixtures it can say it about."""

    def priceable(
        self,
        *,
        competitions: Sequence[str] | None = None,
        limit: int = 20,
        since: dt.date | None = None,
        until: dt.date | None = None,
    ) -> list[Fixture]:
        """Fixtures this model has the inputs to price."""

    def predict(self, match_id: str) -> Prediction | None:
        """One fixture, priced. ``None`` when the provider could not answer."""


@runtime_checkable
class Notifier(Protocol):
    """Somewhere an event can be sent: a webhook, and one day a phone.

    Not a provider — nothing is fetched — but it lives beside them and is
    chosen the same way, because "which transport is configured" is the same
    question as "which feed is configured" and one registry is better than two.

    ``send`` returns whether the event went, and never raises. A transport that
    is refusing must not take down a live section that is otherwise working:
    the reader came for the football, and the reason the webhook is unhappy
    belongs in a caption rather than in a stack trace.
    """

    @property
    def name(self) -> str:
        """How this transport is named in the registry and in a caption."""

    @property
    def available(self) -> bool:
        """Whether it is configured. ``False`` is the ordinary default."""

    def send(self, event: MatchEvent) -> bool:
        """Deliver one event. ``False`` when it did not go."""
