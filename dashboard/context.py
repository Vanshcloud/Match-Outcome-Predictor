"""The providers one page render needs, resolved once and handed round.

Six views, each wanting some of four sources and the settings. Building
those inside each view would mean six places that know how a client is
constructed and six places to change when a fourth provider is added.

**Built per script run, not cached.** Streamlit reruns the whole script on
every interaction, and everything here is cheap to construct: parsing the
settings is a pydantic model over a small YAML file, and
:class:`~dashboard.client.PredictionClient` is a frozen dataclass that opens no
connection until it is called. A cached context would be a cached client
holding a session across reruns, which is exactly the lifetime that class
documents itself as refusing to hold.

**This is the only module that names concrete providers.** Views take what is
on the context and ask it questions; none of them imports
:mod:`dashboard.providers.historical` or knows that the null fixture feed
exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dashboard import providers
from dashboard.client import PredictionClient
from dashboard.providers.api import ApiPredictions
from dashboard.providers.base import FixtureProvider, Notifier
from dashboard.providers.historical import HistoricalResults
from src.ingestion.base import MATCHES_FILENAME
from src.utils.config import Settings, load_settings


@dataclass(frozen=True, slots=True)
class Context:
    """The resolved dependencies of one page render."""

    settings: Settings
    results: HistoricalResults
    predictions: ApiPredictions
    fixtures: FixtureProvider
    notifier: Notifier

    @property
    def reports_dir(self) -> Path:
        return self.settings.paths.reports_dir

    @property
    def matches_path(self) -> str:
        """Where the match table is, as the string the caches are keyed on."""
        return str(self.results.path)

    @property
    def has_matches(self) -> bool:
        """Whether there is a match table at all — the clean-checkout question."""
        return self.results.available


def resolve() -> Context:
    """Resolve the settings, the three providers and the transport.

    Called as ``context.resolve()`` from every view rather than imported by
    name, deliberately: a module attribute is looked up when it is called, so a
    test replaces the providers for a whole page with one ``monkeypatch``
    against this module — and an imported name would already be bound.
    """
    settings = load_settings()
    client = PredictionClient(
        base_url=settings.dashboard.api_url,
        timeout_seconds=settings.dashboard.request_timeout_seconds,
    )
    return Context(
        settings=settings,
        results=HistoricalResults(settings.paths.processed_dir / MATCHES_FILENAME),
        predictions=ApiPredictions(client=client),
        fixtures=providers.fixtures(),
        notifier=providers.notifier(),
    )
