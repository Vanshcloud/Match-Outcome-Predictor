"""Which provider answers, chosen in one place.

The registry, and the two functions that read it. Nothing in
:mod:`dashboard.views` names a provider class, imports a provider module or
knows how many exist — a view asks :func:`fixtures` for the fixture feed and
renders whatever comes back, including nothing.

**Adding the live feed was exactly this change.**
:class:`~dashboard.providers.football_data_org.FootballDataOrgFixtures` is one
class implementing :class:`~dashboard.providers.base.FixtureProvider`, one
entry below, and one environment variable. No view moved, no card changed, and
the two forward sections stop being empty. The webhook notifier is registered
the same way.

**An unknown name falls back rather than failing.** A dashboard that refused to
start because of a typo in an environment variable, on a page whose live
section is one of six, would be a worse failure than the misconfiguration.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from dashboard.providers.base import FixtureProvider, Notifier
from dashboard.providers.football_data_org import FootballDataOrgFixtures
from dashboard.providers.null import NullFixtures, NullNotifier
from dashboard.providers.webhook import WebhookNotifier
from src.utils.logging import get_logger

logger = get_logger(__name__)

FIXTURE_PROVIDER_ENV = "DASHBOARD_FIXTURE_PROVIDER"
DEFAULT_FIXTURE_PROVIDER = "none"

NOTIFIER_ENV = "DASHBOARD_NOTIFIER"
DEFAULT_NOTIFIER = "none"


FIXTURE_PROVIDERS: dict[str, Callable[[], FixtureProvider]] = {
    "none": NullFixtures,
    "football-data.org": FootballDataOrgFixtures,
}
"""Every fixture feed this build can use, by the name that selects it.

``none`` stays the default and stays a real implementation rather than a
placeholder: a checkout with no API key is the ordinary state, and every empty
state on this dashboard is rendered from a provider's own answer rather than
from a special case in a view.

``football-data.org`` — note the ``.org``. The results this project *ingests*
come from football-data.co.uk, a different organisation serving static CSV with
no key at all. Two feeds with similar names, and this line is the one place
both appear.
"""


def fixtures(name: str | None = None) -> FixtureProvider:
    """The configured fixture feed, or the null one.

    Read from the environment rather than from
    :class:`~src.utils.config.Settings`, deliberately. The settings model is
    loaded by the pipelines, the API and the dashboard alike, and a section
    only one of the three reads is a setting the other two carry for nothing.
    """
    chosen = name or os.environ.get(FIXTURE_PROVIDER_ENV) or DEFAULT_FIXTURE_PROVIDER
    build = FIXTURE_PROVIDERS.get(chosen)
    if build is None:
        logger.warning(
            "unknown fixture provider %r; known: %s",
            chosen,
            ", ".join(sorted(FIXTURE_PROVIDERS)),
        )
        return NullFixtures()
    return build()


NOTIFIERS: dict[str, Callable[[], Notifier]] = {
    "none": NullNotifier,
    "webhook": WebhookNotifier,
}
"""Where a match event can be sent, by the name that selects it.

The same shape as :data:`FIXTURE_PROVIDERS` and read the same way, because
"which transport is configured" and "which feed is configured" are the same
question asked of different things. ``webhook`` is one entry; a phone or
an inbox is one more entry.
"""


def notifier(name: str | None = None) -> Notifier:
    """The configured transport, or the one that delivers nothing.

    Falls back rather than failing, for the reason the fixture registry does: a
    dashboard that refused to start over a typo in an environment variable —
    on a page whose alerts are one strip of six — would be a worse failure than
    the misconfiguration.
    """
    chosen = name or os.environ.get(NOTIFIER_ENV) or DEFAULT_NOTIFIER
    build = NOTIFIERS.get(chosen)
    if build is None:
        logger.warning("unknown notifier %r; known: %s", chosen, ", ".join(sorted(NOTIFIERS)))
        return NullNotifier()
    return build()
