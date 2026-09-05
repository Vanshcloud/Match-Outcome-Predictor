"""Which provider answers, chosen in one place.

The registry, and the two functions that read it. Nothing in
:mod:`dashboard.views` names a provider class, imports a provider module or
knows how many exist — a view asks :func:`fixtures` for the fixture feed and
renders whatever comes back, including nothing.

**This is the Milestone 13 change, in full.** Write a class implementing
:class:`~dashboard.providers.base.FixtureProvider` against football-data.org,
API-Football or SportMonks; add one entry to :data:`FIXTURE_PROVIDERS`; set
``DASHBOARD_FIXTURE_PROVIDER``. No view moves, no card changes, and the empty
states stop being empty. The same shape takes an odds provider at Milestone 17
and a live socket at Milestone 15.

**An unknown name falls back rather than failing.** A dashboard that refused to
start because of a typo in an environment variable, on a page whose live
section is one of six, would be a worse failure than the misconfiguration.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from dashboard.providers.base import FixtureProvider
from dashboard.providers.null import NullFixtures
from src.utils.logging import get_logger

logger = get_logger(__name__)

FIXTURE_PROVIDER_ENV = "DASHBOARD_FIXTURE_PROVIDER"
DEFAULT_FIXTURE_PROVIDER = "none"

FIXTURE_PROVIDERS: dict[str, Callable[[], FixtureProvider]] = {"none": NullFixtures}
"""Every fixture feed this build can use, by the name that selects it.

One entry today, and it is the one that returns nothing. That is not a
placeholder in the pejorative sense: it is a real implementation of the
interface, it is what the views are written against, and every empty state on
this dashboard is rendered from its answer rather than from a special case.
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
