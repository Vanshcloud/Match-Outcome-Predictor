"""The competitions this application knows about.

**The catalogue is the registry, not a second list.** ``configs/leagues.yaml``
is where a competition is added to this project, and it is asserted by
``tests/unit/test_registry.py`` to be the only place. A dashboard with its own
copy of the league names would be a second list to keep in step, and the next
league added would be added to only one of them.

That is why this module reaches into :mod:`src.ingestion.registry` and CI
permits it by name. The registry is a *schema* module — a YAML file and the
pydantic models that validate it, with no provider, no network and no fetch —
in exactly the sense ``src.ingestion.base`` is one for the models and the
metrics, which are allowed it for the same reason.

What a *reader* follows is a different question with a different lifetime, and
it lives in :mod:`dashboard.domain.favourites`.
"""

from __future__ import annotations

from functools import lru_cache

from src.ingestion.registry import Competition, load_registry


@lru_cache(maxsize=1)
def competitions() -> tuple[Competition, ...]:
    """Every registered competition.

    ``lru_cache`` rather than ``st.cache_data``: the return is a tuple of
    pydantic models, which Streamlit's cache would have to pickle on every
    read, and the underlying file is a few kilobytes of YAML parsed once per
    process.
    """
    return load_registry().competitions


def by_id() -> dict[str, Competition]:
    return {competition.id: competition for competition in competitions()}


def label(competition_id: str) -> str:
    """A competition as a reader would name it: ``"England · Premier League"``.

    Falls back to the raw id for a competition the registry does not hold,
    which is what a report from an older run of the pipelines can contain after
    a league has been renamed or removed.
    """
    competition = by_id().get(competition_id)
    if competition is None:
        return competition_id
    return f"{competition.country} · {competition.name}"


def short_label(competition_id: str) -> str:
    """What a card shows: the competition's own name, without the country.

    Unless another country's competition shares that name — Italy's and
    Brazil's "Serie A", England's and Russia's "Premier League" — in which case
    the country is appended: ``"Serie A — Brazil"``.
    """
    competition = by_id().get(competition_id)
    if competition is None:
        return competition_id
    shared = sum(1 for other in competitions() if other.name == competition.name)
    if shared == 1:
        return competition.name
    return f"{competition.name} — {competition.country}"


def by_country() -> dict[str, list[Competition]]:
    """Every competition grouped under its country, both in name order."""
    grouped: dict[str, list[Competition]] = {}
    for competition in sorted(competitions(), key=lambda one: (one.country, one.tier or 99)):
        grouped.setdefault(competition.country, []).append(competition)
    return grouped
