"""The competition registry: which leagues exist, and where each comes from.

Adding a league is a ``configs/leagues.yaml`` entry, never a code change. That
is the property this module exists to guarantee, and
``tests/unit/test_registry.py`` asserts it by loading the shipped file and
checking every entry resolves to a fetchable location.

Season labels are canonicalised here rather than in the adapter, because the
two feeds disagree about what a season *is*:

- The primary feed encodes it in the URL as ``2425``, always a split
  August-to-May season.
- The secondary feed carries it in a column, as either ``2024`` (a calendar
  season — Brazil, USA, Japan, Norway, Sweden, Ireland) or ``2024/2025`` (a
  split one — Austria, Poland, Romania).

Japan is the case that rules out inferring this per country: the J1 League ran
on calendar years through 2025 and switches to ``2026/2027``. The format is
therefore read per row, never assumed from the competition.

Both are normalised to one string form — ``"2024-25"`` for a split season and
``"2024"`` for a calendar one — so a season label sorts chronologically as
text, which is what makes a temporal split expressible as a comparison.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.ingestion.base import Capability
from src.utils.paths import PROJECT_ROOT

DEFAULT_REGISTRY_PATH: Path = PROJECT_ROOT / "configs" / "leagues.yaml"

# Season codes are two-digit-year pairs. 93 means 1993 and 24 means 2024; the
# feed starts at 1993/94, so anything at or above 93 is twentieth century.
# Valid until 2093, which is a longer horizon than the provider has.
_CENTURY_PIVOT = 93

_SPLIT_LABEL = re.compile(r"^\d{4}-\d{2}$")
_CALENDAR_LABEL = re.compile(r"^\d{4}$")


class Feed(StrEnum):
    """Which of the provider's two file layouts a competition comes from."""

    MAIN = "main"
    """One file per competition-season, rich per-match detail."""

    EXTRA = "extra"
    """One file per country covering all seasons, results and odds only."""


# The most a feed can supply. Actual per-season coverage is narrower and older
# seasons carry less — the primary feed has no shot data before 2000/01 — so
# this is the ceiling used for reporting, not a promise about any one row.
# Coverage that is actually present is measured at parse time.
FEED_CAPABILITIES: dict[Feed, frozenset[Capability]] = {
    Feed.MAIN: frozenset(
        {
            Capability.RESULTS,
            Capability.HALF_TIME,
            Capability.MATCH_STATS,
            Capability.REFEREE,
            Capability.ODDS,
        }
    ),
    Feed.EXTRA: frozenset({Capability.RESULTS, Capability.ODDS}),
}


class Competition(BaseModel):
    """One league or cup this project ingests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[A-Z]{3}_[A-Z0-9]+$")
    """Stable identifier, e.g. ``ENG_1``. Written into every row and used as a
    join key, so it must never change once data has been ingested — which is
    why it is an explicit config value rather than something derived from the
    competition's name."""

    country: str
    name: str
    tier: int | None = Field(default=None, ge=1, le=10)
    feed: Feed
    code: str
    """The provider's own identifier: a division code (``E0``) for the primary
    feed, a country code (``BRA``) for the secondary one."""

    league_filter: str | None = None
    """Selects one competition from a multi-competition country file. Ireland's
    file carries more than one division, so without this its entries would be
    merged into a single impossible league."""

    @field_validator("country", "name", "league_filter")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        """Trim surrounding whitespace.

        Not defensive padding: the provider's own data contains ``'Ireland '``
        alongside ``'Ireland'`` and ``' J1 League'`` alongside ``'J1 League'``.
        A filter that does not account for that silently splits one competition
        into two, each with half its history.
        """
        return value.strip() if value is not None else None

    @property
    def capabilities(self) -> frozenset[Capability]:
        """The ceiling of what this competition's feed can supply."""
        return FEED_CAPABILITIES[self.feed]

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities


class Registry(BaseModel):
    """Every competition, loaded from YAML."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    earliest_season: int = Field(default=1993, ge=1888)
    """First season year to attempt for the primary feed. Availability differs
    per division and the provider gives no index, so the adapter probes forward
    from here and treats "missing" as a normal answer rather than an error."""

    competitions: tuple[Competition, ...]

    @field_validator("competitions")
    @classmethod
    def _ids_are_unique(cls, value: tuple[Competition, ...]) -> tuple[Competition, ...]:
        """A duplicate id would make two leagues share a key and merge silently
        at every join downstream."""
        seen: dict[str, str] = {}
        for competition in value:
            if competition.id in seen:
                raise ValueError(
                    f"duplicate competition id {competition.id!r}: "
                    f"{seen[competition.id]!r} and {competition.name!r}"
                )
            seen[competition.id] = competition.name
        return value

    def by_id(self, competition_id: str) -> Competition:
        """Look up one competition, or raise naming what is available."""
        for competition in self.competitions:
            if competition.id == competition_id:
                return competition
        raise KeyError(
            f"unknown competition {competition_id!r}; "
            f"known ids: {', '.join(sorted(c.id for c in self.competitions))}"
        )

    def for_feed(self, feed: Feed) -> tuple[Competition, ...]:
        return tuple(c for c in self.competitions if c.feed is feed)


def load_registry(path: Path | None = None) -> Registry:
    """Load and validate ``configs/leagues.yaml``."""
    target = path or DEFAULT_REGISTRY_PATH
    with target.open("r", encoding="utf-8") as handle:
        tree: dict[str, Any] = yaml.safe_load(handle) or {}
    return Registry.model_validate(tree)


# --- Season labels -----------------------------------------------------------


def season_code_to_label(code: str) -> str:
    """``"2425"`` -> ``"2024-25"``. The primary feed's URL encoding.

    Raises:
        ValueError: If ``code`` is not four digits or the halves are not
            consecutive years — ``"2426"`` is a typo, not a season.
    """
    if not re.fullmatch(r"\d{4}", code):
        raise ValueError(f"season code must be four digits, got {code!r}")
    start_yy, end_yy = int(code[:2]), int(code[2:])
    if (start_yy + 1) % 100 != end_yy:
        raise ValueError(f"season code {code!r} does not span consecutive years")
    century = 1900 if start_yy >= _CENTURY_PIVOT else 2000
    return f"{century + start_yy}-{end_yy:02d}"


def season_label_to_code(label: str) -> str:
    """``"2024-25"`` -> ``"2425"``. Inverse of :func:`season_code_to_label`."""
    if not _SPLIT_LABEL.match(label):
        raise ValueError(f"not a split-season label: {label!r}")
    start_year = int(label[:4])
    return f"{start_year % 100:02d}{label[5:]}"


def normalise_season(raw: str) -> str:
    """Normalise a secondary-feed ``Season`` value to a canonical label.

    Accepts ``"2024"`` (calendar) and ``"2024/2025"`` or ``"2024/25"`` (split),
    which is the full set observed across the sixteen country files. Values
    arrive as strings because the column's pandas dtype is ``int64`` for some
    countries and ``object`` for others — Japan alone contains both.

    Raises:
        ValueError: On any other shape, rather than guessing.
    """
    text = raw.strip()
    if _CALENDAR_LABEL.match(text):
        return text
    match = re.fullmatch(r"(\d{4})/(\d{2}|\d{4})", text)
    if match:
        start, end = match.group(1), match.group(2)
        end_yy = end[-2:]
        if (int(start[2:]) + 1) % 100 != int(end_yy):
            raise ValueError(f"season {raw!r} does not span consecutive years")
        return f"{start}-{end_yy}"
    raise ValueError(f"unrecognised season format: {raw!r}")


def season_start_year(label: str) -> int:
    """The calendar year a season starts in, for both label forms."""
    if _SPLIT_LABEL.match(label) or _CALENDAR_LABEL.match(label):
        return int(label[:4])
    raise ValueError(f"unrecognised season label: {label!r}")
