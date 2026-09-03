"""What features exist, what each one reads, and which side of kick-off it draws from.

Milestone 3 wrote down which canonical columns are knowable before a match and
which are not. That was a comment. This is the enforcement.

Every feature declares the canonical columns it consults. From that, one thing
follows automatically: a feature whose ``reads`` touch
:data:`~src.ingestion.base.POST_MATCH_COLUMNS` is *capable* of leaking, and one
that reads only pre-match columns is not. There is no third answer and no field
to get wrong — the classification is derived from the declaration rather than
asserted alongside it.

The declaration alone proves nothing, of course. What proves it is
:mod:`src.validation.temporal`, which recomputes a builder over truncated and
rewritten inputs. The registry's job is to say **which features have something
to prove**, so the suite cannot quietly stop checking one.

It is also the single source of truth for the feature table's schema. A column
that is built but not registered, or registered but not built, fails the
validation suite — because a feature table whose shape is defined in two places
is one that will eventually be defined differently in each.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.ingestion.base import CANONICAL_COLUMNS, POST_MATCH_COLUMNS

if TYPE_CHECKING:
    import pandas as pd

KEY_COLUMN = "match_id"

# Columns every per-team feature needs in order to know whose history to look
# at and where to stop. All pre-match: who is playing, and when.
_IDENTITY: frozenset[str] = frozenset({"date", "home_team_id", "away_team_id"})

_GOALS: frozenset[str] = frozenset({"home_goals", "away_goals"})
_SHOTS: frozenset[str] = frozenset({"home_shots", "away_shots"})


@dataclass(frozen=True, slots=True)
class Feature:
    """One column of the feature table."""

    name: str
    dtype: str
    group: str
    """Which family it belongs to. Used for reporting, and by the model
    milestone to switch a whole block on or off in an ablation."""

    reads: frozenset[str]
    """The canonical columns this feature consults, directly or through a
    window. Declared rather than inferred: a static reading of the code cannot
    tell that a rolling mean of ``points`` came from the goal columns."""

    description: str

    @property
    def can_leak(self) -> bool:
        """True when this feature draws on something unknowable before kick-off.

        Not an accusation — every rolling form feature is in this set, and all
        of them are legitimate. It says the feature has something to prove, and
        the temporal probes are what prove it. A feature reading only pre-match
        columns cannot leak whatever it does with them.
        """
        return bool(self.reads & POST_MATCH_COLUMNS)


class RegistryError(ValueError):
    """The feature registry contradicts itself."""


def _sided(
    suffix: str, dtype: str, group: str, reads: frozenset[str], description: str
) -> tuple[Feature, ...]:
    """One feature per side, from a single declaration.

    Home and away features are the same computation seen from two ends, and
    writing them out twice is how the two drift apart.
    """
    return tuple(
        Feature(
            name=f"{side}_{suffix}",
            dtype=dtype,
            group=group,
            reads=reads,
            description=description.format(side=side),
        )
        for side in ("home", "away")
    )


FEATURES: tuple[Feature, ...] = (
    # -- Form: what the team has been doing lately ---------------------------
    *_sided(
        "matches_played",
        "Int32",
        "form",
        _IDENTITY,
        "Matches the {side} team has played before this date. Pre-match: it "
        "counts fixtures, not results. Carried so a consumer can tell a "
        "warmed-up form figure from a cold one.",
    ),
    *_sided(
        "form_points_5",
        "Float64",
        "form",
        _IDENTITY | _GOALS,
        "Points per game for the {side} team over its previous five matches, " "any competition.",
    ),
    *_sided(
        "goals_for_5",
        "Float64",
        "form",
        _IDENTITY | _GOALS,
        "Goals scored per game by the {side} team over its previous five.",
    ),
    *_sided(
        "goals_against_5",
        "Float64",
        "form",
        _IDENTITY | _GOALS,
        "Goals conceded per game by the {side} team over its previous five.",
    ),
    *_sided(
        "shots_for_5",
        "Float64",
        "form",
        _IDENTITY | _SHOTS,
        "Shots per game by the {side} team over its previous five. Null wherever "
        "the competition's feed carries no shot data, which is most of the "
        "secondary feed and every season before 2000/01.",
    ),
    *_sided(
        "shots_against_5",
        "Float64",
        "form",
        _IDENTITY | _SHOTS,
        "Shots faced per game by the {side} team over its previous five.",
    ),
    *_sided(
        "venue_points_5",
        "Float64",
        "form",
        _IDENTITY | _GOALS,
        "Points per game for the {side} team over its previous five matches "
        "*at this venue*. Home and away form differ by more than the average "
        "home advantage, and a single form figure averages that away.",
    ),
    # -- Schedule: how hard the fixture list has been ------------------------
    *_sided(
        "rest_days",
        "Int16",
        "schedule",
        _IDENTITY,
        "Days since the {side} team last played. Null on its first appearance.",
    ),
    *_sided(
        "matches_14d",
        "Int16",
        "schedule",
        _IDENTITY,
        "Matches the {side} team played in the fortnight before this date.",
    ),
    # -- Head to head --------------------------------------------------------
    Feature(
        name="h2h_matches",
        dtype="Int16",
        group="head_to_head",
        reads=_IDENTITY,
        description="Previous meetings between these two teams, either venue.",
    ),
    Feature(
        name="h2h_home_points",
        dtype="Float64",
        group="head_to_head",
        reads=_IDENTITY | _GOALS,
        description=(
            "Points per game the home team took from its previous five meetings "
            "with this opponent, either venue. Null when they have never met."
        ),
    ),
)


def validate(features: Sequence[Feature]) -> None:
    """Check a registry is internally consistent.

    Raises:
        RegistryError: On a duplicate name, or a column that is not canonical.
            The second is the one that earns its keep: ``reads`` is hand-written,
            and a typo in it would silently reclassify a leaking feature as
            safe.
    """
    names = [feature.name for feature in features]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise RegistryError(f"duplicate feature names: {duplicated}")

    canonical = set(CANONICAL_COLUMNS)
    for feature in features:
        unknown = sorted(feature.reads - canonical)
        if unknown:
            raise RegistryError(f"{feature.name} reads non-canonical columns: {unknown}")
        if not feature.reads:
            raise RegistryError(f"{feature.name} declares no inputs")


validate(FEATURES)

FEATURE_SCHEMA: dict[str, str] = {
    KEY_COLUMN: "string",
    **{feature.name: feature.dtype for feature in FEATURES},
}
"""The feature table's shape, derived from the registry so there is one of it."""

FEATURE_COLUMNS: tuple[str, ...] = tuple(feature.name for feature in FEATURES)


def by_name(name: str) -> Feature:
    for feature in FEATURES:
        if feature.name == name:
            return feature
    raise KeyError(name)


def groups(features: Iterable[Feature] | None = None) -> dict[str, tuple[Feature, ...]]:
    """Features by group, in registry order."""
    grouped: dict[str, list[Feature]] = {}
    for feature in features if features is not None else FEATURES:
        grouped.setdefault(feature.group, []).append(feature)
    return {name: tuple(members) for name, members in grouped.items()}


@runtime_checkable
class FeatureBuilder(Protocol):
    """Produces some of the registered features.

    A Protocol, matching every other seam in this project. Builders are grouped
    by the computation they share — a rolling window over a team's own history,
    a window over a pair's meetings — rather than one class per column, because
    the expensive part is reshaping the table and doing it per feature would
    reshape it twenty times.
    """

    @property
    def name(self) -> str:
        """Short identifier, used in logs and in the causality report."""
        ...

    @property
    def features(self) -> tuple[Feature, ...]:
        """The registered features this builder fills."""
        ...

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return one row per input match, keyed by :data:`KEY_COLUMN`.

        Values are what was knowable before each match kicked off. Where there
        is no history the answer is null, not zero: a team with no previous
        matches has no form, and calling that a form of zero would teach a model
        that every debutant is terrible.
        """
        ...
