"""Every producer, found rather than listed, and probed.

The ratings and features pipelines each run the temporal probes inside their own pipeline, over
their own producers. That is one list per pipeline, and a list is a thing you
can forget to add to: a new rating model or feature builder wired into a
pipeline but omitted from the probe call would ship unverified, and nothing
about the output would say so.

So this module does not take a list. It **walks** :mod:`src.ratings` and
:mod:`src.feature_engineering` and finds every class that satisfies the
producer contract, which means a producer that exists is a producer that gets
probed. :func:`check_defaults_are_complete` closes the other half — a producer
found here but missing from a pipeline's defaults is one that would never run.

It also answers the question the model layer is about to ask: *where does this
column come from?* :func:`audit` traces every derived column back to the
canonical columns it reads, by rewriting each input and seeing what moves. The
feature registry declares the same thing by hand; the audit is what makes the
declaration a claim rather than a comment.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType

import pandas as pd

import src.feature_engineering
import src.ratings
from src.feature_engineering.registry import FEATURES
from src.ingestion.base import BENCHMARK_COLUMNS, CANONICAL_COLUMNS, POST_MATCH_COLUMNS
from src.validation.temporal import (
    Compute,
    TemporalResult,
    observed_reads,
    outcome_independence,
    prefix_invariance,
)

KEY_COLUMN = "match_id"

RATING = "rating"
FEATURE = "feature"

# What each kind of producer is called, and the attribute that identifies one.
# Ratings compute through `rate`, features through `build`; both carry a `name`.
_CONTRACTS: tuple[tuple[ModuleType, str, str], ...] = (
    (src.ratings, RATING, "rate"),
    (src.feature_engineering, FEATURE, "build"),
)


@dataclass(frozen=True, slots=True)
class Producer:
    """One thing that turns the canonical table into derived columns."""

    name: str
    kind: str
    compute: Compute
    outputs: tuple[str, ...]
    declared_reads: frozenset[str] | None
    """The canonical columns this producer says it reads, or ``None`` where it
    makes no such claim. Feature builders declare, per feature, in the
    registry; rating models do not, so the audit reports only what it measured
    for them."""


def _classes(package: ModuleType, attribute: str) -> list[type]:
    """Every class in ``package`` implementing ``attribute``, excluding Protocols.

    Walks the package rather than importing a list. ``_is_protocol`` filters out
    the contracts themselves — :class:`RatingModel` has a ``rate`` too, and
    instantiating it would fail in a way that looked like a broken producer.
    """
    found: list[type] = []
    for info in pkgutil.iter_modules(package.__path__, f"{package.__name__}."):
        module = importlib.import_module(info.name)
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if candidate.__module__ != info.name:
                continue
            if getattr(candidate, "_is_protocol", False):
                continue
            if callable(getattr(candidate, attribute, None)) and hasattr(candidate, "name"):
                found.append(candidate)
    return found


def producers() -> tuple[Producer, ...]:
    """Every rating model and feature builder in the codebase, in name order.

    Each is constructed with its defaults, which is how the pipelines construct
    them too — so what is probed here is what actually runs, not a test-tuned
    configuration that could hold while the shipped one does not.
    """
    found: list[Producer] = []
    for package, kind, attribute in _CONTRACTS:
        for cls in _classes(package, attribute):
            instance = cls()
            features = getattr(instance, "features", None)
            outputs = (
                tuple(feature.name for feature in features)
                if features is not None
                else tuple(instance.feature_columns)
            )
            found.append(
                Producer(
                    name=str(instance.name),
                    kind=kind,
                    compute=getattr(instance, attribute),
                    outputs=outputs,
                    declared_reads=(
                        frozenset().union(*(feature.reads for feature in features))
                        if features is not None
                        else None
                    ),
                )
            )
    return tuple(sorted(found, key=lambda producer: (producer.kind, producer.name)))


def check_defaults_are_complete(defaults: Sequence[str]) -> tuple[str, ...]:
    """Producers that exist but are in no pipeline's defaults.

    The mirror of the walk. A producer nobody runs is not a leak, but it is a
    column the model layer expects and will not get, and the failure shows up
    much later as a table of nulls.
    """
    return tuple(sorted({producer.name for producer in producers()} - set(defaults)))


def run_suite(
    matches: pd.DataFrame, found: Sequence[Producer] | None = None
) -> tuple[TemporalResult, ...]:
    """Both temporal probes, over every producer, in one call."""
    chosen = found if found is not None else producers()
    return tuple(
        result
        for producer in chosen
        for result in (
            prefix_invariance(producer.compute, matches, name=producer.name),
            outcome_independence(producer.compute, matches, name=producer.name),
        )
    )


@dataclass(frozen=True, slots=True)
class AuditRow:
    """One derived column, and where its value came from."""

    column: str
    producer: str
    kind: str
    observed: frozenset[str]
    """Canonical columns that, when rewritten, moved this one. A lower bound —
    see :func:`~src.validation.temporal.observed_reads`."""

    declared: frozenset[str] | None

    @property
    def post_match(self) -> frozenset[str]:
        """The observed inputs that are unknowable before kick-off.

        Non-empty for every form and rating column, and legitimately so: they
        read *earlier* matches' results. It is what the temporal probes exist
        to justify, which is why the audit prints both together.
        """
        return self.observed & POST_MATCH_COLUMNS

    @property
    def understated(self) -> frozenset[str]:
        """Inputs that were read but not declared. Empty, or the declaration is wrong."""
        return frozenset() if self.declared is None else self.observed - self.declared


def audit(
    matches: pd.DataFrame,
    found: Sequence[Producer] | None = None,
    *,
    columns: Sequence[str] = CANONICAL_COLUMNS,
) -> tuple[AuditRow, ...]:
    """Trace every derived column back to the canonical columns it reads."""
    chosen = found if found is not None else producers()
    rows: list[AuditRow] = []
    for producer in chosen:
        measured = observed_reads(producer.compute, matches, columns, key=KEY_COLUMN)
        for column in producer.outputs:
            rows.append(
                AuditRow(
                    column=column,
                    producer=producer.name,
                    kind=producer.kind,
                    observed=measured.get(column, frozenset()),
                    declared=_declared_for(column, producer),
                )
            )
    return tuple(rows)


def _declared_for(column: str, producer: Producer) -> frozenset[str] | None:
    """The registry's declaration for one column, where there is one."""
    if producer.declared_reads is None:
        return None
    return next(
        (feature.reads for feature in FEATURES if feature.name == column),
        producer.declared_reads,
    )


def benchmark_leaks(rows: Sequence[AuditRow]) -> tuple[str, ...]:
    """Derived columns that read the bookmaker's price.

    Odds are the benchmark this project measures itself against, and a feature
    that quietly reads them turns that comparison into a comparison with
    itself. The rule is written down on
    :data:`~src.ingestion.base.ODDS_COLUMNS`; this is the enforcement.
    """
    return tuple(sorted(row.column for row in rows if row.observed & BENCHMARK_COLUMNS))
