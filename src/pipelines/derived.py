"""The shape every derived table is built with.

Ratings and features are different things — one has memory, the other is a
window — but the *pipeline* around them is identical: run the producers, prove
they cannot see the future, check the arithmetic, write the table with a
manifest, and report. Written twice, the two would drift, and the half that
drifts is always the probe nobody looked at again.

So the common half lives here and each pipeline supplies what is actually its
own: which producers to run, and which checks to run over the result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.ingestion.manifest import write_manifest
from src.utils.logging import get_logger
from src.validation.report import Outcome, ValidationReport
from src.validation.temporal import TemporalResult, outcome_independence, prefix_invariance

logger = get_logger(__name__)

KEY_COLUMN = "match_id"

DEFAULT_VERIFY_SAMPLE = 4_000
"""Target size of the competition the causality probes run against.

Not the smallest available: a sample too thin for a producer to say anything
would pass every probe trivially, which is the one way a leakage check can be
worse than useless. Not the largest either, because the probes re-run the
producer half a dozen times.
"""

Produce = Callable[[pd.DataFrame], pd.DataFrame]
"""Turns a slice of the canonical table into one row per match."""


@dataclass
class DerivedReport:
    """What one build of a derived table did."""

    matches: int = 0
    producers: tuple[str, ...] = ()
    coverage: dict[str, float] = field(default_factory=dict)
    """Per producer, the share of matches it could give a value for. A
    Dixon-Coles rating has nothing to say about a competition's first season;
    a form feature has nothing to say about a club's first match."""

    temporal: tuple[TemporalResult, ...] = ()
    verified_on: str | None = None
    """Which competition the causality probes ran against, or ``None`` if they
    were skipped. Named rather than implied: "verified" and "verified on
    ENG_1" are different claims."""

    validation: ValidationReport | None = None
    output: Path | None = None

    @property
    def causal(self) -> bool:
        """True when every probe that ran passed. Vacuously true if none did."""
        return all(result.ok for result in self.temporal)

    def parts(self) -> list[str]:
        """Summary fragments, so a subclass can insert its own."""
        parts = [f"{self.matches:,} matches by {', '.join(self.producers)}"]
        parts += [f"{name} {share:.1%}" for name, share in sorted(self.coverage.items())]
        return parts

    def summary(self) -> str:
        parts = self.parts()
        if self.verified_on is not None:
            parts.append(f"causality {'ok' if self.causal else 'FAILED'} on {self.verified_on}")
        if self.validation is not None:
            parts.append(self.validation.summary())
        return "; ".join(parts)


def choose_verification_sample(
    matches: pd.DataFrame, target: int = DEFAULT_VERIFY_SAMPLE
) -> str | None:
    """Pick the competition whose size is closest to ``target``."""
    if matches.empty:
        return None
    sizes = matches.groupby("competition_id", observed=True).size()
    return str((sizes - target).abs().idxmin())


def run_probes(
    matches: pd.DataFrame,
    producers: Sequence[tuple[str, Produce]],
    *,
    competition_id: str,
) -> tuple[TemporalResult, ...]:
    """Run both temporal probes against one competition's history, per producer."""
    sample = matches[matches["competition_id"] == competition_id]
    results: list[TemporalResult] = []
    for name, produce in producers:
        results.append(prefix_invariance(produce, sample, name=name))
        results.append(outcome_independence(produce, sample, name=name))
    for result in results:
        (logger.info if result.ok else logger.error)("%s", result.summary())
    return tuple(results)


def report_validation(validation: ValidationReport) -> None:
    """Log every failed check.

    Warnings included: a validation failure nobody sees is a validation suite
    nobody has, and severity decides whether it stops a caller, not whether it
    is worth printing.
    """
    for failure in validation.of(Outcome.FAILED):
        logger.warning("validation: %s — %s", failure.name, failure.message)


def persist(
    table: pd.DataFrame,
    destination: Path,
    *,
    extra: dict[str, object],
) -> Path:
    """Write ``table`` as Parquet with a manifest beside it.

    The manifest is named from the table, so two derived tables can share a
    directory without one overwriting the other's provenance.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(destination, index=False)
    write_manifest(
        destination.with_suffix(".manifest.json"),
        destination.parent,
        [destination],
        extra=extra,
    )
    return destination
