"""Build the ratings table from the canonical one.

Orchestration only, matching :mod:`src.pipelines.ingest`: this module decides
*what* runs and in what order, and knows nothing about Elo or Poisson
likelihoods.

Three things happen on every run, and the middle one is the reason this is a
pipeline rather than a script:

1. Each model walks the table and returns one row per match.
2. The causality probes run against a sample competition. Ratings are the first
   quantity here with memory, and a leak in one is invisible in the output — it
   simply makes every downstream model look better than it is. Checking on
   every build costs a couple of minutes and is the difference between claiming
   the property and having it.
3. The arithmetic checks run over the assembled table, and the result is
   recorded on the report rather than raised, on the same reasoning as
   ingestion: a table you can inspect beats one the pipeline refused to save.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.ingestion.manifest import write_manifest
from src.ratings.base import KEY_COLUMN, RATINGS_SCHEMA, RatingModel, require_chronological
from src.ratings.dixon_coles import DixonColesRatings
from src.ratings.elo import EloRatings, mean_squared_error
from src.utils.logging import get_logger
from src.validation.ratings import ratings_checks
from src.validation.report import Outcome, ValidationReport, run_checks
from src.validation.temporal import TemporalResult, outcome_independence, prefix_invariance

logger = get_logger(__name__)

RATINGS_FILENAME = "ratings.parquet"

DEFAULT_VERIFY_SAMPLE = 4_000
"""Target size of the competition the causality probes run against.

Not the smallest competition available: a sample too thin for Dixon-Coles to
fit at all would pass both probes trivially, which is the one way a leakage
check can be worse than useless. Not the largest either, because the probes
re-run the model half a dozen times.
"""


def default_models() -> tuple[RatingModel, ...]:
    return (EloRatings(), DixonColesRatings())


@dataclass
class RatingsReport:
    """What one ratings build did."""

    matches: int = 0
    models: tuple[str, ...] = ()
    coverage: dict[str, float] = field(default_factory=dict)
    """Per model, the share of matches it could rate. Dixon-Coles cannot price
    a competition's opening seasons; Elo always answers."""

    elo_mse: float | None = None
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

    def summary(self) -> str:
        parts = [f"{self.matches:,} matches rated by {', '.join(self.models)}"]
        parts += [f"{name} {share:.1%}" for name, share in sorted(self.coverage.items())]
        if self.elo_mse is not None:
            parts.append(f"elo mse {self.elo_mse:.5f}")
        if self.verified_on is not None:
            parts.append(f"causality {'ok' if self.causal else 'FAILED'} on {self.verified_on}")
        if self.validation is not None:
            parts.append(self.validation.summary())
        return "; ".join(parts)


def build_ratings(
    matches: pd.DataFrame, models: tuple[RatingModel, ...] | None = None
) -> pd.DataFrame:
    """Run every model and assemble one table keyed by ``match_id``.

    Columns no model produced are left null rather than dropped. A partial
    build should widen the same schema, not a different one — otherwise every
    consumer has to discover which columns exist before it can read any.
    """
    require_chronological(matches)
    chosen = models if models is not None else default_models()

    assembled = pd.DataFrame({KEY_COLUMN: matches[KEY_COLUMN].to_numpy()})
    for model in chosen:
        rated = model.rate(matches).set_index(KEY_COLUMN)
        for column in model.feature_columns:
            assembled[column] = rated[column].reindex(assembled[KEY_COLUMN]).to_numpy()
        logger.info("%s: rated %d matches", model.name, len(rated))

    return assembled.reindex(columns=list(RATINGS_SCHEMA)).astype(RATINGS_SCHEMA)


def choose_verification_sample(
    matches: pd.DataFrame, target: int = DEFAULT_VERIFY_SAMPLE
) -> str | None:
    """Pick the competition whose size is closest to ``target``."""
    if matches.empty:
        return None
    sizes = matches.groupby("competition_id", observed=True).size()
    return str((sizes - target).abs().idxmin())


def verify_causality(
    matches: pd.DataFrame,
    models: tuple[RatingModel, ...],
    *,
    competition_id: str,
) -> tuple[TemporalResult, ...]:
    """Run both probes against one competition's history, per model."""
    sample = matches[matches["competition_id"] == competition_id]
    results: list[TemporalResult] = []
    for model in models:
        results.append(prefix_invariance(model.rate, sample, name=model.name))
        results.append(outcome_independence(model.rate, sample, name=model.name))
    for result in results:
        (logger.info if result.ok else logger.error)("%s", result.summary())
    return tuple(results)


def run_ratings(
    matches: pd.DataFrame,
    features_dir: Path,
    *,
    models: tuple[RatingModel, ...] | None = None,
    verify: bool = True,
) -> RatingsReport:
    """Build, verify, check and persist the ratings table.

    Args:
        matches: The canonical table, sorted by date.
        features_dir: Destination. Ratings live with the features because that
            is what they are — derived, and rebuildable from ``processed`` at
            any time.
        models: Defaults to Elo and Dixon-Coles.
        verify: Run the causality probes. On by default; the only reason to
            turn it off is a run whose output is going to be thrown away.
    """
    chosen = models if models is not None else default_models()
    report = RatingsReport(matches=len(matches), models=tuple(m.name for m in chosen))

    ratings = build_ratings(matches, chosen)
    for model in chosen:
        first = next(iter(model.feature_columns))
        report.coverage[model.name] = float(ratings[first].notna().mean()) if len(ratings) else 0.0

    elo = next((m for m in chosen if isinstance(m, EloRatings)), None)
    if elo is not None and not matches.empty:
        report.elo_mse = mean_squared_error(matches, elo)

    if verify:
        competition_id = choose_verification_sample(matches)
        if competition_id is not None:
            report.verified_on = competition_id
            report.temporal = verify_causality(matches, chosen, competition_id=competition_id)

    report.validation = run_checks(
        ratings,
        ratings_checks(
            matches[KEY_COLUMN],
            expect_dixon_coles=any(isinstance(m, DixonColesRatings) for m in chosen),
        ),
    )
    for failure in report.validation.of(Outcome.FAILED):
        logger.warning("validation: %s — %s", failure.name, failure.message)

    features_dir.mkdir(parents=True, exist_ok=True)
    output = features_dir / RATINGS_FILENAME
    ratings.to_parquet(output, index=False)
    report.output = output

    write_manifest(
        output.with_suffix(".manifest.json"),
        features_dir,
        [output],
        extra={
            "kind": "ratings",
            "models": list(report.models),
            "matches": report.matches,
            "coverage": report.coverage,
            "elo_mse": report.elo_mse,
            "causality_verified_on": report.verified_on,
            "causal": report.causal if report.verified_on else None,
        },
    )
    logger.info("wrote %s — %s", output.name, report.summary())
    return report
