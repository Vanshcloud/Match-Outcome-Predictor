"""Build the ratings table from the canonical one.

Orchestration only, matching :mod:`src.pipelines.ingest`: this module decides
*what* runs and in what order, and knows nothing about Elo or Poisson
likelihoods. The parts it shares with :mod:`src.pipelines.features` — the
probes, the report, the manifest — live in :mod:`src.pipelines.derived`.

The causality probes run on every build, not just in the test suite. Ratings
are the first quantity here with memory, and a leak in one is invisible in the
output: it simply makes every downstream model look better than it is. A couple
of minutes per build is the difference between claiming the property and having
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.pipelines.derived import (
    KEY_COLUMN,
    DerivedReport,
    choose_verification_sample,
    persist,
    report_validation,
    run_probes,
)
from src.ratings.base import RATINGS_SCHEMA, RatingModel, require_chronological
from src.ratings.dixon_coles import DixonColesRatings
from src.ratings.elo import EloRatings, mean_squared_error
from src.utils.logging import get_logger
from src.validation.ratings import ratings_checks
from src.validation.report import run_checks

logger = get_logger(__name__)

RATINGS_FILENAME = "ratings.parquet"


def default_models() -> tuple[RatingModel, ...]:
    return (EloRatings(), DixonColesRatings())


@dataclass
class RatingsReport(DerivedReport):
    """What one ratings build did."""

    elo_mse: float | None = None

    def parts(self) -> list[str]:
        parts = super().parts()
        if self.elo_mse is not None:
            parts.append(f"elo mse {self.elo_mse:.5f}")
        return parts


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
    report = RatingsReport(matches=len(matches), producers=tuple(m.name for m in chosen))

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
            report.temporal = run_probes(
                matches,
                [(model.name, model.rate) for model in chosen],
                competition_id=competition_id,
            )

    validation = run_checks(
        ratings,
        ratings_checks(
            matches[KEY_COLUMN],
            expect_dixon_coles=any(isinstance(m, DixonColesRatings) for m in chosen),
        ),
    )
    report.validation = validation
    report_validation(validation)

    report.output = persist(
        ratings,
        features_dir / RATINGS_FILENAME,
        extra={
            "kind": "ratings",
            "models": list(report.producers),
            "matches": report.matches,
            "coverage": report.coverage,
            "elo_mse": report.elo_mse,
            "causality_verified_on": report.verified_on,
            "causal": report.causal if report.verified_on else None,
        },
    )
    logger.info("wrote %s — %s", report.output.name, report.summary())
    return report
