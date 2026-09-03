"""Build the feature table from the canonical one.

Orchestration only. The shared half — probes, report, manifest — is in
:mod:`src.pipelines.derived`, so this module is a list of builders and the
checks to run over what they produce.

Features are cheap where ratings are expensive: a full build is a few seconds
against ten minutes, because every feature here is a window over a sorted array
rather than a maximum-likelihood fit. That difference is why they are separate
artefacts — the feature set will churn through the modelling milestones and the
ratings will not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.feature_engineering.head_to_head import HeadToHeadFeatures
from src.feature_engineering.registry import (
    FEATURE_SCHEMA,
    FEATURES,
    KEY_COLUMN,
    FeatureBuilder,
)
from src.feature_engineering.team_history import TeamHistoryFeatures
from src.pipelines.derived import (
    DerivedReport,
    choose_verification_sample,
    persist,
    report_validation,
    run_probes,
)
from src.ratings.base import require_chronological
from src.utils.logging import get_logger
from src.validation.features import feature_checks
from src.validation.report import run_checks

logger = get_logger(__name__)

FEATURES_FILENAME = "features.parquet"


def default_builders() -> tuple[FeatureBuilder, ...]:
    return (TeamHistoryFeatures(), HeadToHeadFeatures())


@dataclass
class FeaturesReport(DerivedReport):
    """What one feature build did."""

    features: int = 0
    """How many columns were produced. Compared against the registry by the
    check suite, so a builder that silently stopped filling one is visible."""

    def parts(self) -> list[str]:
        return [*super().parts(), f"{self.features} features"]


def build_features(
    matches: pd.DataFrame, builders: tuple[FeatureBuilder, ...] | None = None
) -> pd.DataFrame:
    """Run every builder and assemble one table keyed by ``match_id``.

    Columns no builder produced are left null rather than dropped, on the same
    reasoning as the ratings table: a partial build should widen the same
    schema, not a different one.
    """
    require_chronological(matches)
    chosen = builders if builders is not None else default_builders()

    assembled = pd.DataFrame({KEY_COLUMN: matches[KEY_COLUMN].to_numpy()})
    for builder in chosen:
        built = builder.build(matches).set_index(KEY_COLUMN)
        for feature in builder.features:
            assembled[feature.name] = built[feature.name].reindex(assembled[KEY_COLUMN]).to_numpy()
        logger.info("%s: built %d features", builder.name, len(builder.features))

    return assembled.reindex(columns=list(FEATURE_SCHEMA)).astype(FEATURE_SCHEMA)


def run_features(
    matches: pd.DataFrame,
    features_dir: Path,
    *,
    builders: tuple[FeatureBuilder, ...] | None = None,
    verify: bool = True,
) -> FeaturesReport:
    """Build, verify, check and persist the feature table.

    Args:
        matches: The canonical table, sorted by date.
        features_dir: Destination, beside the ratings.
        builders: Defaults to every registered builder.
        verify: Run the causality probes. On by default. Features are where
            leakage is most likely and hardest to see by reading, so the only
            reason to skip them is a run whose output is going to be discarded.
    """
    chosen = builders if builders is not None else default_builders()
    report = FeaturesReport(
        matches=len(matches),
        producers=tuple(builder.name for builder in chosen),
        features=sum(len(builder.features) for builder in chosen),
    )

    features = build_features(matches, chosen)
    for builder in chosen:
        # Coverage is reported against the first feature a builder declares,
        # which is by convention its history count — the column that is filled
        # whenever the builder ran at all.
        first = builder.features[0].name
        report.coverage[builder.name] = (
            float(features[first].notna().mean()) if len(features) else 0.0
        )

    if verify:
        competition_id = choose_verification_sample(matches)
        if competition_id is not None:
            report.verified_on = competition_id
            report.temporal = run_probes(
                matches,
                [(builder.name, builder.build) for builder in chosen],
                competition_id=competition_id,
            )

    validation = run_checks(
        features,
        feature_checks(
            matches[KEY_COLUMN],
            built=[feature.name for builder in chosen for feature in builder.features],
        ),
    )
    report.validation = validation
    report_validation(validation)

    report.output = persist(
        features,
        features_dir / FEATURES_FILENAME,
        extra={
            "kind": "features",
            "builders": list(report.producers),
            "features": [feature.name for feature in FEATURES],
            "matches": report.matches,
            "coverage": report.coverage,
            "causality_verified_on": report.verified_on,
            "causal": report.causal if report.verified_on else None,
        },
    )
    logger.info("wrote %s — %s", report.output.name, report.summary())
    return report
