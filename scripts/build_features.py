#!/usr/bin/env python
"""Build the feature table from the canonical match table.

    python scripts/build_features.py                       # every builder
    python scripts/build_features.py --builder team_history
    python scripts/build_features.py --competition ENG_1   # one competition
    python scripts/build_features.py --list                # show the registry

A full run takes a few seconds: every feature is a window over a sorted array,
not a fit. That is why features and ratings are separate commands — the feature
set changes with modelling work and a ten-minute rebuild each time would
discourage that.

Exits non-zero if a causality probe fails. A feature that can see its own match
does not produce a worse model; it produces a better-looking one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_engineering.head_to_head import HeadToHeadFeatures  # noqa: E402
from src.feature_engineering.registry import FEATURES, FeatureBuilder, groups  # noqa: E402
from src.feature_engineering.team_history import TeamHistoryFeatures  # noqa: E402
from src.pipelines.features import run_features  # noqa: E402
from src.pipelines.ingest import MATCHES_FILENAME  # noqa: E402
from src.storage.duckdb_store import DuckDBStore  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402
from src.validation.report import Outcome  # noqa: E402

logger = get_logger("build_features")

BUILDERS: dict[str, type[TeamHistoryFeatures] | type[HeadToHeadFeatures]] = {
    "team_history": TeamHistoryFeatures,
    "head_to_head": HeadToHeadFeatures,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matches", type=Path, default=None, help="canonical table to read")
    parser.add_argument("--output", type=Path, default=None, help="directory for features.parquet")
    parser.add_argument(
        "--builder",
        action="append",
        default=[],
        choices=sorted(BUILDERS),
        help="feature builder. Repeatable. Defaults to all.",
    )
    parser.add_argument(
        "--competition",
        action="append",
        default=[],
        metavar="ID",
        help="restrict to this competition. Repeatable.",
    )
    parser.add_argument("--list", action="store_true", help="list the feature registry and exit")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the causality probes. They are seconds here, and they are the "
        "difference between claiming the property and having it.",
    )
    parser.add_argument("--log-level", default=None, help="override the configured level")
    return parser.parse_args(argv)


def select_builders(names: list[str]) -> tuple[FeatureBuilder, ...]:
    chosen = names or sorted(BUILDERS)
    return tuple(BUILDERS[name]() for name in chosen)


def print_registry() -> None:
    print(f"{len(FEATURES)} features in {len(groups())} groups\n")
    for group, members in groups().items():
        print(f"{group}:")
        for feature in members:
            leak = "post-match" if feature.can_leak else "pre-match "
            print(f"  {feature.name:24s} {feature.dtype:8s} {leak}  {feature.description}")
        print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(level=args.log_level or settings.logging.level, fmt=settings.logging.format)

    if args.list:
        print_registry()
        return 0

    matches_path = args.matches or settings.paths.processed_dir / MATCHES_FILENAME
    if not matches_path.is_file():
        logger.error("no match table at %s; run scripts/fetch_data.py first", matches_path)
        return 1

    with DuckDBStore.open_matches(matches_path) as store:
        matches = store.read_matches(competitions=args.competition or None)
    if matches.empty:
        logger.error(
            "no matches for %s", ", ".join(args.competition) if args.competition else matches_path
        )
        return 1

    report = run_features(
        matches,
        args.output or settings.paths.features_dir,
        builders=select_builders(args.builder),
        verify=not args.no_verify,
        sources=[matches_path],
    )

    print(f"\n{report.summary()}")
    print(f"written to {report.output}")
    for result in report.temporal:
        print(f"  {result.summary()}")
    if report.validation is not None:
        for failure in report.validation.of(Outcome.FAILED):
            print(f"  validation: {failure.name} — {failure.message}")

    if not report.causal:
        print("\nFAILED: a causality probe did not hold")
        return 1
    if report.validation is not None and not report.validation.ok:
        print("\nFAILED: a blocking check did not hold")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
