#!/usr/bin/env python
"""Validate the canonical match table and regenerate its dataset card.

    python scripts/validate_data.py             # validate, write docs/DATASET_CARD.md
    python scripts/validate_data.py --strict    # warnings fail too
    python scripts/validate_data.py --no-card   # just the checks
    python scripts/validate_data.py --json out.json

Exits non-zero when a blocking check fails, so it can be a step in a pipeline
rather than something a person reads and forgets. Warnings do not fail the run
by default: one Argentinian match filed under the wrong season is worth
printing and is not worth refusing three hundred thousand rows over. `--strict`
is for the caller who disagrees.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.registry import load_registry  # noqa: E402
from src.pipelines.ingest import MATCHES_FILENAME  # noqa: E402
from src.storage.duckdb_store import DuckDBStore  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402
from src.validation.card import build_card, write_card  # noqa: E402
from src.validation.matches import match_checks  # noqa: E402
from src.validation.report import Outcome, run_checks  # noqa: E402

logger = get_logger("validate_data")

DEFAULT_CARD = PROJECT_ROOT / "docs" / "DATASET_CARD.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matches", type=Path, default=None, help="Parquet file to validate")
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD, help="dataset card destination")
    parser.add_argument("--no-card", action="store_true", help="skip writing the dataset card")
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    parser.add_argument("--strict", action="store_true", help="fail on warnings as well as errors")
    parser.add_argument("--log-level", default=None, help="override the configured level")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(level=args.log_level or settings.logging.level, fmt=settings.logging.format)

    matches_path = args.matches or settings.paths.processed_dir / MATCHES_FILENAME
    if not matches_path.is_file():
        logger.error("no match table at %s; run scripts/fetch_data.py first", matches_path)
        return 1

    with DuckDBStore.open_matches(matches_path) as store:
        report = run_checks(store.read_matches(), match_checks(load_registry()))
        print(report.to_markdown())
        print(f"\n{report.summary()}")

        if not args.no_card:
            destination = write_card(args.card, build_card(store, report, source=matches_path))
            print(f"dataset card written to {destination}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"report written to {args.json}")

    failed = report.of(Outcome.FAILED)
    if args.strict and failed:
        print(f"\nFAILED (strict): {len(failed)} check(s) failed")
        return 1
    if not report.ok:
        print(f"\nFAILED: {len(report.blocking)} blocking check(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
