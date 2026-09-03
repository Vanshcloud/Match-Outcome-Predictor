#!/usr/bin/env python
"""Probe every producer and trace every derived column back to its inputs.

    python scripts/audit_columns.py                          # sample competition
    python scripts/audit_columns.py --competition ENG_1 --competition ESP_1
    python scripts/audit_columns.py --markdown               # the table in docs/

The pipelines already probe the producers they run. This runs the same probes
over every producer that *exists*, found by walking the packages rather than by
reading a list, so a builder wired in without being added to a probe call is
still caught. It is the command behind docs/LEAKAGE.md.

The trace is measured, not declared: each canonical column is rewritten in turn
and the columns that move are the ones that read it. That costs a few hundred
recomputations, so it runs against a sample rather than the full table — a few
minutes on a league-sized one, most of it Dixon-Coles refitting.

Two things about the sample decide what the audit can see. It takes the *most
recent* matches, because the older ones have no shot data and a column that is
entirely null cannot be perturbed — a run over 1993 would report that no
feature reads shots. And a column that does not vary cannot be perturbed
either, so a partition like `competition_id` only shows up in a sample holding
more than one; pass two competitions to see it.

Exits non-zero if a probe fails, if a feature reads more than it declares, or
if anything at all reads the bookmaker's price.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.pipelines.derived import choose_verification_sample  # noqa: E402
from src.pipelines.ingest import MATCHES_FILENAME  # noqa: E402
from src.storage.duckdb_store import DuckDBStore  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402
from src.validation.leakage import (  # noqa: E402
    KEY_COLUMN,
    AuditRow,
    audit,
    benchmark_leaks,
    producers,
    run_suite,
)

logger = get_logger("audit_columns")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matches", type=Path, default=None, help="canonical table to read")
    parser.add_argument(
        "--competition",
        action="append",
        default=[],
        metavar="ID",
        help="competition to probe against. Repeatable. Defaults to the one "
        "closest to 4,000 matches.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1500,
        help="most recent N matches per competition. The default is a few minutes.",
    )
    parser.add_argument(
        "--markdown", action="store_true", help="print the audit as a markdown table"
    )
    parser.add_argument("--log-level", default=None, help="override the configured level")
    return parser.parse_args(argv)


def as_markdown(rows: Sequence[AuditRow]) -> str:
    """The audit as a table, for pasting into docs/LEAKAGE.md."""
    lines = ["| Column | Producer | Reads (measured) | Post-match inputs |", "|---|---|---|---|"]
    for row in rows:
        reads = ", ".join(f"`{name}`" for name in sorted(row.observed)) or "—"
        post = ", ".join(f"`{name}`" for name in sorted(row.post_match)) or "—"
        lines.append(f"| `{row.column}` | {row.producer} | {reads} | {post} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(level=args.log_level or settings.logging.level, fmt=settings.logging.format)

    matches_path = args.matches or settings.paths.processed_dir / MATCHES_FILENAME
    if not matches_path.is_file():
        logger.error("no match table at %s; run scripts/fetch_data.py first", matches_path)
        return 1

    with DuckDBStore.open_matches(matches_path) as store:
        matches = store.read_matches(competitions=args.competition or None)

    # `choose_verification_sample` answers None on an empty frame, which is the
    # clean-checkout case and the unknown-competition case at once.
    asked = args.competition or [choose_verification_sample(matches)]
    chosen = [one for one in asked if one is not None]
    if not chosen:
        logger.error("no matches to audit in %s", matches_path)
        return 1

    # Sorted after concatenating, because every producer requires it and the
    # per-competition tails are interleaved in time.
    sample = (
        pd.concat(
            [matches[matches["competition_id"] == one].tail(args.limit) for one in chosen],
            ignore_index=True,
        )
        .sort_values(["date", "competition_id", KEY_COLUMN], kind="stable")
        .reset_index(drop=True)
    )

    found = producers()
    named = ", ".join(chosen)
    print(f"{len(found)} producers, {len(sample):,} matches from {named}\n")

    results = run_suite(sample, found)
    for result in results:
        print(f"  {result.summary()}")

    rows = audit(sample, found)
    understated = [row for row in rows if row.understated]
    odds = benchmark_leaks(rows)

    print()
    print(as_markdown(rows) if args.markdown else f"{len(rows)} derived columns traced")
    print()

    for row in understated:
        print(f"UNDECLARED: {row.column} reads {sorted(row.understated)}")
    for column in odds:
        print(f"BENCHMARK LEAK: {column} reads the bookmaker's price")

    if not all(result.ok for result in results):
        print("\nFAILED: a causality probe did not hold")
        return 1
    if understated or odds:
        print("\nFAILED: the audit contradicts the registry")
        return 1
    print(f"clean: every producer probed on {named}, every column traced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
