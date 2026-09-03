#!/usr/bin/env python
"""Score every baseline over walk-forward folds.

    python scripts/backtest.py                          # 5 folds of a year each
    python scripts/backtest.py --folds 10 --horizon-days 182
    python scripts/backtest.py --competition ENG_1
    python scripts/backtest.py --markdown               # the tables in docs/

Each fold trains on everything strictly earlier than the year it is scored on,
and every fold is checked by the split-boundary probe before it is used. The
baselines are the class prior counted on the training half, Dixon-Coles from
the ratings table, the bookmaker's closing line with the overround removed, and
the home-always forecast that exists to show what a proper scoring rule does to
certainty.

Two tables come out, and they are not interchangeable. The first scores each
forecaster over the matches *it* could price; the second scores all of them
over the matches *every* one could price, which is the only table whose rows
may be read against each other.

Exits non-zero if no fold can be built, or if a fold fails the boundary probe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.models.splits import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_HORIZON_DAYS,
    SplitError,
)
from src.pipelines.backtest import (  # noqa: E402
    COMMON,
    PRICED,
    BacktestReport,
    per_competition_table,
    pooled_table,
    run_backtest,
)
from src.pipelines.ingest import MATCHES_FILENAME  # noqa: E402
from src.pipelines.ratings import RATINGS_FILENAME  # noqa: E402
from src.storage.duckdb_store import DuckDBStore  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("backtest")

KEY_COLUMN = "match_id"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matches", type=Path, default=None, help="canonical table to read")
    parser.add_argument(
        "--ratings",
        type=Path,
        default=None,
        help="ratings table. Without it the Dixon-Coles baseline is dropped, "
        "rather than reported as a forecaster that priced nothing.",
    )
    parser.add_argument("--output", type=Path, default=None, help="directory for backtest.parquet")
    parser.add_argument(
        "--competition",
        action="append",
        default=[],
        metavar="ID",
        help="restrict to this competition. Repeatable.",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="walk-forward steps")
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=DEFAULT_HORIZON_DAYS,
        help="days scored per fold. A year by default: the competitions here "
        "start in three different months, so a season is not a shared boundary.",
    )
    parser.add_argument("--markdown", action="store_true", help="print the tables as markdown")
    parser.add_argument("--log-level", default=None, help="override the configured level")
    return parser.parse_args(argv)


def _cell(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _markdown(table: pd.DataFrame) -> str:
    """A markdown table, without a formatting dependency for it.

    `DataFrame.to_markdown` needs tabulate, which would be a new pin earning
    its keep in one `--markdown` flag used when a doc is regenerated.
    """
    header = list(table.columns)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += [
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join(lines)


def render(table: pd.DataFrame, *, markdown: bool) -> str:
    if table.empty:
        return "  (nothing priced by every forecaster)"
    if markdown:
        return _markdown(table)
    return table.to_string(index=False, float_format=lambda value: f"{value:.4f}")


def report_tables(report: BacktestReport, *, markdown: bool) -> None:
    print(f"\n{report.folds} fold(s), {report.matches:,} matches scored\n")

    priced = pooled_table(report.scores, PRICED) if report.scores is not None else pd.DataFrame()
    if not priced.empty:
        priced.insert(1, "coverage", priced["forecaster"].map(report.coverage))
    print("Each forecaster over the matches it could price:")
    print(render(priced, markdown=markdown))

    common = pooled_table(report.scores, COMMON) if report.scores is not None else pd.DataFrame()
    print("\nEvery forecaster over the matches all of them could price:")
    print(render(common, markdown=markdown))

    if report.scores is not None:
        print("\nLog loss by competition, on that common subset:")
        print(render(per_competition_table(report.scores), markdown=markdown))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(level=args.log_level or settings.logging.level, fmt=settings.logging.format)

    matches_path = args.matches or settings.paths.processed_dir / MATCHES_FILENAME
    if not matches_path.is_file():
        logger.error("no match table at %s; run scripts/fetch_data.py first", matches_path)
        return 1

    ratings_path = args.ratings or settings.paths.features_dir / RATINGS_FILENAME
    if not ratings_path.is_file():
        logger.warning("no ratings at %s; the Dixon-Coles baseline will be skipped", ratings_path)
        ratings_path = None

    with DuckDBStore.open_matches(matches_path, ratings=ratings_path) as store:
        matches = store.read_matches(competitions=args.competition or None)
        ratings = store.read_ratings() if ratings_path is not None else None

    if matches.empty:
        logger.error(
            "no matches for %s", ", ".join(args.competition) if args.competition else matches_path
        )
        return 1
    if ratings is not None:
        # A left join, so a match the ratings have nothing to say about keeps
        # its row and loses only its Dixon-Coles columns — which is what the
        # coverage figure is then measuring.
        matches = matches.merge(ratings, on=KEY_COLUMN, how="left")

    try:
        report = run_backtest(
            matches,
            args.output or settings.paths.reports_dir,
            folds=args.folds,
            horizon_days=args.horizon_days,
        )
    except SplitError as error:
        logger.error("%s", error)
        return 1

    report_tables(report, markdown=args.markdown)
    print(f"\nwritten to {report.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
