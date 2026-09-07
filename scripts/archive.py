#!/usr/bin/env python
"""Score what the service actually served, and say whether it has drifted.

    python scripts/archive.py                     # read the log, write the report
    python scripts/archive.py --dry-run           # print it, write nothing
    python scripts/archive.py --limit 5000        # read fewer rows from the log

Milestone 19. Every served prediction has been written to PostgreSQL since
Milestone 11 with the model, the version and the moment it was served; this
reads them back, joins the matches that have since been played, scores them
with the same function the backtest uses, and compares the result to what the
walk-forward folds said the same model does.

**The one report `make reproduce` cannot rebuild.** Every other table in
`data/reports/` is derived from the ingested data and comes back byte for byte.
This one is derived from things that happened — requests, at times, from a
service that was running — and if the log is lost the archive is gone. That is
why it has its own command rather than being another table `make card` writes.

**It prints how many forecasts a verdict would need.** A drift figure over
eleven matches is a reading of noise, and the honest form of "no drift
detected" is "this archive is too small to detect one, and here is how small".
Both numbers come out of the same command so neither can be quoted alone.

Exits non-zero when the log is not configured or the tables it needs are
missing. An *empty* log is not an error: a service nobody has called yet is the
ordinary state of a fresh deployment, and the report says so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation.archive import (  # noqa: E402
    Reference,
    horizon,
    reference,
    summarise,
)
from src.ingestion.base import TARGET_COLUMN  # noqa: E402
from src.models.ensemble import KEY_COLUMN, SHIPPED  # noqa: E402
from src.pipelines.report import write_archive  # noqa: E402
from src.pipelines.tables import (  # noqa: E402
    FORECASTS_FILENAME,
    read_forecasts,
    read_matches,
    resolve_tables,
)
from src.pipelines.train import ENSEMBLE_SUBDIR  # noqa: E402
from src.storage.predictions import open_prediction_log  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("archive")

DEFAULT_LIMIT = 1_000_000
"""How many rows are read back from the log.

A bound rather than an unbounded ``SELECT``: this runs against a table that
grows with every request forever, and a command that pulls all of it into a
frame is a command that stops working on the day the archive gets interesting.
A million rows is far beyond anything this deployment will hold and small
enough to fail loudly rather than swap.
"""

WORTH_SEEING: tuple[float, ...] = (0.10, 0.05, 0.02, 0.0163, 0.01)
"""The shifts in log loss the horizon table is priced for.

0.0163 is in the list because it is this project's own number — what separates
the shipped model from the closing line — so a reader can see how much archive
it would take to notice a change the size of the gap the whole project is
about.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matches", type=Path, default=None, help="canonical table to join")
    parser.add_argument("--forecasts", type=Path, default=None, help="fold forecasts to compare to")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="rows to read back from the log"
    )
    parser.add_argument("--model", default=SHIPPED, help="which forecaster the baseline is for")
    parser.add_argument("--dry-run", action="store_true", help="print, but write nothing")
    parser.add_argument("--log-level", default=None, help="override the configured level")
    return parser.parse_args(argv)


def render(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "  (nothing to report)"
    return frame.to_string(index=False, float_format=lambda value: f"{value:.4f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(level=args.log_level or settings.logging.level, fmt=settings.logging.format)

    dsn = settings.api.prediction_log_dsn
    if not dsn:
        logger.error(
            "no prediction log: set PREDICTION_LOG_DSN to the database the service writes to"
        )
        return 1

    tables = resolve_tables(settings.paths, matches=args.matches)
    # Two columns of three hundred thousand rows. The join needs the id and
    # the outcome and nothing else, and the default projection is a dozen
    # columns shaped for a results page.
    matches = read_matches(tables.matches, columns=(KEY_COLUMN, TARGET_COLUMN))
    if matches is None:
        logger.error("no match table at %s; run `make data` first", tables.matches)
        return 1

    forecasts_path = (
        args.forecasts or settings.paths.reports_dir / ENSEMBLE_SUBDIR / FORECASTS_FILENAME
    )
    forecasts = read_forecasts(forecasts_path)
    if forecasts is None:
        logger.error(
            "no fold forecasts at %s; run `make card` — there is nothing to compare against",
            forecasts_path,
        )
        return 1

    log = open_prediction_log(dsn)
    try:
        archive = log.recent(limit=args.limit)
    finally:
        log.close()

    against: Reference = reference(forecasts, name=args.model)
    report = summarise(archive, matches, spread=against.spread, baseline=against.log_loss)

    print(f"\n{len(archive):,} row(s) read from the prediction log\n")
    print(
        f"Baseline: {args.model} scored {against.log_loss:.4f} over {against.n:,} "
        f"walk-forward forecasts, with a per-match spread of {against.spread:.4f}."
    )
    print("\nWhat was served, by model version:")
    print(render(report))

    # Printed whether or not anything was scorable, and deliberately: this is
    # the half that makes an empty report readable instead of embarrassing.
    print("\nHow much archive a verdict would need:")
    print(render(horizon(against.spread, WORTH_SEEING)))

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return 0

    written = write_archive(report, settings.paths.reports_dir / ENSEMBLE_SUBDIR)
    print(f"\n{len(report)} version(s) written to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
