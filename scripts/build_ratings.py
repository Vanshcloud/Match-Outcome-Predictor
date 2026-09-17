#!/usr/bin/env python
"""Build the ratings table from the canonical match table.

    python scripts/build_ratings.py                        # every model, every match
    python scripts/build_ratings.py --model elo            # one model, repeatable
    python scripts/build_ratings.py --competition ENG_1    # one competition, repeatable
    python scripts/build_ratings.py --fit-until 2005-07-01 # re-derive the Elo constants

A full run takes roughly twenty minutes, almost all of it Dixon-Coles: it
refits per competition every thirty days across three decades, and each refit
is a maximum-likelihood problem in two strengths per team. Elo alone takes
about a second.

Exits non-zero if a causality probe fails. A rating that can see its own match
does not produce a worse model — it produces a better-looking one, which is
worse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.ingest import MATCHES_FILENAME  # noqa: E402
from src.pipelines.ratings import run_ratings  # noqa: E402
from src.ratings import elo as elo_module  # noqa: E402
from src.ratings.base import RatingModel  # noqa: E402
from src.ratings.dixon_coles import DixonColesRatings  # noqa: E402
from src.ratings.elo import EloRatings  # noqa: E402
from src.storage.duckdb_store import DuckDBStore  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402
from src.validation.report import Outcome  # noqa: E402

logger = get_logger("build_ratings")

MODELS: dict[str, type[EloRatings] | type[DixonColesRatings]] = {
    "elo": EloRatings,
    "dixon_coles": DixonColesRatings,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matches", type=Path, default=None, help="canonical table to read")
    parser.add_argument("--output", type=Path, default=None, help="directory for ratings.parquet")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        choices=sorted(MODELS),
        help="rating model. Repeatable. Defaults to all.",
    )
    parser.add_argument(
        "--competition",
        action="append",
        default=[],
        metavar="ID",
        help="restrict to this competition. Repeatable.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the causality probes. They cost a couple of minutes and are "
        "the difference between claiming the property and having it.",
    )
    parser.add_argument(
        "--fit-until",
        metavar="DATE",
        default=None,
        help="re-derive the Elo constants from matches before DATE, print them, and exit. "
        "A maintenance tool: fitting on rows the ratings later predict is a leak.",
    )
    parser.add_argument("--log-level", default=None, help="override the configured level")
    return parser.parse_args(argv)


def select_models(names: list[str]) -> tuple[RatingModel, ...]:
    chosen = names or sorted(MODELS)
    return tuple(MODELS[name]() for name in chosen)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(level=args.log_level or settings.logging.level, fmt=settings.logging.format)

    matches_path = args.matches or settings.paths.processed_dir / MATCHES_FILENAME
    if not matches_path.is_file():
        logger.error("no match table at %s; run scripts/fetch_data.py first", matches_path)
        return 1

    # Through the store rather than pandas: the competition filter and the
    # ordering are the store's job, and an entry point that opens a path is one
    # more caller to rewrite the day the matches stop living in a file.
    with DuckDBStore.open_matches(matches_path) as store:
        matches = store.read_matches(competitions=args.competition or None)
    if matches.empty:
        logger.error(
            "no matches for %s", ", ".join(args.competition) if args.competition else matches_path
        )
        return 1

    if args.fit_until is not None:
        window = matches[matches["date"] < args.fit_until]
        if window.empty:
            logger.error("no matches before %s", args.fit_until)
            return 1
        parameters, error = elo_module.fit(window)
        print(f"fitted on {len(window):,} matches before {args.fit_until}")
        print(f"  k               = {parameters.k:g}")
        print(f"  home_advantage  = {parameters.home_advantage:g}")
        print(f"  season_carry    = {parameters.season_carry:g}")
        print(f"  mean squared error = {error:.5f}")
        print("\nEdit src/ratings/elo.py:EloParameters to adopt these.")
        return 0

    report = run_ratings(
        matches,
        args.output or settings.paths.features_dir,
        models=select_models(args.model),
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
