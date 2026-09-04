#!/usr/bin/env python
"""Fit the shipped model on the whole history and write it down.

    python scripts/build_model.py                     # the blend the card describes
    python scripts/build_model.py --member xgboost    # one family, for comparison
    python scripts/build_model.py --competition ENG_1 # a small artefact, quickly
    python scripts/build_model.py --dry-run           # fit and report, write nothing

Every other command in this project fits a model *inside* a backtest fold and
throws it away, which is what makes the folds honest. The API cannot work that
way, so this is the one place a fit outlives its process.

**What it produces is in-sample over its whole training window, and the
artefact says so.** The manifest records the last date the fit could see, and
the service reports ``in_sample`` on every prediction about a fixture at or
before it. The out-of-sample numbers are the walk-forward ones in
docs/MODEL_CARD.md and this does not change them: it fits the same composition
on more matches, which is what you serve and not what you quote.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Same three lines as every other script here: run from anywhere, import `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.ensemble import MEMBERS  # noqa: E402
from src.pipelines.serving import (  # noqa: E402
    MODEL_FILENAME,
    build_servable,
    save_servable,
)
from src.pipelines.tables import resolve_tables  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("build-model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--matches", type=Path, default=None, help="canonical table override")
    parser.add_argument("--ratings", type=Path, default=None, help="ratings table override")
    parser.add_argument("--features", type=Path, default=None, help="feature table override")
    parser.add_argument("--output", type=Path, default=None, help="directory for the artefact")
    parser.add_argument(
        "--member",
        action="append",
        default=None,
        help=f"a blend member; repeatable. Defaults to {', '.join(MEMBERS)}.",
    )
    parser.add_argument(
        "--competition",
        action="append",
        default=None,
        help="restrict the training history to these competitions; repeatable",
    )
    parser.add_argument("--dry-run", action="store_true", help="fit and report, write nothing")
    parser.add_argument("--log-level", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(level=args.log_level or settings.logging.level, fmt=settings.logging.format)

    tables = resolve_tables(
        settings.paths, matches=args.matches, ratings=args.ratings, features=args.features
    )
    model = build_servable(
        tables,
        members=args.member or MEMBERS,
        competitions=args.competition or None,
    )
    if model is None:
        # `load_modelling_frame` has already logged which table is missing.
        # A second message here would say the same thing less specifically.
        return 1

    logger.info(
        "%s: %d matches, %s to %s, members %s, temperature %.4f",
        model.name,
        model.trained_matches,
        model.trained_from.date(),
        model.trained_through.date(),
        ", ".join(model.member_names),
        model.temperature,
    )
    if args.dry_run:
        logger.info("dry run: %s not written", MODEL_FILENAME)
        return 0

    written = save_servable(model, args.output or settings.paths.model_dir)
    print(f"\nwritten to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
