#!/usr/bin/env python
"""Regenerate docs/MODEL_CARD.md for the shipped model, from what measured it.

    python scripts/model_card.py                  # the calibrated blend
    python scripts/model_card.py --model xgboost --scores data/reports/zoo/backtest.parquet
    python scripts/model_card.py --dry-run        # print the breakdowns, write nothing

The card is generated for the same reason the dataset card is: one typed by
hand is one whose numbers came from whichever run its author had open, and the
figures that go stale first — the worst competition, the date the history stops
— are the ones a reader actually needs.

It costs a walk over the folds. The scored table on disk holds means, and the
card's subject is where the model is honest, which is a question about
individual probabilities: per class, and per competition. Both breakdowns are
printed here as well as written, because the run that produces them is long
enough that nobody should have to open a file to see whether it worked.

Reads the ensemble backtest by default, so the row it reports is the
shipped model. Exits non-zero if the tables or the scores are missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation.reliability import DEFAULT_BINS  # noqa: E402
from src.models.calibration import Calibrated  # noqa: E402
from src.models.dataset import DESIGN_COLUMNS  # noqa: E402
from src.models.ensemble import fold_forecasts  # noqa: E402
from src.models.splits import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_HORIZON_DAYS,
    SplitError,
)
from src.models.zoo import FAMILIES, ModelError, build  # noqa: E402
from src.pipelines.backtest import BACKTEST_FILENAME  # noqa: E402
from src.pipelines.report import (  # noqa: E402
    SHIPPED,
    build_model_card,
    market_comparison,
    reliability_by_class,
    reliability_by_competition,
    shipped_forecaster,
    write_forecasts,
    write_market,
    write_model_card,
)
from src.pipelines.tables import (  # noqa: E402
    load_modelling_frame,
    read_scores,
    resolve_tables,
)
from src.pipelines.train import ENSEMBLE_SUBDIR  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402

logger = get_logger("model-card")

DOCS_DIRNAME = "docs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matches", type=Path, default=None, help="canonical table to read")
    parser.add_argument("--ratings", type=Path, default=None, help="ratings table to join")
    parser.add_argument("--features", type=Path, default=None, help="feature table to join")
    parser.add_argument("--scores", type=Path, default=None, help="backtest scores to report")
    parser.add_argument("--docs", type=Path, default=None, help="where the card is written")
    parser.add_argument(
        "--model",
        default=None,
        choices=[*sorted(FAMILIES), SHIPPED],
        help=f"the model the card is about. Defaults to {SHIPPED}.",
    )
    parser.add_argument(
        "--competition",
        action="append",
        default=[],
        metavar="ID",
        help="restrict to this competition. Repeatable.",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="walk-forward steps")
    parser.add_argument(
        "--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS, help="days scored per fold"
    )
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="reliability bins")
    parser.add_argument("--dry-run", action="store_true", help="print, but write nothing")
    parser.add_argument("--log-level", default=None, help="override the configured level")
    return parser.parse_args(argv)


def forecaster(name: str) -> Calibrated:
    """The thing the card is about, by name.

    The shipped model is a wrapper around a blend and has no entry in the zoo;
    everything else is one family wrapped the same way, so a card for a single
    model reports the same kind of probabilities as the card for the blend
    rather than uncalibrated ones. Either way the name it reports under is the
    wrapper's, which is what the score table calls it.
    """
    return shipped_forecaster() if name == SHIPPED else Calibrated(build(name))


def render(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "  (nothing to report)"
    return frame.to_string(index=False, float_format=lambda value: f"{value:.4f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(level=args.log_level or settings.logging.level, fmt=settings.logging.format)

    name = args.model or SHIPPED
    tables = resolve_tables(
        settings.paths, matches=args.matches, ratings=args.ratings, features=args.features
    )
    frame = load_modelling_frame(tables, competitions=args.competition or None)
    if frame is None:
        return 1

    scores = read_scores(
        args.scores or settings.paths.reports_dir / ENSEMBLE_SUBDIR / BACKTEST_FILENAME
    )
    if scores is None:
        logger.error("no scores to report; run `make ensemble` first")
        return 1

    model = forecaster(name)
    try:
        forecasts = fold_forecasts(frame, [model], folds=args.folds, horizon_days=args.horizon_days)
    except (SplitError, ModelError) as error:
        logger.error("%s", error)
        return 1

    print(f"\n{model.name} over {len(forecasts):,} match forecasts\n")
    print("How honest it is about each class:")
    for outcome, table in reliability_by_class(forecasts, bins=args.bins).items():
        print(f"\n  {outcome}:")
        print(render(table))
    print("\nWhere it is least reliable, by competition:")
    print(render(reliability_by_competition(forecasts, bins=args.bins).head(10)))

    # Printed rather than only written, because the direction
    # down this table is the finding and a number in a Parquet file nobody
    # reads at the end of a five-minute command is a finding nobody has.
    market = market_comparison(forecasts, frame, name=model.name)
    print("\nAgainst the closing line, by how far apart the two were:")
    print(render(market))

    card = build_model_card(
        scores,
        forecasts,
        frame,
        name=model.name,
        columns=DESIGN_COLUMNS if name == SHIPPED else build(name).columns,
        folds=args.folds,
        bins=args.bins,
    )
    if args.dry_run:
        print("\n(dry run: nothing written)")
        return 0

    # The forecasts as well as the card. They cost most of this command's
    # runtime, they are the only rows that can answer whether a stated
    # probability happens at the rate it states, and until now they were
    # computed and thrown away — so the dashboard would have had to spend the
    # same five minutes on every page load.
    scores_path = args.scores or settings.paths.reports_dir / ENSEMBLE_SUBDIR / BACKTEST_FILENAME
    recorded = write_forecasts(forecasts, scores_path.parent, sources=tables.inputs())
    print(f"\n{len(forecasts):,} forecasts written to {recorded}")

    compared = write_market(market, scores_path.parent, sources=tables.inputs())
    print(f"{len(market)} disagreement band(s) written to {compared}")

    written = write_model_card(card, args.docs or PROJECT_ROOT / DOCS_DIRNAME)
    print(f"written to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
