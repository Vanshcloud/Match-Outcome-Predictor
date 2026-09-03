#!/usr/bin/env python
"""Fit the model zoo over the walk-forward folds, ablate it, or tune it.

    python scripts/train.py                       # every model, beside the baselines
    python scripts/train.py --model lightgbm --model xgboost
    python scripts/train.py --ablate lightgbm     # what each feature block is worth
    python scripts/train.py --tune xgboost        # search, then print the constants
    python scripts/train.py --markdown            # the tables in docs/

Models are scored by the same backtest the baselines are, on the same folds and
the same two subsets — a trained model is just a forecaster that fits inside
its own `forecast`, so nothing in the evaluation layer knows an estimator
exists. The baselines are included in the run by default, because a model's log
loss means nothing without the class prior beside it on the same matches.

A full run is about ten minutes, most of it the MLP and the random forest.
`--tune` is far longer: one trial is three fits over most of the history.

Exits non-zero if the tables cannot be read or no fold can be built.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.models.dataset import BLOCKS  # noqa: E402
from src.models.splits import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_HORIZON_DAYS,
    SplitError,
)
from src.models.tuning import DEFAULT_TRIALS, as_constants, tune, tuning_slice  # noqa: E402
from src.models.zoo import FAMILIES, ModelError  # noqa: E402
from src.pipelines.backtest import COMMON, PRICED, pooled_table  # noqa: E402
from src.pipelines.features import FEATURES_FILENAME  # noqa: E402
from src.pipelines.ingest import MATCHES_FILENAME  # noqa: E402
from src.pipelines.ratings import RATINGS_FILENAME  # noqa: E402
from src.pipelines.train import ablation_table, run_ablation, run_training  # noqa: E402
from src.storage.duckdb_store import DuckDBStore  # noqa: E402
from src.utils.config import Settings, load_settings  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("train")

KEY_COLUMN = "match_id"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matches", type=Path, default=None, help="canonical table to read")
    parser.add_argument("--ratings", type=Path, default=None, help="ratings table to join")
    parser.add_argument("--features", type=Path, default=None, help="feature table to join")
    parser.add_argument("--output", type=Path, default=None, help="reports directory")
    parser.add_argument(
        "--competition",
        action="append",
        default=[],
        metavar="ID",
        help="restrict to this competition. Repeatable.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        choices=sorted(FAMILIES),
        help="model family. Repeatable. Defaults to all six.",
    )
    parser.add_argument(
        "--ablate",
        metavar="MODEL",
        default=None,
        choices=sorted(FAMILIES),
        help="score this model with each feature block withheld, in one run",
    )
    parser.add_argument(
        "--tune",
        metavar="MODEL",
        default=None,
        choices=sorted(FAMILIES),
        help="search this model's space on matches earlier than every reported fold",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="search budget")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="walk-forward steps")
    parser.add_argument(
        "--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS, help="days scored per fold"
    )
    parser.add_argument("--no-baselines", action="store_true", help="score the models on their own")
    parser.add_argument(
        "--no-tracking", action="store_true", help="skip the MLflow run log entirely"
    )
    parser.add_argument("--markdown", action="store_true", help="print the tables as markdown")
    parser.add_argument("--log-level", default=None, help="override the configured level")
    return parser.parse_args(argv)


def _cell(value: object, *, signed: bool) -> str:
    if not isinstance(value, float):
        return str(value)
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def render(table: pd.DataFrame, *, markdown: bool, signed: bool = False) -> str:
    """A score table as text or markdown.

    ``signed`` is for the ablation and nothing else: a leading ``+`` on a delta
    says which way it went, and the same ``+`` on a log loss is noise dressed
    up as precision.
    """
    if table.empty:
        return "  (nothing to report)"
    if not markdown:
        return table.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    header = list(table.columns)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += [
        "| " + " | ".join(_cell(value, signed=signed) for value in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join(lines)


def load_frame(args: argparse.Namespace, settings: Settings) -> pd.DataFrame | None:
    """The canonical table joined to the ratings and the features, or None."""
    paths = settings.paths
    matches_path = args.matches or paths.processed_dir / MATCHES_FILENAME
    ratings_path = args.ratings or paths.features_dir / RATINGS_FILENAME
    features_path = args.features or paths.features_dir / FEATURES_FILENAME

    for label, path in (
        ("match table", matches_path),
        ("ratings", ratings_path),
        ("features", features_path),
    ):
        if not path.is_file():
            logger.error("no %s at %s; a model needs all three tables", label, path)
            return None

    with DuckDBStore.open_matches(
        matches_path, ratings=ratings_path, features=features_path
    ) as store:
        matches = store.read_matches(competitions=args.competition or None)
        ratings = store.read_ratings()
        features = store.read_features()

    if matches.empty:
        logger.error("no matches for %s", ", ".join(args.competition) or matches_path)
        return None
    # Left joins: a match the ratings or the features have nothing to say about
    # keeps its row and loses those columns, which the models read as the null
    # they are — "no history yet" — rather than losing the match.
    return matches.merge(ratings, on=KEY_COLUMN, how="left").merge(
        features, on=KEY_COLUMN, how="left"
    )


def run_tuning(frame: pd.DataFrame, args: argparse.Namespace) -> int:
    slice_ = tuning_slice(frame, folds=args.folds, horizon_days=args.horizon_days)
    print(
        f"searching {args.tune} on {len(slice_):,} matches to "
        f"{slice_['date'].max():%Y-%m-%d} — every one earlier than the first reported fold\n"
    )
    result = tune(slice_, args.tune, trials=args.trials, horizon_days=args.horizon_days)
    print(result.summary())
    print("\n" + as_constants(result))
    if result.improvement <= 0:
        print("\nThe search did not beat the shipped settings. Leave them alone.")
    return 0


def report_training(frame: pd.DataFrame, args: argparse.Namespace, settings: Settings) -> int:
    paths = settings.paths
    report = run_training(
        frame,
        args.output or paths.reports_dir,
        models=args.model or None,
        baselines=not args.no_baselines,
        folds=args.folds,
        horizon_days=args.horizon_days,
        model_dir=None if args.no_tracking else paths.model_dir,
    )
    backtest = report.backtest
    if backtest is None or backtest.scores is None:  # pragma: no cover - see run_training
        logger.error("nothing was scored")
        return 1

    print(f"\n{backtest.folds} fold(s), {backtest.matches:,} matches scored\n")
    print("Every forecaster over the matches all of them could price:")
    print(render(pooled_table(backtest.scores, COMMON), markdown=args.markdown))
    print("\nEach over the matches it could price:")
    priced = pooled_table(backtest.scores, PRICED)
    priced.insert(1, "coverage", priced["forecaster"].map(backtest.coverage))
    print(render(priced, markdown=args.markdown))
    if report.tracked:
        print(f"\ntracked {len(report.tracked)} run(s) to {paths.model_dir}")
    print(f"\nwritten to {backtest.output}")
    return 0


def report_ablation(frame: pd.DataFrame, args: argparse.Namespace, settings: Settings) -> int:
    paths = settings.paths
    report = run_ablation(
        frame,
        args.output or paths.reports_dir,
        model=args.ablate,
        folds=args.folds,
        horizon_days=args.horizon_days,
    )
    if report.scores is None:  # pragma: no cover - run_backtest always writes one
        logger.error("nothing was scored")
        return 1
    print(f"\n{report.folds} fold(s), {report.matches:,} matches scored\n")
    print(f"What each of the {len(BLOCKS)} blocks is worth to {args.ablate}:")
    print("(positive delta = withholding it made the model worse)")
    print(render(ablation_table(report.scores, args.ablate), markdown=args.markdown, signed=True))
    print(f"\nwritten to {report.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(level=args.log_level or settings.logging.level, fmt=settings.logging.format)

    frame = load_frame(args, settings)
    if frame is None:
        return 1

    try:
        if args.tune is not None:
            return run_tuning(frame, args)
        if args.ablate is not None:
            return report_ablation(frame, args, settings)
        return report_training(frame, args, settings)
    except (SplitError, ModelError) as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
