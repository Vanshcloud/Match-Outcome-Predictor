#!/usr/bin/env python
"""Fit the model zoo over the walk-forward folds, ablate it, blend it, or tune it.

    python scripts/train.py                       # every model, beside the baselines
    python scripts/train.py --model lightgbm --model xgboost
    python scripts/train.py --ablate lightgbm     # what each feature block is worth
    python scripts/train.py --ensemble            # the blend and the calibration layer
    python scripts/train.py --correlations        # whose errors are alike, and who that admits
    python scripts/train.py --tune xgboost        # search, then print the constants
    python scripts/train.py --markdown            # the tables in docs/

Models are scored by the same backtest the baselines are, on the same folds and
the same two subsets — a trained model is just a forecaster that fits inside
its own `forecast`, so nothing in the evaluation layer knows an estimator
exists. The baselines are included in the run by default, because a model's log
loss means nothing without the class prior beside it on the same matches.

A full run is about five minutes, most of it the MLP and the random forest.
`--ensemble` is about four times that: the calibration layer fits its inner
model twice per fold, and the reliability tables need a second pass over the
folds because the backtest persists means rather than matches. `--tune` is
longer still: one trial is three fits over most of the history.

Exits non-zero if the tables cannot be read or no fold can be built.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation.reliability import (  # noqa: E402
    DEFAULT_BINS,
    expected_calibration_error,
)
from src.models.dataset import BLOCKS  # noqa: E402
from src.models.ensemble import (  # noqa: E402
    BEST,
    MAX_ERROR_CORRELATION,
    MEMBERS,
    by_log_loss,
    error_correlations,
    fold_forecasts,
    select_members,
)
from src.models.splits import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_HORIZON_DAYS,
    SplitError,
)
from src.models.tuning import (  # noqa: E402
    DEFAULT_TRIALS,
    TUNING_FOLDS,
    as_constants,
    tune,
    tuning_slice,
)
from src.models.zoo import FAMILIES, ModelError, default_zoo  # noqa: E402
from src.pipelines.backtest import COMMON, PRICED, pooled_table  # noqa: E402
from src.pipelines.tables import load_modelling_frame, resolve_tables  # noqa: E402
from src.pipelines.train import (  # noqa: E402
    ablation_table,
    ensemble_forecasters,
    reliability_tables,
    run_ablation,
    run_ensemble,
    run_training,
)
from src.utils.config import Settings, load_settings  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("train")


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
        "--ensemble",
        nargs="?",
        const=BEST,
        default=None,
        metavar="MODEL",
        choices=sorted(FAMILIES),
        help="score the blend and MODEL's calibration layer beside MODEL. Defaults to the "
        "best family.",
    )
    parser.add_argument(
        "--member",
        action="append",
        default=[],
        choices=sorted(FAMILIES),
        help="a blend member. Repeatable. Defaults to the measured set.",
    )
    parser.add_argument(
        "--correlations",
        action="store_true",
        help="print how alike the families' errors are, on matches earlier than every reported fold",
    )
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="reliability bins")
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
    tables = resolve_tables(
        settings.paths, matches=args.matches, ratings=args.ratings, features=args.features
    )
    return load_modelling_frame(tables, competitions=args.competition or None)


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


def report_ensemble(frame: pd.DataFrame, args: argparse.Namespace, settings: Settings) -> int:
    """The blend, the calibration layer, and how honest each of them is.

    Two passes over the folds, which is deliberate. The first is the unchanged
    backtest, and it produces the scored table beside the same baselines. The
    second re-forecasts the folds to get the matches back, because reliability
    is a question about individual probabilities and the backtest persists
    means — the alternative is a side channel out of a scoring pipeline whose
    ignorance of what it scores is the reason a model and a baseline can share
    a table.
    """
    paths = settings.paths
    members = args.member or MEMBERS
    report = run_ensemble(
        frame,
        args.output or paths.reports_dir,
        model=args.ensemble,
        members=members,
        baselines=not args.no_baselines,
        folds=args.folds,
        horizon_days=args.horizon_days,
    )
    if report.scores is None:  # pragma: no cover - run_backtest always writes one
        logger.error("nothing was scored")
        return 1

    print(f"\n{report.folds} fold(s), {report.matches:,} matches scored\n")
    print("Every forecaster over the matches all of them could price:")
    print(render(pooled_table(report.scores, COMMON), markdown=args.markdown))

    tables = reliability_tables(
        frame,
        ensemble_forecasters(args.ensemble, members),
        folds=args.folds,
        horizon_days=args.horizon_days,
        bins=args.bins,
    )
    print("\nHow honest each is — mean gap between a stated probability and how")
    print("often it happened, over the same folds:")
    errors = pd.DataFrame(
        [
            {"forecaster": name, "calibration_error": expected_calibration_error(table)}
            for name, table in tables.items()
        ]
    ).sort_values("calibration_error")
    print(render(errors, markdown=args.markdown))

    for name in (args.ensemble, f"{args.ensemble}-calibrated"):
        print(f"\n{name}, by stated probability:")
        print(render(tables[name], markdown=args.markdown))

    print(f"\nwritten to {report.output}")
    return 0


def report_correlations(frame: pd.DataFrame, args: argparse.Namespace) -> int:
    """Whose mistakes are alike, and which members that admits.

    On the tuning slice, for the reason `--tune` is: members chosen by looking
    at errors on the folds they are then scored on is selection on the test set
    with an extra step.
    """
    slice_ = tuning_slice(frame, folds=args.folds, horizon_days=args.horizon_days)
    print(
        f"correlating the zoo's errors on {len(slice_):,} matches to "
        f"{slice_['date'].max():%Y-%m-%d} — every one earlier than the first reported fold\n"
    )
    forecasts = fold_forecasts(
        slice_, default_zoo(), folds=TUNING_FOLDS, horizon_days=args.horizon_days
    )
    matrix = error_correlations(forecasts)
    print(render(matrix.reset_index(names="forecaster"), markdown=args.markdown))

    order = by_log_loss(forecasts)
    chosen = select_members(matrix, order)
    print(f"\nbest first: {', '.join(order)}")
    print(f"admitted below {MAX_ERROR_CORRELATION}: {', '.join(chosen)}\n")
    print(f"MEMBERS: tuple[str, ...] = {chosen!r}")
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
        if args.correlations:
            return report_correlations(frame, args)
        if args.ensemble is not None:
            return report_ensemble(frame, args, settings)
        if args.ablate is not None:
            return report_ablation(frame, args, settings)
        return report_training(frame, args, settings)
    except (SplitError, ModelError) as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
