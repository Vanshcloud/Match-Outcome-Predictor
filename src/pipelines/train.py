"""Fit the zoo over the walk-forward folds, ablate it, blend it, calibrate it.

Orchestration only, and deliberately thin: a trained model is a
:class:`~src.models.baselines.Forecaster` that happens to fit inside its own
``forecast``, so **the evaluation pipeline is used unchanged**. Nothing in :mod:`src.pipelines.backtest` knows an estimator exists. That
is what keeps a model's score comparable with a baseline's: the same folds, the
same two subsets, the same weighting, the same file format.

The ablation is the same trick again. Rather than running six backtests and
comparing tables computed over six different common subsets, it builds one
forecaster per variant — the model with all thirty columns, and the model with
one block withheld — and runs them **in a single backtest**. Every variant is
then scored on identical matches, and the difference between two rows is the
block, not the subset.

The ensemble and calibration layer arrive the same way: both are
:class:`~src.models.baselines.Forecaster` implementations, so the blend, the
calibrated model and the four baselines are one more list handed to the same
unchanged backtest, and the gap to the closing line is measured on the same
matches throughout.

**Reliability costs a second pass, on purpose.** The backtest persists means —
one score per fold, competition, forecaster and subset — which is right for a
report and useless for asking whether a stated 30% happens 30% of the time.
That question needs the matches back, so :func:`reliability_tables` walks the
folds again through :func:`~src.models.ensemble.fold_forecasts`. The cheaper
arrangement is a side channel out of the scoring pipeline, and the pipeline
staying ignorant of what it is scoring is worth more than the minutes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from src.evaluation.reliability import DEFAULT_BINS, reliability
from src.ingestion.base import TARGET_COLUMN
from src.models.baselines import Forecaster, default_forecasters
from src.models.calibration import Calibrated
from src.models.dataset import BLOCKS, DESIGN_COLUMNS, without
from src.models.ensemble import BEST, FORECAST_COLUMNS, MEMBERS, ensemble, fold_forecasts
from src.models.splits import DEFAULT_FOLDS, DEFAULT_HORIZON_DAYS
from src.models.tracking import log_run
from src.models.zoo import TrainedForecaster, build, default_zoo
from src.pipelines.backtest import (
    COMMON,
    BacktestReport,
    pooled_table,
    run_backtest,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

ZOO_SUBDIR = "zoo"
ABLATION_SUBDIR = "ablation"
ENSEMBLE_SUBDIR = "ensemble"
"""Subdirectories of the reports directory.

Each holds a ``backtest.parquet`` written by the unchanged evaluation pipeline.
Separate directories rather than separate filenames, because the filename is
the evaluation layer's to choose and this layer has no business renaming it.
"""

ALL_BLOCKS = "none"
"""The ablation's control: the variant with nothing withheld."""


@dataclass
class TrainingReport:
    """What one training run did."""

    backtest: BacktestReport | None = None
    models: tuple[str, ...] = ()
    tracked: dict[str, str] = field(default_factory=dict)
    """Model name to MLflow run id, for the runs that were tracked. Missing
    entries are runs that were not — tracking is allowed to fail."""

    def summary(self) -> str:
        if self.backtest is None:
            return "nothing trained"
        return self.backtest.summary()


def zoo_forecasters(
    columns: Sequence[str] = DESIGN_COLUMNS, models: Sequence[str] | None = None
) -> tuple[TrainedForecaster, ...]:
    """The requested models, or every family."""
    if models is None:
        return default_zoo(columns)
    return tuple(build(name, columns) for name in models)


def run_training(
    matches: pd.DataFrame,
    reports_dir: Path,
    *,
    models: Sequence[str] | None = None,
    columns: Sequence[str] = DESIGN_COLUMNS,
    baselines: bool = True,
    folds: int = DEFAULT_FOLDS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    model_dir: Path | None = None,
    sources: Sequence[Path] = (),
) -> TrainingReport:
    """Score every model over the folds, beside the baselines they must beat.

    Args:
        matches: The canonical table joined to the ratings and the features,
            sorted by date.
        reports_dir: Parent of :data:`ZOO_SUBDIR`.
        models: Model names. Defaults to the whole zoo.
        columns: The design matrix. Defaults to all thirty.
        baselines: Score the four baselines in the same run. On by
            default: a model's log loss means nothing without the class prior
            beside it, computed on the same matches.
        folds: Walk-forward steps.
        horizon_days: Days scored per step.
        model_dir: Where MLflow's file store lives. ``None`` skips tracking.
    """
    chosen = zoo_forecasters(columns, models)
    forecasters: tuple[Forecaster, ...] = chosen
    if baselines:
        forecasters = (*default_forecasters(matches), *chosen)

    report = TrainingReport(models=tuple(model.name for model in chosen))
    backtest = run_backtest(
        matches,
        reports_dir / ZOO_SUBDIR,
        forecasters=forecasters,
        folds=folds,
        horizon_days=horizon_days,
        sources=sources,
    )
    report.backtest = backtest

    if model_dir is not None:
        _track(report, chosen, backtest, model_dir)
    logger.info("trained %s — %s", ", ".join(report.models), report.summary())
    return report


def _track(
    report: TrainingReport,
    models: Sequence[TrainedForecaster],
    backtest: BacktestReport,
    model_dir: Path,
) -> None:
    """Log one MLflow run per model, with its settings and its pooled scores.

    Takes the backtest it should read rather than reaching back through the
    report for it. The report's field is optional — a `TrainingReport` can be
    constructed empty — and narrowing that with a runtime assertion would put a
    check in the shipped path whose only job is to satisfy the type checker.
    """
    scored = backtest.scores if backtest.scores is not None else pd.DataFrame({"forecaster": []})
    scores = pooled_table(scored, COMMON).set_index("forecaster")
    for model in models:
        if model.name not in scores.index:
            continue
        row = scores.loc[model.name]
        run = log_run(
            model.name,
            {**model.parameters, "columns": len(model.columns), "folds": backtest.folds},
            {
                "log_loss": float(row["log_loss"]),
                "rps": float(row["rps"]),
                "accuracy": float(row["accuracy"]),
                "matches": float(row["n"]),
            },
            model_dir=model_dir,
        )
        if run is not None:
            report.tracked[model.name] = run


def ablation_forecasters(model: str, blocks: Sequence[str] | None = None) -> tuple[Forecaster, ...]:
    """One forecaster per variant: the full model, then one block withheld each.

    Named for what is missing, because that is what the table is read for.
    """
    chosen = tuple(blocks) if blocks is not None else tuple(BLOCKS)
    full = build(model, DESIGN_COLUMNS)
    return (
        replace(full, name=f"{model}-{ALL_BLOCKS}"),
        *(replace(full, name=f"{model}-no-{block}", columns=without(block)) for block in chosen),
    )


def run_ablation(
    matches: pd.DataFrame,
    reports_dir: Path,
    *,
    model: str,
    blocks: Sequence[str] | None = None,
    folds: int = DEFAULT_FOLDS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    sources: Sequence[Path] = (),
) -> BacktestReport:
    """Score the model with each feature block withheld, all in one backtest.

    One run rather than one per block, so every variant is scored over the same
    matches. Six separate backtests would each compute their own common subset,
    and the differences between those subsets are the same size as the
    differences the ablation is trying to measure.
    """
    return run_backtest(
        matches,
        reports_dir / ABLATION_SUBDIR,
        forecasters=ablation_forecasters(model, blocks),
        folds=folds,
        horizon_days=horizon_days,
        sources=sources,
    )


def ablation_table(scores: pd.DataFrame, model: str) -> pd.DataFrame:
    """What each block is worth, as a log-loss delta against the full model.

    Positive is a block that earns its place: withholding it made the model
    worse. Negative is a block the model would rather not have had, which is a
    result and not a bug — rest days were added knowing they carried no
    marginal signal, and docs/FEATURES.md says so.
    """
    pooled = pooled_table(scores, COMMON).set_index("forecaster")
    control = f"{model}-{ALL_BLOCKS}"
    if control not in pooled.index:
        return pd.DataFrame(columns=["block", "log_loss", "delta_log_loss", "delta_rps"])

    rows = [
        {
            "block": name.removeprefix(f"{model}-no-"),
            "log_loss": float(pooled.loc[name, "log_loss"]),
            "delta_log_loss": float(pooled.loc[name, "log_loss"] - pooled.loc[control, "log_loss"]),
            "delta_rps": float(pooled.loc[name, "rps"] - pooled.loc[control, "rps"]),
        }
        for name in pooled.index
        if name.startswith(f"{model}-no-")
    ]
    return pd.DataFrame(rows).sort_values("delta_log_loss", ascending=False).reset_index(drop=True)


def ensemble_forecasters(
    model: str = BEST,
    members: Sequence[str] = MEMBERS,
    columns: Sequence[str] = DESIGN_COLUMNS,
) -> tuple[Forecaster, ...]:
    """The four rows the ensemble report compares.

    The best single family, that family calibrated, the blend, and the blend
    calibrated — in one list, so one backtest scores them on identical matches
    and the difference between two rows is the layer rather than the subset.
    Every one of them is measured against the same four baselines in the same
    run, which is what keeps the remaining gap to the closing line a number
    about football rather than about which table it was read off.
    """
    single = build(model, columns)
    blend = ensemble(members, columns)
    return (single, Calibrated(single), blend, Calibrated(blend))


def run_ensemble(
    matches: pd.DataFrame,
    reports_dir: Path,
    *,
    model: str = BEST,
    members: Sequence[str] = MEMBERS,
    columns: Sequence[str] = DESIGN_COLUMNS,
    baselines: bool = True,
    folds: int = DEFAULT_FOLDS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    sources: Sequence[Path] = (),
) -> BacktestReport:
    """Score the blend and the calibration layer beside the model they are made
    of, through the unchanged backtest."""
    chosen = ensemble_forecasters(model, members, columns)
    forecasters = (*default_forecasters(matches), *chosen) if baselines else chosen
    return run_backtest(
        matches,
        reports_dir / ENSEMBLE_SUBDIR,
        forecasters=forecasters,
        folds=folds,
        horizon_days=horizon_days,
        sources=sources,
    )


def reliability_tables(
    matches: pd.DataFrame,
    forecasters: Sequence[Forecaster],
    *,
    folds: int = DEFAULT_FOLDS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    bins: int = DEFAULT_BINS,
) -> dict[str, pd.DataFrame]:
    """One reliability table per forecaster, over the same folds.

    Unpriced rows are dropped per forecaster rather than reduced to a common
    subset: reliability is a statement about the probabilities a forecaster
    actually issued, and the bookmaker's line is not less honest for being
    absent on the matches it never quoted.
    """
    forecasts = fold_forecasts(matches, forecasters, folds=folds, horizon_days=horizon_days)
    priced = forecasts.dropna(subset=list(FORECAST_COLUMNS))
    return {
        str(name): reliability(
            group[list(FORECAST_COLUMNS)].to_numpy(dtype=float), group[TARGET_COLUMN], bins=bins
        )
        for name, group in priced.groupby("forecaster")
    }
