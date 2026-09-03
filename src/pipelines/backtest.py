"""Score every baseline over walk-forward folds, per competition and pooled.

Orchestration only, in the shape the other pipelines use: run the producers —
here, forecasters over folds — check the result, persist it with a manifest,
report. What is specific to this one is *what it refuses to average*.

**Two subsets, always.** The bookmaker prices about three-quarters of the
matches Dixon-Coles can price, and putting one number from 12,000 matches
beside another from 8,800 is a comparison of two different questions. So every
forecaster is scored twice: over the matches it could price, with its coverage
stated, and over the matches *every* forecaster could price, which is the only
table where the rows may be read against each other.

**The finest grain is what gets written.** One row per fold, per competition,
per forecaster, per subset. Log loss, RPS and accuracy are all means over
matches, so a weighted mean by ``n`` reconstructs any coarser view exactly —
which is why the pooled figures are computed rather than scored a second time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import Scores, terms
from src.ingestion.base import TARGET_COLUMN
from src.models.baselines import Forecaster, default_forecasters
from src.models.splits import DEFAULT_FOLDS, DEFAULT_HORIZON_DAYS, Fold, walk_forward
from src.pipelines.derived import persist
from src.utils.logging import get_logger

logger = get_logger(__name__)

BACKTEST_FILENAME = "backtest.parquet"

POOLED = "ALL"
"""The competition_id standing for every competition at once. A word rather
than a null, because a null in a grouping column is a row people drop by
accident."""

PRICED = "priced"
COMMON = "common"
"""The two subsets. ``priced`` is what a forecaster could price; ``common`` is
what every forecaster could price, and is the only one where two rows are
comparable."""

SCORE_COLUMNS: dict[str, str] = {
    "fold": "Int16",
    "competition_id": "string",
    "forecaster": "string",
    "subset": "string",
    "n": "Int32",
    "log_loss": "Float64",
    "rps": "Float64",
    "accuracy": "Float64",
}


@dataclass
class BacktestReport:
    """What one backtest did."""

    folds: int = 0
    matches: int = 0
    forecasters: tuple[str, ...] = ()
    coverage: dict[str, float] = field(default_factory=dict)
    """Per forecaster, the share of evaluation matches it could price."""

    scores: pd.DataFrame | None = None
    output: Path | None = None

    def pooled(self, subset: str = COMMON) -> dict[str, Scores]:
        """The headline table: one row per forecaster, over every fold."""
        if self.scores is None:
            return {}
        return {
            str(name): _weighted(group)
            for name, group in _pooled_rows(self.scores, subset).groupby("forecaster")
        }

    def summary(self) -> str:
        parts = [f"{self.matches:,} matches over {self.folds} fold(s)"]
        parts += [
            f"{name} {scores.log_loss:.4f}/{scores.rps:.4f}"
            for name, scores in sorted(self.pooled().items())
        ]
        return "; ".join(parts)


def _weighted(rows: pd.DataFrame) -> Scores:
    """Collapse score rows by weighting each metric by the matches behind it.

    Every metric here is a mean over matches, so this is the same number a
    single pass over the union would produce — not an approximation of it, and
    not the average-of-averages that would quietly over-weight a small fold.
    """
    weights = rows["n"].to_numpy(dtype=float)
    return Scores(
        n=int(weights.sum()),
        log_loss=float(np.average(rows["log_loss"].to_numpy(dtype=float), weights=weights)),
        rps=float(np.average(rows["rps"].to_numpy(dtype=float), weights=weights)),
        accuracy=float(np.average(rows["accuracy"].to_numpy(dtype=float), weights=weights)),
    )


def _pooled_rows(scores: pd.DataFrame, subset: str) -> pd.DataFrame:
    return scores[(scores["competition_id"] == POOLED) & (scores["subset"] == subset)]


POOLED_COLUMNS: tuple[str, ...] = ("forecaster", "n", "log_loss", "rps", "accuracy")


def pooled_table(scores: pd.DataFrame, subset: str = COMMON) -> pd.DataFrame:
    """Every forecaster's score over every fold, as one row each.

    Empty when the subset is: the common subset is empty whenever one
    forecaster could price nothing at all, which is a real answer — usually
    that the ratings table was built for a different set of matches — and not
    a reason to fail.
    """
    rows = [
        {"forecaster": str(name), **asdict(_weighted(group))}
        for name, group in _pooled_rows(scores, subset).groupby("forecaster")
    ]
    if not rows:
        return pd.DataFrame(columns=list(POOLED_COLUMNS))
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def per_competition_table(
    scores: pd.DataFrame, subset: str = COMMON, metric: str = "log_loss"
) -> pd.DataFrame:
    """One row per competition, one column per forecaster.

    Folds are collapsed by the same weighting :func:`pooled_table` uses, so a
    competition's figure here is what a single pass over its evaluation matches
    would have produced. Sorted by match count: a competition contributing
    forty matches and one contributing four thousand are not equally
    interesting, and alphabetical order hides which is which.
    """
    rows = scores[(scores["competition_id"] != POOLED) & (scores["subset"] == subset)]
    collapsed = [
        {
            "competition_id": str(competition),
            "forecaster": str(name),
            "n": _weighted(group).n,
            metric: getattr(_weighted(group), metric),
        }
        for (competition, name), group in rows.groupby(["competition_id", "forecaster"])
    ]
    if not collapsed:
        return pd.DataFrame(columns=["competition_id", "n"])

    tidy = pd.DataFrame(collapsed)
    counts = tidy.groupby("competition_id")["n"].max()
    table = tidy.pivot(index="competition_id", columns="forecaster", values=metric)
    table.insert(0, "n", counts)
    return table.sort_values("n", ascending=False).reset_index().rename_axis(None, axis=1)


def score_fold(fold: Fold, forecasters: Sequence[Forecaster]) -> pd.DataFrame:
    """Score every forecaster over one fold, per competition and pooled."""
    outcomes = fold.evaluate[TARGET_COLUMN]
    forecasts = {
        forecaster.name: forecaster.forecast(fold.train, fold.evaluate)
        for forecaster in forecasters
    }
    priced = {name: ~np.isnan(values).any(axis=1) for name, values in forecasts.items()}
    common = np.logical_and.reduce(list(priced.values())) if priced else np.zeros(0, dtype=bool)

    tidy: list[pd.DataFrame] = []
    for name, values in forecasts.items():
        for subset, mask in ((PRICED, priced[name]), (COMMON, common)):
            if not mask.any():
                continue
            contributions = terms(values[mask], outcomes[mask])
            contributions["competition_id"] = fold.evaluate["competition_id"][mask].to_numpy()
            contributions["forecaster"] = name
            contributions["subset"] = subset
            tidy.append(contributions)

    if not tidy:
        return pd.DataFrame(columns=list(SCORE_COLUMNS)).astype(SCORE_COLUMNS)

    together = pd.concat(tidy, ignore_index=True)
    per_competition = _aggregate(together, ["competition_id", "forecaster", "subset"])
    everywhere = _aggregate(together, ["forecaster", "subset"]).assign(competition_id=POOLED)
    scored = pd.concat([per_competition, everywhere], ignore_index=True)
    scored["fold"] = fold.index
    return scored.reindex(columns=list(SCORE_COLUMNS)).astype(SCORE_COLUMNS)


def _aggregate(tidy: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean each metric over the matches in each group."""
    grouped = tidy.groupby(by, observed=True).agg(
        n=("log_loss", "size"),
        log_loss=("log_loss", "mean"),
        rps=("rps", "mean"),
        accuracy=("hit", "mean"),
    )
    return grouped.reset_index()


def run_backtest(
    matches: pd.DataFrame,
    reports_dir: Path,
    *,
    forecasters: Sequence[Forecaster] | None = None,
    folds: int = DEFAULT_FOLDS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> BacktestReport:
    """Walk forward, score every baseline, and persist the result.

    Args:
        matches: The canonical table joined to whatever a forecaster reads —
            the ratings, in the default set — sorted by date.
        reports_dir: Destination for ``backtest.parquet``.
        forecasters: Defaults to every baseline the table can support.
        folds: How many walk-forward steps.
        horizon_days: How much time each step is scored over.
    """
    chosen = tuple(forecasters if forecasters is not None else default_forecasters(matches))
    report = BacktestReport(forecasters=tuple(one.name for one in chosen))

    scored: list[pd.DataFrame] = []
    for fold in walk_forward(matches, folds=folds, horizon_days=horizon_days):
        scored.append(score_fold(fold, chosen))
        report.folds += 1
        report.matches += len(fold.evaluate)

    # `walk_forward` raises rather than yielding nothing, so there is at least
    # one fold here and at least one match in it.
    report.scores = pd.concat(scored, ignore_index=True)
    priced = _pooled_rows(report.scores, PRICED)
    for name in report.forecasters:
        matched = priced[priced["forecaster"] == name]["n"].sum()
        report.coverage[name] = float(matched) / report.matches

    report.output = persist(
        report.scores,
        reports_dir / BACKTEST_FILENAME,
        extra={
            "kind": "backtest",
            "forecasters": list(report.forecasters),
            "folds": report.folds,
            "horizon_days": horizon_days,
            "matches": report.matches,
            "coverage": report.coverage,
        },
    )
    logger.info("wrote %s — %s", report.output.name, report.summary())
    return report
