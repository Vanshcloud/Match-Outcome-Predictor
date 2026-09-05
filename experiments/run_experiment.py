"""Does the candidate block beat the thirty columns? Measured, three ways.

Run this to reproduce every number in ``experiments/README.md``:

    python -m experiments.run_experiment            # all three
    python -m experiments.run_experiment --paired   # just the headline

The protocol is the project's own, not a second one: the same
:func:`~src.models.splits.walk_forward` folds, the same
:func:`~src.models.zoo.build` estimators with the same tuned settings, the
same :func:`~src.evaluation.metrics.terms`. Only the column list changes, which
is what makes the comparison a comparison.

**Three experiments, because one is not enough to believe.**

1. ``--paired`` — base thirty against base plus fourteen, on identical folds,
   compared per match with a paired t-test. A paired test is the right one:
   both models score the same fixtures, so the pairing removes the variance
   that comes from some weeks being harder than others.
2. ``--seeds`` — the control that decides whether the headline is real. A
   gradient-boosted model refitted under a different seed lands somewhere
   slightly different, and an improvement smaller than that spread is noise
   with a p-value. Measured here rather than assumed.
3. ``--blocks`` — which of the four groups carries it, so fourteen columns are
   not adopted for two columns' worth of signal.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from experiments.candidate_features import CANDIDATE_COLUMNS, build_candidates
from src.evaluation.metrics import terms
from src.models import zoo
from src.models.dataset import DESIGN_COLUMNS
from src.models.splits import walk_forward
from src.pipelines.tables import load_modelling_frame, resolve_tables
from src.utils.config import load_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL = "xgboost"
"""The family the comparison runs on. A member of the shipped blend, and the
best of the six on the tuning slice — the same one Milestone 9 measures
calibration over, chosen for the same reason and not by looking at these folds.
"""

SEEDS: tuple[int, ...] = (20260904, 7, 99)
"""The project seed and two arbitrary others. Three is enough to see whether an
effect survives a refit; it is not enough to put an interval on."""

BLOCKS: dict[str, tuple[str, ...]] = {
    "ht (half-time, 71% coverage)": tuple(
        f"{v}_{c}" for v in ("home", "away") for c in ("ht_points_5", "ht_goals_for_5")
    ),
    "sot (shots on target, 42%)": tuple(
        f"{v}_{c}" for v in ("home", "away") for c in ("sot_for_5", "sot_against_5")
    ),
    "long form (20-match, 99.8%)": tuple(
        f"{v}_{c}" for v in ("home", "away") for c in ("form_points_20", "goal_diff_20")
    ),
    "context (tier, season_days)": ("tier", "season_days"),
}


def modelling_frame() -> pd.DataFrame:
    """The three tables joined, with the candidates alongside them.

    ``tier`` is dropped from the candidate frame before the join: it is already
    a canonical column, and merging it twice produces ``tier_x``/``tier_y`` and
    a design matrix that is missing the column it asked for.
    """
    settings = load_settings()
    frame = load_modelling_frame(resolve_tables(settings.paths))
    if frame is None:
        raise SystemExit("no tables; run `make reproduce` first")
    matches = pd.read_parquet(settings.paths.processed_dir / "matches.parquet")
    candidates = build_candidates(matches).drop(columns=["tier"])
    frame = frame.merge(candidates, on="match_id", how="left")
    frame["tier"] = frame["tier"].astype("Float64")
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def per_match_loss(frame: pd.DataFrame, columns: Sequence[str], seed: int) -> np.ndarray:
    """One log loss term per evaluated match, pooled over the folds.

    Pooled by match rather than by fold, for the reason
    :func:`~src.models.tuning.walk_forward_log_loss` gives: log loss is a mean,
    and averaging fold averages lets a short fold count as much as a long one.
    """
    zoo.SEED = seed
    parts = [
        terms(
            zoo.build(MODEL, columns).forecast(fold.train, fold.evaluate), fold.evaluate["result"]
        )["log_loss"].to_numpy()
        for fold in walk_forward(frame)
    ]
    return np.concatenate(parts)


def paired(frame: pd.DataFrame) -> None:
    """The headline: thirty columns against forty-four, matched per fixture."""
    base = per_match_loss(frame, DESIGN_COLUMNS, SEEDS[0])
    extended = per_match_loss(frame, (*DESIGN_COLUMNS, *CANDIDATE_COLUMNS), SEEDS[0])
    delta = base - extended
    result = stats.ttest_rel(base, extended)
    interval = 1.96 * stats.sem(delta)
    print(f"\nbase {len(DESIGN_COLUMNS)} columns      {base.mean():.5f}")
    print(f"plus {len(CANDIDATE_COLUMNS)} candidates   {extended.mean():.5f}")
    print(f"delta                 {delta.mean():+.6f}   (positive favours the candidates)")
    print(f"paired t-test         t={result.statistic:.3f}  p={result.pvalue:.4f}  n={len(base):,}")
    print(f"95% CI on delta       {delta.mean() - interval:+.6f} .. {delta.mean() + interval:+.6f}")


def seeds(frame: pd.DataFrame) -> None:
    """The control. An effect smaller than the seed spread is not an effect."""
    extended = (*DESIGN_COLUMNS, *CANDIDATE_COLUMNS)
    base_means, extended_means = [], []
    for seed in SEEDS:
        base_means.append(per_match_loss(frame, DESIGN_COLUMNS, seed).mean())
        extended_means.append(per_match_loss(frame, extended, seed).mean())
        print(f"  seed {seed:>8}: base {base_means[-1]:.5f}  extended {extended_means[-1]:.5f}")
    base, extend = np.array(base_means), np.array(extended_means)
    print(f"\nbase spread across seeds      {base.max() - base.min():.5f}  <- the noise floor")
    print(f"extended spread across seeds  {extend.max() - extend.min():.5f}")
    print(f"improvement, per seed         {np.round(base - extend, 6)}")
    print(f"improvement, mean             {(base - extend).mean():+.6f}")


def blocks(frame: pd.DataFrame) -> None:
    """Which group carries the gain, one block at a time."""
    base = per_match_loss(frame, DESIGN_COLUMNS, SEEDS[0])
    print(f"\nbase {len(DESIGN_COLUMNS)} columns{'':22s}{base.mean():.5f}")
    for name, columns in BLOCKS.items():
        scored = per_match_loss(frame, (*DESIGN_COLUMNS, *columns), SEEDS[0])
        delta = base - scored
        result = stats.ttest_rel(base, scored)
        print(
            f"+{name:<34s}{scored.mean():.5f}  "
            f"delta={delta.mean():+.6f}  p={result.pvalue:.4f}  ({len(columns)} cols)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", action="store_true", help="the headline comparison")
    parser.add_argument("--seeds", action="store_true", help="the seed-noise control")
    parser.add_argument("--blocks", action="store_true", help="per-block decomposition")
    args = parser.parse_args()
    chosen = [name for name in ("paired", "seeds", "blocks") if getattr(args, name)] or [
        "paired",
        "seeds",
        "blocks",
    ]

    started = time.perf_counter()
    frame = modelling_frame()
    logger.info("frame: %d matches, %d columns", len(frame), frame.shape[1])
    for name in chosen:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        globals()[name](frame)
    print(f"\ntotal {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
