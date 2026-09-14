#!/usr/bin/env python
"""Which feature blocks a model actually uses, by two methods and a third for free.

    python scripts/explain.py                     # lightgbm, the family the ablation used
    python scripts/explain.py --model xgboost --model mlp
    python scripts/explain.py --markdown          # the tables in docs/

Two measurements, per block, over the most recent fold:

  permutation  break the block at prediction time and see what the log loss
               loses. Model-agnostic, so every family gets a number, and it is
               in the same units as the ablation.
  shap         decompose the model's own arithmetic. Exact and fast for the
               three boosted families; the other three are pipelines that
               `TreeExplainer` refuses, and permutation already covers them.

Where a persisted ablation for the same family exists it is joined on, so all
three columns can be read at once. They answer different questions — "does this
model use the block", "how much of its output does the block move", and "would
a model built without it be worse" — and the places they disagree are the
interesting rows, not the broken ones.

One fold, not five: a fit and thirty-one predictions rather than five of each,
on the most recent twelve thousand matches. The output is a ranking whose gaps
are an order of magnitude apart, and a second fold does not move it.

Exits non-zero if the tables cannot be read or no fold can be built.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.explainability.permutation import DEFAULT_REPEATS, importance  # noqa: E402
from src.explainability.shapley import (  # noqa: E402
    DEFAULT_SAMPLE,
    ExplainError,
    attribution,
)
from src.models.splits import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_HORIZON_DAYS,
    SplitError,
    walk_forward,
)
from src.models.zoo import FAMILIES, ModelError, build  # noqa: E402
from src.pipelines.backtest import BACKTEST_FILENAME  # noqa: E402
from src.pipelines.tables import (  # noqa: E402
    load_modelling_frame,
    read_scores,
    resolve_tables,
)
from src.pipelines.train import ABLATION_SUBDIR, ALL_BLOCKS, ablation_table  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("explain")

DEFAULT_MODEL = "lightgbm"
"""The family `make ablation` runs, so the third column is populated by default."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matches", type=Path, default=None, help="canonical table to read")
    parser.add_argument("--ratings", type=Path, default=None, help="ratings table to join")
    parser.add_argument("--features", type=Path, default=None, help="feature table to join")
    parser.add_argument("--ablation", type=Path, default=None, help="ablation scores to compare")
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
        help=f"model family. Repeatable. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="walk-forward steps")
    parser.add_argument(
        "--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS, help="days scored per fold"
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="shuffles per block")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="matches to explain")
    parser.add_argument("--markdown", action="store_true", help="print the tables as markdown")
    parser.add_argument("--log-level", default=None, help="override the configured level")
    return parser.parse_args(argv)


def render(table: pd.DataFrame, *, markdown: bool) -> str:
    """A table as text or markdown. Deltas keep their sign; shares do not."""
    if table.empty:
        return "  (nothing to report)"
    if not markdown:
        return table.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    header = [str(column) for column in table.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in table.itertuples(index=False):
        rendered = [f"{value:+.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def ablation_for(path: Path) -> tuple[str, pd.DataFrame] | None:
    """The persisted ablation and the family it was run on, or None.

    The family is read back off the control row's name rather than assumed:
    `make ablation` runs one model, someone may have run another, and joining a
    LightGBM ablation onto an XGBoost attribution would put three numbers in a
    row that are not about the same model.
    """
    scores = read_scores(path)
    if scores is None:
        logger.info("run `make ablation` for the third column")
        return None
    control = [
        str(name) for name in scores["forecaster"].unique() if str(name).endswith(f"-{ALL_BLOCKS}")
    ]
    if not control:
        logger.warning("no control row in %s; it is not an ablation", path)
        return None
    model = control[0].removesuffix(f"-{ALL_BLOCKS}")
    return model, ablation_table(scores, model)


def explain(
    frame: pd.DataFrame,
    name: str,
    args: argparse.Namespace,
    ablated: tuple[str, pd.DataFrame] | None,
) -> None:
    """Print both attributions for one family, and the ablation if it fits."""
    fold = list(walk_forward(frame, folds=args.folds, horizon_days=args.horizon_days))[-1]
    model = build(name)
    print(f"\n## {name} — {fold.summary()}\n")

    broken = importance(model, fold.train, fold.evaluate, repeats=args.repeats)
    if ablated is not None and ablated[0] == name:
        broken = broken.merge(
            ablated[1][["block", "delta_log_loss"]].rename(columns={"delta_log_loss": "ablation"}),
            on="block",
            how="left",
        )
    print("What breaking each block costs, in log loss:")
    print(render(broken, markdown=args.markdown))

    try:
        print("\nWhat each block contributes to the model's own arithmetic:")
        print(
            render(
                attribution(model, fold.train, fold.evaluate, sample=args.sample),
                markdown=args.markdown,
            )
        )
    except ExplainError as error:
        print(f"  (no SHAP for {name}: {error})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(level=args.log_level or settings.logging.level, fmt=settings.logging.format)

    tables = resolve_tables(
        settings.paths, matches=args.matches, ratings=args.ratings, features=args.features
    )
    frame = load_modelling_frame(tables, competitions=args.competition or None)
    if frame is None:
        return 1

    ablated = ablation_for(
        args.ablation or settings.paths.reports_dir / ABLATION_SUBDIR / BACKTEST_FILENAME
    )
    if ablated is not None:
        print(f"comparing against the persisted ablation of {ablated[0]}")
    try:
        for name in args.model or [DEFAULT_MODEL]:
            explain(frame, name, args, ablated)
    except (SplitError, ModelError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
