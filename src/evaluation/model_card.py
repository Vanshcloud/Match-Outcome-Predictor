"""The model card: what a model was trained on, what it scores, where it fails.

The sibling of `src/validation/card.py`, and written for the same reason. A
model card that is typed by hand is a model card whose numbers are from
whichever run the author happened to have open, and the ones that go stale
first are exactly the ones a reader needs — the coverage, the worst
competition, the date the training data stops.

So every figure here arrives as data. This module renders; it does not measure,
does not open a file, and does not know that DuckDB, Parquet or a backtest
exist. That is what keeps `src/evaluation` a package of arithmetic over arrays,
and it is why the card can be built in a test from four small frames.

**What it must not be used for is a section, not a disclaimer.** A card that
lists metrics and stops is a scoreboard. The limitations below are the part
that changes what someone does with the model, and each one is a measured fact
from this project rather than boilerplate: the bookmaker wins in all 39
competitions, the model has no opinion on a club with no history, and the
probabilities are honest to about 0.002 rather than exactly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.evaluation.reliability import expected_calibration_error

CARD_FILENAME = "MODEL_CARD.md"

LIMITATIONS: tuple[str, ...] = (
    "- **Betting.** The bookmaker's closing line beats this model in every",
    "  one of the competitions it was scored on, before any margin is taken",
    "  off. A model that lost to the market on public data and was staked",
    "  anyway would lose the margin as well as the gap.",
    "- **A club with no history in the table.** A promoted side, a first",
    "  season, a competition's first year: the form windows are null and the",
    "  ratings are at their priors, and the forecast is close to the base",
    "  rate. That is the correct answer and a weak one.",
    "- **Anything that needs the reason.** These probabilities are ranked,",
    "  not explained. `docs/EXPLAINABILITY.md` says which blocks the model",
    "  leans on across many matches; none of that is a claim about why one",
    "  fixture came out the way it did.",
    "- **Live or in-play prediction.** Every input is knowable before",
    "  kick-off by construction, and nothing here updates during a match.",
)
"""The section that changes what someone does with the model, as lines.

A constant rather than a literal inside :meth:`ModelCard.render`, because
the API serves these four bullets — the card's limitations should be reachable from a response rather than buried in a
repository, and the only way that stays true is for the served text and the
rendered text to be the same object. Wrapped at the width the card is written
at, so splicing them into the document needs no reflow.
"""


def _table(frame: pd.DataFrame, *, decimals: int = 4) -> list[str]:
    """A frame as GitHub-flavoured markdown rows.

    Written out rather than taken from ``DataFrame.to_markdown``, which needs
    tabulate — a dependency for one function, in a file whose output is four
    tables.
    """
    if frame.empty:
        return ["_(nothing to report)_"]
    header = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in frame.itertuples(index=False):
        rendered = [
            (
                f"{value:,.{decimals}f}"
                if isinstance(value, float)
                else f"{value:,}" if isinstance(value, int) else str(value)
            )
            for value in row
        ]
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Everything the card states, as data, so the rendering has no opinions.

    Built by :mod:`src.pipelines.report` from the persisted score tables and a
    diagnostic pass; kept separate from that assembly so this file can be
    tested with four hand-written frames and no data on disk.
    """

    name: str
    """The shipped model, by the name it is scored under."""

    version: str
    settings: Mapping[str, Any]
    """What the model was configured with, for the run log's own reasons."""

    scores: pd.DataFrame
    """The pooled table: one row per forecaster, the model and its baselines."""

    reliability: pd.DataFrame
    """Bins of stated probability against how often the thing happened."""

    per_class: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    """The same, split by H, D and A."""

    worst: pd.DataFrame = field(default_factory=pd.DataFrame)
    """The competitions where the model is least reliable."""

    matches: int = 0
    competitions: int = 0
    folds: int = 0
    trained_from: str = ""
    trained_to: str = ""
    columns: Sequence[str] = ()

    def _headline(self) -> pd.DataFrame:
        """The model's own row, and the two rows it is read against."""
        if self.scores.empty:
            return self.scores
        keep = {self.name, "bookmaker", "dixon_coles", "class_prior"}
        return self.scores[self.scores["forecaster"].isin(keep)]

    def render(self) -> str:
        """The card, as Markdown."""
        error = (
            expected_calibration_error(self.reliability)
            if not self.reliability.empty
            else float("nan")
        )
        lines: list[str] = [
            f"# Model card — {self.name}",
            "",
            "<!-- GENERATED by scripts/model_card.py. Do not edit by hand: the next",
            "     run overwrites it, and a hand-edited figure is a figure that no",
            "     longer describes the model. -->",
            "",
            f"match-outcome-predictor {self.version}.",
            "",
            "## What it does",
            "",
            "Turns one football fixture into three probabilities — home win, draw,",
            "away win — that sum to one. It is a probabilistic forecaster, not a",
            "tipster: the output is the *distribution*, and the most likely class is",
            "the least interesting thing in it.",
            "",
            "## What it was trained on",
            "",
            "| | |",
            "|---|---|",
            f"| Matches scored | {self.matches:,} |",
            f"| Competitions | {self.competitions} |",
            f"| Walk-forward folds | {self.folds} |",
            f"| History | {self.trained_from} to {self.trained_to} |",
            f"| Input columns | {len(self.columns)} |",
            "",
            "Every fold trains on matches strictly earlier than the ones it is scored",
            "on, and the boundary is a date rather than a row — two matches played on",
            "the same afternoon are never split across it. The inputs are the twenty",
            "features and ten rating columns from the ratings table. **The",
            "bookmaker's odds are not among them**: they are the benchmark this",
            "project measures itself against, and a model given the closing line",
            "learns to copy it.",
            "",
            "## What it scores",
            "",
            *_table(self._headline()),
            "",
            "Log loss first, RPS second, accuracy last and close to meaningless — a",
            "well-calibrated model that rarely *predicts* a draw is behaving",
            "correctly, since roughly a quarter of matches are drawn and almost none",
            "of them are the modal outcome beforehand.",
            "",
            "## Where its probabilities are honest",
            "",
            f"Calibration error, pooled over every statement: **{error:.4f}**. That is",
            "the mean gap between a stated probability and how often the thing",
            "happened, weighted by how many statements sit behind each bin.",
            "",
            *_table(self.reliability),
            "",
        ]

        if self.per_class:
            lines += [
                "### By class",
                "",
                "Pooling the three answers whether the model is honest; splitting",
                "them says which class it is dishonest about.",
                "",
                "| Class | statements | calibration error |",
                "|---|---:|---:|",
            ]
            lines += [
                f"| {label} | {int(table['n'].sum()):,} | "
                f"{expected_calibration_error(table):.4f} |"
                for label, table in self.per_class.items()
                if not table.empty
            ]
            lines.append("")

        if not self.worst.empty:
            lines += [
                "### Where it is least reliable",
                "",
                "Aggregate honesty hides the competitions that pay for it.",
                "",
                *_table(self.worst),
                "",
            ]

        lines += [
            "## What it must not be used for",
            "",
            *LIMITATIONS,
            "",
            "## How it was built",
            "",
            "| | |",
            "|---|---|",
        ]
        lines += [f"| `{key}` | {value} |" for key, value in sorted(self.settings.items())]
        lines += [
            "",
            "Hyperparameters were searched on matches strictly earlier than the first",
            "reported fold, and are version-controlled constants rather than a file",
            "beside the data. Full method in [MODELS.md](MODELS.md), the leakage",
            "argument in [LEAKAGE.md](LEAKAGE.md), the data in",
            "[DATASET_CARD.md](DATASET_CARD.md).",
            "",
        ]
        return "\n".join(lines)
