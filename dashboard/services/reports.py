"""Reading the measurements, and saying plainly when they are not there.

Every table on this dashboard was produced by a pipeline command. None is
recomputed here: :mod:`src.pipelines.report` and :mod:`src.pipelines.backtest`
own that arithmetic, the model card is generated from the same functions, and a
dashboard with its own copy would be a second number to reconcile.

**A missing table is a state, not a crash.** A clean checkout has no reports,
which is exactly what a new reader has. Every loader here returns ``None`` and
every panel renders the command that produces what is missing — the same
decision `/health` makes in the API, for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation.metrics import CLASSES
from src.evaluation.reliability import expected_calibration_error, reliability
from src.models.ensemble import FORECAST_COLUMNS, SHIPPED
from src.pipelines.backtest import BACKTEST_FILENAME, per_competition_table, pooled_table
from src.pipelines.report import reliability_by_class, reliability_by_competition
from src.pipelines.tables import FORECASTS_FILENAME, read_forecasts, read_scores
from src.utils.logging import get_logger

logger = get_logger(__name__)

ENSEMBLE_SUBDIR = "ensemble"
"""Which backtest the dashboard reads.

The ensemble run, because it holds the shipped model beside the bookmaker and
the baselines — the comparison the whole project is arranged around. The zoo
run scores six families against each other and is a modelling question rather
than a reader's one.
"""

BUILD_COMMAND = "make reproduce && make card"
"""What produces everything this page needs, named in every empty state."""


@dataclass(frozen=True, slots=True)
class Reports:
    """The two tables the dashboard reads, and where they came from.

    Both optional and independently so: `make ensemble` writes the scores and
    `make card` writes the forecasts, so a reader can easily have one and not
    the other, and a page that demanded both would go blank over the half it
    did not need.
    """

    scores: pd.DataFrame | None
    forecasts: pd.DataFrame | None
    reports_dir: Path

    @property
    def has_scores(self) -> bool:
        return self.scores is not None and not self.scores.empty

    @property
    def has_forecasts(self) -> bool:
        return self.forecasts is not None and not self.forecasts.empty

    def missing(self) -> tuple[str, ...]:
        """What is absent, named the way a reader would go looking for it."""
        absent = []
        if not self.has_scores:
            absent.append(f"{ENSEMBLE_SUBDIR}/{BACKTEST_FILENAME} (run `make ensemble`)")
        if not self.has_forecasts:
            absent.append(f"{ENSEMBLE_SUBDIR}/{FORECASTS_FILENAME} (run `make card`)")
        return tuple(absent)


def load_reports(reports_dir: Path) -> Reports:
    """Both report tables, each ``None`` when its file is not there.

    Takes the reports directory rather than the whole ``PathsConfig``: this is
    the only thing it needs, and it is what lets the dashboard's cache be keyed
    on a plain string that Streamlit can hash.
    """
    directory = reports_dir / ENSEMBLE_SUBDIR
    scores = read_scores(directory / BACKTEST_FILENAME)
    forecasts = read_forecasts(directory / FORECASTS_FILENAME)
    logger.info(
        "reports: scores=%s forecasts=%s",
        "yes" if scores is not None else "no",
        f"{len(forecasts):,} rows" if forecasts is not None else "no",
    )
    return Reports(scores=scores, forecasts=forecasts, reports_dir=directory)


@st.cache_data(show_spinner="reading the reports…")
def _cached_reports(reports_dir: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Both report tables, cached on the directory they came from.

    Keyed by a plain string because Streamlit hashes a function's arguments to
    decide whether the cache is still valid, and a ``Path`` or a pydantic model
    is not something it can hash cheaply. Returns the two frames rather than
    the :class:`Reports` that holds them, for the same reason: what goes into
    the cache should be what pandas already knows how to store.
    """
    loaded = load_reports(Path(reports_dir))
    return loaded.scores, loaded.forecasts


def reports(reports_dir: Path) -> Reports:
    """The reports, through the cache. What every view calls.

    Streamlit reruns the whole script on every interaction and four views read
    these tables, so a reader dragging a filter must not re-read a
    two-megabyte Parquet each time.
    """
    scores, forecasts = _cached_reports(str(reports_dir))
    return Reports(scores=scores, forecasts=forecasts, reports_dir=reports_dir / ENSEMBLE_SUBDIR)


# ---- the tables a panel renders ----------------------------------------------


def leaderboard(scores: pd.DataFrame) -> pd.DataFrame:
    """Every forecaster over the matches all of them could price.

    :func:`~src.pipelines.backtest.pooled_table` unchanged — the common subset,
    which is the only one on which two forecasters' numbers answer the same
    question.
    """
    return pooled_table(scores)


def by_competition(scores: pd.DataFrame, metric: str = "log_loss") -> pd.DataFrame:
    """One row per competition, one column per forecaster."""
    return per_competition_table(scores, metric=metric)


def filtered_forecasts(
    forecasts: pd.DataFrame,
    *,
    competitions: list[str] | None = None,
    folds: list[int] | None = None,
    name: str = SHIPPED,
) -> pd.DataFrame:
    """The shipped model's rows, narrowed to what the reader asked for.

    Filtering *before* the reliability table is computed is the point of the
    panel: the card reports one pooled figure and the interesting question is
    which competitions and which years it is an average over.
    """
    rows = forecasts[forecasts["forecaster"] == name]
    if competitions:
        rows = rows[rows["competition_id"].isin(competitions)]
    if folds:
        rows = rows[rows["fold"].isin(folds)]
    return rows.dropna(subset=list(FORECAST_COLUMNS))


def reliability_table(forecasts: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
    """The reliability table for whatever rows it is handed.

    :func:`~src.evaluation.reliability.reliability` unchanged, over a subset —
    which is the whole feature. Empty in, empty out: a filter that selects
    nothing is a reader's question with the answer "no matches", not an error.
    """
    if forecasts.empty:
        return pd.DataFrame(columns=["bin", "lower", "upper", "n", "predicted", "observed", "gap"])
    return reliability(
        forecasts[list(FORECAST_COLUMNS)].to_numpy(dtype=float),
        forecasts["result"],
        bins=bins,
    )


def calibration_error(table: pd.DataFrame) -> float:
    """The reliability table as one number. ``nan`` when there is nothing in it."""
    if table.empty:
        return float("nan")
    return expected_calibration_error(table)


def per_class_tables(forecasts: pd.DataFrame, *, bins: int = 10) -> dict[str, pd.DataFrame]:
    """One reliability table per outcome, over the same matches."""
    if forecasts.empty:
        return dict.fromkeys(CLASSES, pd.DataFrame())
    return reliability_by_class(forecasts, bins=bins)


def worst_competitions(forecasts: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
    """Where the model's stated probabilities are least honest, worst first."""
    if forecasts.empty:
        return pd.DataFrame(columns=["competition_id", "matches", "calibration_error"])
    return reliability_by_competition(forecasts, bins=bins)
