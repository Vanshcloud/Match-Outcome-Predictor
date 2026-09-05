"""The three tables, joined, the way every modelling command needs them.

The canonical matches, the ratings and the features live in three files written
by three pipelines, and a model needs all thirty columns of the join. Three
commands now do this — train, explain, and the model card — and the version
that lived in one of them was on its way to being copied into the other two,
which is how two commands end up training on twenty-five columns and one on
thirty without anyone noticing.

**Left joins, deliberately.** A match the ratings or the features have nothing
to say about keeps its row and loses those columns, which the models read as
the null it is — "no history yet" — rather than losing the match. An inner join
here would silently drop every club's first fixtures and improve every score.

**Through the store, never by opening a path.** The one rule Milestone 3 set
and CI enforces: nothing outside `src/storage` names a file format.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ingestion.base import MATCHES_FILENAME
from src.pipelines.features import FEATURES_FILENAME
from src.pipelines.ratings import RATINGS_FILENAME
from src.storage.duckdb_store import DuckDBStore
from src.utils.config import PathsConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

KEY_COLUMN = "match_id"

SCORES_VIEW = "scores"

FORECASTS_FILENAME = "forecasts.parquet"
"""The per-match diagnostic pass, beside the scores it summarises.

:mod:`src.pipelines.backtest` persists *means* — one row per fold,
competition, forecaster and subset — which is right for a report and cannot
answer whether a stated probability happens at the rate it states. The card
recomputes the rows to ask that, at about five minutes a run, and Milestone 12
needs the same rows on every page load. So the card writes them down.
"""


@dataclass(frozen=True, slots=True)
class TablePaths:
    """Where the three tables are. Named rather than positional, because two of
    them are Parquet files in the same directory with similar names."""

    matches: Path
    ratings: Path
    features: Path

    def missing(self) -> tuple[str, ...]:
        """The labels of the tables that are not on disk. Empty when all three
        are, which is the only state a model can be built from."""
        return tuple(
            label
            for label, path in (
                ("match table", self.matches),
                ("ratings", self.ratings),
                ("features", self.features),
            )
            if not path.is_file()
        )


def resolve_tables(
    paths: PathsConfig,
    *,
    matches: Path | None = None,
    ratings: Path | None = None,
    features: Path | None = None,
) -> TablePaths:
    """Where the three tables live, with any of them overridden by a caller.

    The override is what the ``--matches`` family of flags does on every
    command that reads them, and resolving it once here is the difference
    between three commands agreeing about the default and three commands that
    happen to.
    """
    return TablePaths(
        matches=matches or paths.processed_dir / MATCHES_FILENAME,
        ratings=ratings or paths.features_dir / RATINGS_FILENAME,
        features=features or paths.features_dir / FEATURES_FILENAME,
    )


def load_modelling_frame(
    paths: TablePaths, *, competitions: Sequence[str] | None = None
) -> pd.DataFrame | None:
    """The canonical table joined to the ratings and the features.

    Returns:
        The joined frame, or ``None`` when a table is absent or the filter
        selected nothing — both of which are a caller's problem to report, not
        an exception to raise. A clean checkout has none of the three, and a
        command that stack-traced there would be failing at the expected state.
    """
    absent = paths.missing()
    if absent:
        logger.error("no %s; a model needs all three tables", ", ".join(absent))
        return None

    with DuckDBStore.open_matches(
        paths.matches, ratings=paths.ratings, features=paths.features
    ) as store:
        matches = store.read_matches(competitions=list(competitions) if competitions else None)
        ratings = store.read_ratings()
        features = store.read_features()

    if matches.empty:
        logger.error("no matches for %s", ", ".join(competitions or ()) or paths.matches)
        return None
    return matches.merge(ratings, on=KEY_COLUMN, how="left").merge(
        features, on=KEY_COLUMN, how="left"
    )


def read_forecasts(path: Path) -> pd.DataFrame | None:
    """The persisted per-match forecasts, or ``None`` when they are absent.

    Absent is the clean-checkout state and the state before `make card` has
    run, so the dashboard reports what is missing and which command produces
    it rather than failing at the expected condition.
    """
    return read_scores(path)


def read_scores(path: Path) -> pd.DataFrame | None:
    """A persisted score table, read through the store like everything else.

    The backtest writes Parquet and `src/storage` is the only package allowed
    to know that. A report that opened the file directly would work today and
    be the one caller left behind the day the tables move.

    Returns:
        The scores, or ``None`` when the file is not there — which is the state
        of a clean checkout and the state before the run that produces it, not
        an error worth a stack trace.
    """
    if not path.is_file():
        logger.info("no scores at %s", path)
        return None
    with DuckDBStore() as store:
        store.attach_parquet(SCORES_VIEW, path)
        return store.query(f"SELECT * FROM {SCORES_VIEW}")


RESULT_COLUMNS: tuple[str, ...] = (
    KEY_COLUMN,
    "competition_id",
    "competition",
    "country",
    "season",
    "date",
    "kickoff",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
)
"""What a finished match is *displayed* by, and nothing else.

A reader's columns: who played, when, and how it ended. Not the thirty design
columns — a page that held those would be one keystroke from pricing a fixture
itself — and not the three odds columns, which are the benchmark this project
measures itself against and do not belong beside a forecast.
"""


def read_matches(
    path: Path,
    *,
    competitions: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    columns: Sequence[str] | None = RESULT_COLUMNS,
) -> pd.DataFrame | None:
    """Finished matches, projected and filtered, or ``None`` when there are none.

    The read behind every result, head-to-head and form table a reader sees.
    It exists because those are *results* — the thing the provider actually
    publishes — and the service deliberately does not serve them: a prediction
    endpoint that returned the scoreline beside its forecast would be answering
    a different question.

    Filters are the store's own and are pushed into the query rather than
    applied afterwards, so "the last fortnight in the Premier League" does not
    first pull three hundred thousand rows into memory.

    Returns:
        The matching rows, or ``None`` when the table is not on disk — the
        clean-checkout state, which is a caller's to report rather than an
        exception to raise.
    """
    if not path.is_file():
        logger.info("no match table at %s", path)
        return None
    with DuckDBStore.open_matches(path) as store:
        return store.read_matches(
            competitions=list(competitions) if competitions is not None else None,
            since=since,
            until=until,
            columns=list(columns) if columns is not None else None,
        )
