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

from src.pipelines.features import FEATURES_FILENAME
from src.pipelines.ingest import MATCHES_FILENAME
from src.pipelines.ratings import RATINGS_FILENAME
from src.storage.duckdb_store import DuckDBStore
from src.utils.config import PathsConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

KEY_COLUMN = "match_id"

SCORES_VIEW = "scores"


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
