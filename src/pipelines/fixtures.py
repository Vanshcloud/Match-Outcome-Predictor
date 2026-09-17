"""Design rows for matches that have not been played.

:mod:`src.ingestion.fixtures` gets the fixture list; this turns each fixture
into the thirty columns the model prices, and the entire design is one sentence:

**The same builders, over the same history, with the fixtures on the end.**

Nothing here computes a feature. :func:`~src.pipelines.ratings.build_ratings`
and :func:`~src.pipelines.features.build_features` are handed the canonical
table with the fixture rows appended, and the rows they produce for those
fixtures are kept. The alternative — a function that computes a team's form
from its last five matches for a fixture — is a second implementation of
the feature layer, living outside every leakage probe that guards the first,
and the day the two disagree the service is quietly pricing on different
arithmetic than the backtest was scored on. ``src/pipelines/serving.py`` refused
to rebuild a design row inside a request handler for exactly this reason; this
module refuses for the same one, and pays a full rebuild of the derived tables
to do it.

**Why appending is safe, and how that is checked rather than argued.** Every
producer in this project is causal: the row emitted for a match is built from
matches strictly before it. Fixtures are dated after every match in the table,
so appending them cannot change a single historical row — and the claim is not
left as reasoning. :func:`design_rows` is the thing the holdout test drives:
take the last day of the canonical table, blank its scorelines, run it back
through here as though it were a fixture list, and the design rows that come out
are the ones `make features` already wrote for those matches.

**What it costs is the whole history, every time.** Measured on the real table:
**676 seconds for the ratings and 6 for the features**, to price three
fixtures. Elo is a fold over 303,517 matches and Dixon-Coles refits along the
sequence, so nearly all of that is re-deriving rows that have not changed since
yesterday. An incremental rater resuming from persisted state would fix it and
would be a second code path through the one arithmetic the leakage suite
guards, so the slow version is the one that ships. If eleven minutes a day ever
matters, the thing to add is a cached rating *state* — not a second rater.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.feature_engineering.registry import FeatureBuilder
from src.pipelines.derived import persist
from src.pipelines.features import build_features
from src.pipelines.ratings import build_ratings
from src.pipelines.serving import SERVED_COLUMNS
from src.pipelines.tables import KEY_COLUMN, UPCOMING_FILENAME
from src.ratings.base import RatingModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class UpcomingReport:
    """What one fixture build produced."""

    fixtures: int = 0
    already_played: int = 0
    """Fixtures dropped because the canonical table already has them.

    The published file keeps a match in it for a while after it is played, and
    `make data` may well have caught up with it in the meantime. Two rows with
    one ``match_id`` would give the index a duplicate key, so the played row —
    the real one — wins and this counts what it displaced."""

    competitions: tuple[str, ...] = ()
    complete: int = 0
    """Fixtures whose design row has no nulls in it.

    Reported rather than enforced. A promoted club playing its first match in a
    division has no history in it and gets a partial row, which every
    gradient-boosted member reads as the "no history yet" it is — the same
    thing that happens to that club's first *played* match. It is worth
    counting because a sudden collapse in it means a name stopped matching,
    not that football changed."""

    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    output: Path | None = None
    coverage: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.fixtures:
            return "no fixtures"
        window = f"{self.first:%Y-%m-%d} to {self.last:%Y-%m-%d}"
        return (
            f"{self.fixtures} fixture(s) over {len(self.competitions)} competition(s), "
            f"{window}; {self.complete} with a complete design row"
        )


def design_rows(
    matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    models: tuple[RatingModel, ...] | None = None,
    builders: tuple[FeatureBuilder, ...] | None = None,
) -> pd.DataFrame:
    """The served columns for every fixture, built by the producers themselves.

    Args:
        matches: The canonical table. Read, never modified — the fixtures are
            appended to a copy that is thrown away with the derived tables it
            produced.
        fixtures: Canonical-shaped rows with no result, from
            :func:`src.ingestion.fixtures.to_frame`.

    Returns:
        One row per fixture with :data:`~src.pipelines.serving.SERVED_COLUMNS`,
        in date order. Empty — with the columns still present — when there is
        nothing to price, which is the ordinary state on a Wednesday in June.
    """
    empty = pd.DataFrame(columns=list(SERVED_COLUMNS))
    if fixtures.empty:
        return empty

    wanted = fixtures[KEY_COLUMN]
    # Typed to the match table before the concatenation rather than after it. A
    # fixture's result columns are entirely null, and pandas decides the dtype
    # of a concatenated column from the frames it is given: an all-null block
    # arriving as `object` makes the join downstream produce object columns
    # that `design_matrix` cannot convert, with an error naming neither the
    # column nor this project.
    shared = [column for column in fixtures.columns if column in matches.columns]
    aligned = fixtures.astype({column: matches[column].dtype for column in shared})
    frame = pd.concat([matches, aligned], ignore_index=True).sort_values(
        "date", kind="stable", ignore_index=True
    )
    ratings = build_ratings(frame, models)
    features = build_features(frame, builders)
    joined = frame.merge(ratings, on=KEY_COLUMN, how="left").merge(
        features, on=KEY_COLUMN, how="left"
    )

    rows = joined[joined[KEY_COLUMN].isin(set(wanted))]
    missing = [column for column in SERVED_COLUMNS if column not in rows.columns]
    if missing:
        # The same failure `load_index` reports, caught one step earlier: it
        # means a producer stopped filling a column the artefact was fitted on,
        # and a fixture table written without it would be a 500 per request
        # instead of one message here.
        raise ValueError(f"the build produced no {missing[:5]}")
    return rows[list(SERVED_COLUMNS)].reset_index(drop=True)


def run_upcoming(
    matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    features_dir: Path,
    *,
    models: tuple[RatingModel, ...] | None = None,
    builders: tuple[FeatureBuilder, ...] | None = None,
    sources: Sequence[Path] = (),
) -> UpcomingReport:
    """Build the fixture design rows, report them, and write the table.

    Written even when it is empty, and that is deliberate: an empty file means
    "there are no fixtures", and leaving yesterday's file in place would mean
    the service went on offering to price matches that have since kicked off.

    ``sources`` are the tables the design rows were built from; their checksums
    go into the manifest's ``inputs``, which is what says whether the form
    windows behind a fixture came from the current match table.
    """
    known = set(matches[KEY_COLUMN]) if not matches.empty else set()
    fresh = fixtures[~fixtures[KEY_COLUMN].isin(known)] if not fixtures.empty else fixtures
    report = UpcomingReport(
        fixtures=len(fresh),
        already_played=len(fixtures) - len(fresh),
        competitions=tuple(sorted(fresh["competition_id"].unique())) if len(fresh) else (),
    )
    if len(fresh):
        report.first, report.last = fresh["date"].min(), fresh["date"].max()

    rows = design_rows(matches, fresh, models=models, builders=builders)
    if len(rows):
        design = rows.drop(columns=[column for column in rows.columns if column in fresh.columns])
        report.complete = int(design.notna().all(axis=1).sum())
        report.coverage = {column: float(rows[column].notna().mean()) for column in design.columns}

    report.output = persist(
        rows,
        features_dir / UPCOMING_FILENAME,
        extra={
            "kind": "upcoming",
            "fixtures": report.fixtures,
            "competitions": list(report.competitions),
            "complete": report.complete,
            "first": None if report.first is None else report.first.strftime("%Y-%m-%d"),
            "last": None if report.last is None else report.last.strftime("%Y-%m-%d"),
            "coverage": report.coverage,
        },
        sources=sources,
    )
    logger.info("wrote %s — %s", report.output.name, report.summary())
    return report
