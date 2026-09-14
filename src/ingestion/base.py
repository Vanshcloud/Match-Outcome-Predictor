"""The canonical match schema and the provider contract.

Every adapter in this package produces a frame with exactly these columns, in
this order, with these dtypes. That uniformity is the whole point: features,
storage and evaluation are written once against this schema, and a new provider
is a new adapter rather than a new code path in six other modules.

Two ideas carry most of the weight:

**Core versus optional.** A 1993 English match record has a date, two teams and
a score. A 2024 one adds shots, corners, cards and a referee; a Brazilian one
from the secondary feed has none of those, ever. Rather than pretend otherwise,
the schema marks a small set of columns as *core* (never null, validated) and
the rest as *optional* (nullable by design). Competitions declare what they can
supply via :class:`Capability`, so a feature can ask "is this available here?"
instead of discovering a column of nulls at training time.

**Nullable integer dtypes.** ``home_shots`` is ``Int64``, not ``int64``. NumPy's
integer type has no null, so a single missing value silently promotes the whole
column to float and 13 shots becomes ``13.0``. pandas' nullable ``Int64`` keeps
integers integral and missing values missing, which matters because "no shot
data for this competition" and "zero shots" are different facts.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    from src.ingestion.registry import Competition


class Capability(StrEnum):
    """What a competition's feed is able to supply.

    Declared per competition in ``configs/leagues.yaml`` and checked by feature
    code before it asks for a column. A capability is about the *feed*, not
    about one match: a competition either reports shots or it does not.
    """

    RESULTS = "results"
    """Date, teams, full-time score. Every competition has this or is unusable."""

    HALF_TIME = "half_time"
    """Half-time score. Absent from the earliest English seasons and from the
    entire secondary feed."""

    MATCH_STATS = "match_stats"
    """Shots, shots on target, corners, fouls, cards. The single biggest
    difference between the two football-data.co.uk feeds."""

    REFEREE = "referee"
    """Named official. Present in the primary feed from 2000/01."""

    ODDS = "odds"
    """Bookmaker prices. Carried for use as a *benchmark*, and deliberately not
    as a model feature by default — see the note on `odds_*` below."""


class Result(StrEnum):
    """Full-time outcome from the home team's perspective. The prediction target."""

    HOME = "H"
    DRAW = "D"
    AWAY = "A"


# --- The canonical schema ----------------------------------------------------
#
# Ordered deliberately: identity, then competition, then time, then teams, then
# outcome, then optional detail. A stable column order makes a Parquet schema
# diff readable and keeps `df.head()` legible in a notebook.

IDENTITY_COLUMNS: dict[str, str] = {
    "match_id": "string",
    "provider": "string",
}

COMPETITION_COLUMNS: dict[str, str] = {
    "competition_id": "string",
    "country": "string",
    "competition": "string",
    "tier": "Int8",
    "season": "string",
}

TIME_COLUMNS: dict[str, str] = {
    "date": "datetime64[ns]",
    "kickoff": "string",
}

TEAM_COLUMNS: dict[str, str] = {
    "home_team": "string",
    "away_team": "string",
    "home_team_id": "string",
    "away_team_id": "string",
}

OUTCOME_COLUMNS: dict[str, str] = {
    "home_goals": "Int16",
    "away_goals": "Int16",
    "result": "string",
}

HALF_TIME_COLUMNS: dict[str, str] = {
    "ht_home_goals": "Int16",
    "ht_away_goals": "Int16",
    "ht_result": "string",
}

MATCH_STAT_COLUMNS: dict[str, str] = {
    "home_shots": "Int16",
    "away_shots": "Int16",
    "home_shots_on_target": "Int16",
    "away_shots_on_target": "Int16",
    "home_corners": "Int16",
    "away_corners": "Int16",
    "home_fouls": "Int16",
    "away_fouls": "Int16",
    "home_yellows": "Int16",
    "away_yellows": "Int16",
    "home_reds": "Int16",
    "away_reds": "Int16",
}

OFFICIAL_COLUMNS: dict[str, str] = {
    "referee": "string",
}

# Closing decimal odds, market average where the feed provides one.
#
# These are NOT model features by default. A model trained on odds learns to
# copy the bookmaker, which both inflates its apparent skill and collapses the
# moment odds are unavailable — exactly the position a forecast should not be
# in. They are carried for two legitimate uses: as the benchmark this project
# measures itself against (the closing line is the strongest public forecast),
# and as an explicitly opt-in experimental feature block later.
ODDS_COLUMNS: dict[str, str] = {
    "odds_home": "Float64",
    "odds_draw": "Float64",
    "odds_away": "Float64",
}

CANONICAL_SCHEMA: dict[str, str] = {
    **IDENTITY_COLUMNS,
    **COMPETITION_COLUMNS,
    **TIME_COLUMNS,
    **TEAM_COLUMNS,
    **OUTCOME_COLUMNS,
    **HALF_TIME_COLUMNS,
    **MATCH_STAT_COLUMNS,
    **OFFICIAL_COLUMNS,
    **ODDS_COLUMNS,
}

CANONICAL_COLUMNS: tuple[str, ...] = tuple(CANONICAL_SCHEMA)

# Columns that must never be null in a row that survives ingestion. A row
# missing any of these is not a match this project can use, and dropping it
# loudly is better than carrying a hole into feature engineering.
CORE_COLUMNS: frozenset[str] = frozenset(
    {
        "match_id",
        "provider",
        "competition_id",
        "country",
        "competition",
        "season",
        "date",
        "home_team",
        "away_team",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
        "result",
    }
)

# Which capability gates which optional block. Used by the validation report to
# distinguish "this competition does not supply shots" from "the shot columns
# came back empty and something is wrong".
CAPABILITY_COLUMNS: dict[Capability, frozenset[str]] = {
    Capability.HALF_TIME: frozenset(HALF_TIME_COLUMNS),
    Capability.MATCH_STATS: frozenset(MATCH_STAT_COLUMNS),
    Capability.REFEREE: frozenset(OFFICIAL_COLUMNS),
    Capability.ODDS: frozenset(ODDS_COLUMNS),
}


# --- When each column becomes knowable ----------------------------------------
#
# The single most expensive mistake available in this project is training on a
# column that does not exist until after the match it is meant to predict.
# `home_shots` is not a property of the fixture; it is a summary of the ninety
# minutes. A model given it scores brilliantly in validation and is useless on
# a Saturday morning, because on a Saturday morning the column is empty.
#
# The canonical table stores those columns anyway — they are the raw material
# for *lagged* features, where a team's shots in its previous matches are
# entirely legitimate. The distinction is temporal, not columnar, so it cannot
# be enforced by leaving data out. It is enforced by naming which columns are
# knowable before kick-off, here, once, and by making the feature layer say
# which side of the line it is drawing from.
#
# `TARGET_COLUMN` and the score are post-match by construction. Everything in
# the odds block is genuinely pre-match — a closing price exists before kick-off
# — but is still not a default feature, for the separate reason recorded on
# ODDS_COLUMNS above.

PRE_MATCH_COLUMNS: frozenset[str] = frozenset(
    {
        *IDENTITY_COLUMNS,
        *COMPETITION_COLUMNS,
        *TIME_COLUMNS,
        *TEAM_COLUMNS,
        *ODDS_COLUMNS,
    }
)
"""Known before kick-off. Safe to use directly as a feature for the same match
— with the odds block excluded by default, see :data:`BENCHMARK_COLUMNS`."""

POST_MATCH_COLUMNS: frozenset[str] = frozenset(
    {
        *OUTCOME_COLUMNS,
        *HALF_TIME_COLUMNS,
        *MATCH_STAT_COLUMNS,
        *OFFICIAL_COLUMNS,
    }
)
"""Observed during or after the match. Usable only through a lag: a team's
value in an *earlier* match, never in this one.

``referee`` is the arguable member. Appointments are published days ahead in
most leagues, so in principle it is pre-match — but this provider supplies it
only in the results file, so the pipeline itself does not have it until the
match is over. Classified by what the data source can actually deliver at
prediction time rather than by what is true in principle; the conservative
direction is the one that cannot leak."""

BENCHMARK_COLUMNS: frozenset[str] = frozenset(ODDS_COLUMNS)
"""Pre-match, but reserved as the comparison this project measures itself
against rather than spent as an input. See the note on :data:`ODDS_COLUMNS`."""

TARGET_COLUMN = "result"
"""The prediction target: :class:`Result`, three ordered classes."""

MATCHES_FILENAME = "matches.parquet"
"""What the canonical table is called on disk.

Here rather than in :mod:`src.pipelines.ingest`, which writes it, because
several readers need the name and only one writer needs the pipeline. Importing
it from the pipeline meant importing the provider adapter, the registry, the
cache and — through them — ``requests``, which is how the serving
image once came to need an HTTP library to look up a string. This module is the
schema: the columns, the classes, and the name of the file they live in.

:mod:`src.pipelines.ingest` re-exports it, so the fifteen call sites that
already say ``from src.pipelines.ingest import MATCHES_FILENAME`` keep working
and keep meaning the same thing.
"""


def make_match_id(
    provider: str,
    competition_id: str,
    season: str,
    date: str,
    home_team: str,
    away_team: str,
) -> str:
    """Return a deterministic identifier for one match.

    Deterministic across processes and runs, which is what makes ingestion
    idempotent: re-fetching a season and re-writing it produces the same ids,
    so a storage upsert replaces rows instead of duplicating them. Python's
    built-in ``hash`` is unusable here — it is salted per process, so the same
    match would get a different id on every run.

    The inputs are the natural key. Two matches in the same competition, season
    and date between the same two teams do not exist; a competition that
    replays a fixture does so on a different date.

    ``blake2b`` at 8 bytes gives a 16-character hex id. With ~305k matches the
    collision probability is around 1 in 10^-10, and a collision would be
    caught by the duplicate check in validation rather than passing silently.
    """
    key = "|".join((provider, competition_id, season, date, home_team, away_team))
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()


class SeasonUnavailableError(LookupError):
    """A competition-season the provider does not hold.

    A normal outcome, not a failure: not every division has played every season
    since 1993, and the provider signals this three different ways — HTTP 404,
    HTTP 300 with an HTML body, or a country file that simply lacks the rows.
    One exception type spares callers from knowing which.
    """


@runtime_checkable
class MatchProvider(Protocol):
    """What every data source must implement.

    A Protocol rather than an abstract base class: adapters do not share
    implementation, only shape, and structural typing means a new provider does
    not have to import and inherit from this module to satisfy it.

    ``runtime_checkable`` so a registry can assert an object is a provider
    before calling it. Note that this only verifies the methods exist, which is
    all a registry needs.
    """

    @property
    def name(self) -> str:
        """Short provider identifier, written into every row's ``provider``."""
        ...

    def competitions(self) -> tuple[Competition, ...]:
        """Every competition this provider can supply."""
        ...

    def candidate_seasons(self, competition: Competition) -> tuple[str, ...]:
        """Canonical season labels worth *attempting* for ``competition``.

        Candidates, not guarantees. Some providers publish an index and can
        answer exactly; football-data.co.uk does not, and establishing the true
        answer for its primary feed would cost one request per division-season
        — over 700 before a single match is read. So a provider may return a
        plausible range, and :meth:`fetch` raises
        :class:`SeasonUnavailableError` for the ones that turn out not to exist.
        Callers treat that as a normal outcome, not an error.
        """
        ...

    def fetch(self, competition: Competition, season: str) -> pd.DataFrame:
        """Return one competition-season as a frame in :data:`CANONICAL_SCHEMA`.

        Implementations must return every canonical column, in order, with the
        declared dtypes — using nulls for anything the feed does not carry.
        Returning a subset is what forces downstream code to start asking which
        provider it is looking at, which is the coupling this package exists to
        prevent.
        """
        ...
