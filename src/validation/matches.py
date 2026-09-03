"""The checks that must hold of the canonical match table.

Four groups, in the order a problem is worth catching:

1. **Schema** — the columns and dtypes are what everything downstream compiles
   against, and the leakage classification covers every one of them.
2. **Integrity** — statements that are true of football. A result that
   disagrees with its own score, a team playing itself, a half-time score
   higher than full-time: each of these means a column was read from the wrong
   position, and each produces a model that trains perfectly and is wrong.
3. **Referential** — the table agrees with the registry it was built from.
4. **Distribution** — the shape of the data. These are the ones that catch the
   errors nothing else can see: home and away swapped somewhere in the adapter
   is invisible to every per-row check and obvious in the home-win rate.

Every threshold here is a measurement, not a guess, and the comment on each
says what was measured over the 305,499-row ingest of 2026-09-03. A bound
invented from intuition either never fires or fires constantly, and both teach
people to ignore the report.
"""

from __future__ import annotations

import re

import pandas as pd

from src.ingestion.base import (
    CANONICAL_COLUMNS,
    CANONICAL_SCHEMA,
    CORE_COLUMNS,
    POST_MATCH_COLUMNS,
    PRE_MATCH_COLUMNS,
    Capability,
    Result,
)
from src.ingestion.registry import Registry, load_registry
from src.ingestion.teams import find_single_season_teams
from src.validation.report import Check, Severity, check

# --- Measured thresholds -----------------------------------------------------

MAX_PLAUSIBLE_GOALS = 15
"""The record top-flight scoreline is in the low teens. Past that, a column
shifted. Measured maximum in the ingest: 13."""

HOME_WIN_RATE = (0.40, 0.52)
"""Home advantage in professional football is large and stable. Measured:
0.450. A rate near 0.33 would mean home and away had been swapped — the error
that produces a model which is confidently, exactly wrong."""

DRAW_RATE = (0.20, 0.32)
"""Measured: 0.267."""

MIN_OVERROUND = 0.99
"""A book's implied probabilities sum above 1; the excess is the margin.
Triples below 1.0 are provider typos and are nulled at ingest, so none should
survive. 0.99 rather than 1.0 leaves room for rounding in the published price."""

OVERROUND_MEDIAN = (1.02, 1.15)
"""Measured median: 1.074. Asserted alongside the minimum because a systematic
column misread moves the median, whereas thirty-three bad rows in 246,052
would not."""

RECENT_STATS_FROM = "2021-08-01"
RECENT_STATS_FLOOR = 0.90
"""Shot coverage among competitions whose feed can supply it, over recent
seasons only. Measured: 0.930 since 2021-08-01.

Recent seasons only, because coverage is a timeline rather than a constant —
the provider added shot data to most European divisions around 2019/20, so the
same check over all history reads 0.53 and means nothing. The floor sits below
the measurement because one competition (ENG_5, the National League) supplies
almost no statistics at all: the feed's capability flag is a ceiling, not a
promise, and the per-competition breakdown lives in the dataset card where it
can be seen rather than averaged away."""

MIN_MATCHES_PER_COMPETITION = 500
"""A competition with a couple of hundred fixtures is a fragment, not a league,
and it reaches every per-competition report as a caveat. Switzerland's
'Challenge League' label was exactly this — two promotion play-off matches —
and was removed from the registry rather than carried. Measured minimum across
the 39 competitions: 920."""

MAX_SINGLE_SEASON_TEAM_SHARE = 0.30
"""Teams appearing in exactly one season are mostly real — a single promotion,
a club that folded. A large share of them instead means team identity broke and
one club's history was split across several ids, which silently truncates every
rolling feature. Measured: 202 of 1,361 teams, 0.148."""

# A season label is either split ("2024-25") or calendar ("2024"); the registry
# normalises both. Duplicated here as a regex rather than imported, because the
# point of the check is to catch a table whose labels stopped matching the rule
# — importing the producer's own parser would make it unable to fail.
_SEASON_LABEL = re.compile(r"^\d{4}(-\d{2})?$")

# `Timedelta(60, "D")` rather than `Timedelta(days=60)`: the keyword form
# routes through a bare-integer NumPy conversion that pandas 2.3 deprecates,
# and this suite runs with DeprecationWarning as an error.
_SEASON_SLACK = pd.Timedelta(60, "D")
"""How far past its nominal window a season may legitimately run.

A split season nominally runs July to June and a calendar one January to
December, but 2020 broke both: the suspended European seasons finished in
August 2020 and the Brasileirao 2020 finished in February 2021. Measured worst
legitimate overshoot across every competition-season in the ingest: 43 days
(ROU_1 2019-20) for split, 57 days (BRA_1 2020) for calendar.

Sixty days clears both and still catches the one genuine error, which misses
by 213. A tighter bound would fire on the pandemic for no useful reason, and a
looser one would stop distinguishing a mislabelled row from a late final."""


# --- Schema ------------------------------------------------------------------


@check("columns are exactly the canonical schema, in order")
def _columns_match(matches: pd.DataFrame) -> str | None:
    actual = tuple(matches.columns)
    if actual == CANONICAL_COLUMNS:
        return None
    missing = sorted(set(CANONICAL_COLUMNS) - set(actual))
    extra = sorted(set(actual) - set(CANONICAL_COLUMNS))
    if missing or extra:
        return f"missing={missing} unexpected={extra}"
    return "same columns, wrong order"


@check("dtypes are the canonical dtypes")
def _dtypes_match(matches: pd.DataFrame) -> str | None:
    """Where a nullable ``Int16`` quietly becomes ``float64``, turning 13 shots
    into 13.0 and "no data" into NaN — two facts that must stay distinct."""
    actual = {column: str(dtype) for column, dtype in matches.dtypes.items()}
    wrong = {
        column: f"{actual[column]} != {expected}"
        for column, expected in CANONICAL_SCHEMA.items()
        if column in actual and actual[column] != expected
    }
    return f"wrong dtypes: {wrong}" if wrong else None


@check("every column is classified as pre-match or post-match")
def _leakage_classification_is_complete(matches: pd.DataFrame) -> str | None:
    """The leakage gate.

    Training on a column that does not exist until after the match is the most
    expensive mistake available here, and it does not announce itself: the
    model scores brilliantly and is useless on a Saturday morning. The defence
    is that every column has a declared side of kick-off, so the feature layer
    cannot reach for one by accident — and an unclassified column is a column
    nobody decided about, which is how the first leak gets in.
    """
    classified = PRE_MATCH_COLUMNS | POST_MATCH_COLUMNS
    overlap = sorted(PRE_MATCH_COLUMNS & POST_MATCH_COLUMNS)
    if overlap:
        return f"columns classified as both pre- and post-match: {overlap}"
    unclassified = sorted(set(matches.columns) - classified)
    return f"unclassified columns: {unclassified}" if unclassified else None


# --- Integrity ---------------------------------------------------------------


@check("match ids are unique")
def _ids_are_unique(matches: pd.DataFrame) -> str | None:
    duplicated = int(matches["match_id"].duplicated().sum())
    return f"{duplicated:,} duplicate match ids" if duplicated else None


@check("core columns are never null")
def _core_columns_are_populated(matches: pd.DataFrame) -> str | None:
    nulls = {
        column: int(matches[column].isna().sum())
        for column in sorted(CORE_COLUMNS)
        if column in matches and matches[column].isna().any()
    }
    return f"nulls in core columns: {nulls}" if nulls else None


@check("no team plays itself")
def _no_team_plays_itself(matches: pd.DataFrame) -> str | None:
    offending = int((matches["home_team_id"] == matches["away_team_id"]).sum())
    return f"{offending:,} matches with the same team on both sides" if offending else None


@check("the stated result agrees with the score")
def _result_agrees_with_score(matches: pd.DataFrame) -> str | None:
    derived = pd.Series("D", index=matches.index, dtype="object")
    derived[matches["home_goals"] > matches["away_goals"]] = "H"
    derived[matches["home_goals"] < matches["away_goals"]] = "A"
    disagreeing = int((derived != matches["result"]).sum())
    return f"{disagreeing:,} results disagree with their score" if disagreeing else None


@check("result values are H, D or A")
def _results_are_known_values(matches: pd.DataFrame) -> str | None:
    unknown = sorted(set(matches["result"].dropna()) - {member.value for member in Result})
    return f"unknown result values: {unknown}" if unknown else None


@check("goals are within plausible bounds")
def _goals_are_plausible(matches: pd.DataFrame) -> str | None:
    for side in ("home", "away"):
        # dropna before comparing: min() over an empty nullable column returns
        # pd.NA, and `pd.NA < 0` raises rather than being falsy — so an empty
        # table would crash the suite instead of passing it, which is exactly
        # the moment a report is most needed.
        column = matches[f"{side}_goals"].dropna()
        if column.empty:
            continue
        low, high = int(column.min()), int(column.max())
        if low < 0 or high > MAX_PLAUSIBLE_GOALS:
            return f"{side}_goals ranges {low}..{high}"
    return None


@check("the half-time score never exceeds full-time")
def _half_time_is_consistent(matches: pd.DataFrame) -> str | None:
    """Goals are not un-scored. A violation means the HT and FT columns were
    read from the wrong positions — an error that leaves every row plausible on
    its own."""
    known = matches.dropna(subset=["ht_home_goals", "ht_away_goals"])
    impossible = int(
        (known["ht_home_goals"] > known["home_goals"]).sum()
        + (known["ht_away_goals"] > known["away_goals"]).sum()
    )
    return f"{impossible:,} half-time scores exceed full-time" if impossible else None


@check("shots on target never exceed shots")
def _shots_are_consistent(matches: pd.DataFrame) -> str | None:
    """Both sides are checked separately: an error affecting only one would mean
    the columns were read from the wrong offsets rather than being bad data."""
    offending: dict[str, int] = {}
    for side in ("home", "away"):
        known = matches.dropna(subset=[f"{side}_shots", f"{side}_shots_on_target"])
        count = int((known[f"{side}_shots_on_target"] > known[f"{side}_shots"]).sum())
        if count:
            offending[side] = count
    return f"shots on target exceed shots: {offending}" if offending else None


@check("the table is chronological")
def _table_is_chronological(matches: pd.DataFrame) -> str | None:
    """Every temporal split and every rolling feature downstream assumes this.
    Relying on an incidental ordering is how a 'time-aware' split quietly stops
    being one."""
    return None if matches["date"].is_monotonic_increasing else "dates are not ordered"


@check("season labels are canonical")
def _season_labels_are_canonical(matches: pd.DataFrame) -> str | None:
    bad = sorted(
        {label for label in matches["season"].dropna().unique() if not _SEASON_LABEL.match(label)}
    )
    return f"non-canonical season labels: {bad[:10]}" if bad else None


@check("team ids carry their competition's country", severity=Severity.ERROR)
def _team_ids_match_their_country(matches: pd.DataFrame) -> str | None:
    """Team identity is scoped to a country so a promoted club keeps one id
    across divisions. If the prefix and the competition disagree, that scoping
    broke and two divisions of the same country no longer join."""
    expected = matches["competition_id"].str.split("_").str[0].str.lower()
    offending = 0
    for side in ("home", "away"):
        prefix = matches[f"{side}_team_id"].str.split(":").str[0]
        offending += int((prefix != expected).sum())
    return f"{offending:,} team ids with the wrong country prefix" if offending else None


# --- Distribution ------------------------------------------------------------


@check("the home win rate is football", min_rows=5_000)
def _home_win_rate_is_plausible(matches: pd.DataFrame) -> str | None:
    share = float((matches["result"] == "H").mean())
    low, high = HOME_WIN_RATE
    return None if low <= share <= high else f"home win rate {share:.3f} outside {low}-{high}"


@check("the draw rate is football", min_rows=5_000)
def _draw_rate_is_plausible(matches: pd.DataFrame) -> str | None:
    share = float((matches["result"] == "D").mean())
    low, high = DRAW_RATE
    return None if low <= share <= high else f"draw rate {share:.3f} outside {low}-{high}"


@check("odds are decimal and imply a real book", min_rows=1_000)
def _odds_imply_a_real_book(matches: pd.DataFrame) -> str | None:
    columns = ["odds_home", "odds_draw", "odds_away"]
    priced = matches.dropna(subset=columns)
    if priced.empty:
        return None
    if not bool((priced[columns] > 1.0).all().all()):
        return "decimal odds at or below 1.0"
    overround = 1 / priced["odds_home"] + 1 / priced["odds_draw"] + 1 / priced["odds_away"]
    if float(overround.min()) < MIN_OVERROUND:
        return f"implied book sums to {overround.min():.3f}, below {MIN_OVERROUND}"
    low, high = OVERROUND_MEDIAN
    median = float(overround.median())
    return None if low < median < high else f"median overround {median:.3f} outside {low}-{high}"


@check("no team appears in only one season", severity=Severity.WARNING, min_rows=5_000)
def _team_history_is_continuous(matches: pd.DataFrame) -> str | None:
    """The rename detector. A club that genuinely played one season looks
    identical here to one that was renamed; the difference needs a human, and a
    rename splits one club's history into two half-length records that every
    rolling feature then computes over the wrong window."""
    pairs = list(zip(matches["home_team_id"], matches["season"], strict=True))
    single = find_single_season_teams(pairs)
    total = matches["home_team_id"].nunique()
    if not total:
        return None
    share = len(single) / total
    if share <= MAX_SINGLE_SEASON_TEAM_SHARE:
        return None
    return f"{len(single)} of {total} teams ({share:.1%}) appear in one season only"


# The full suite in report order. Registry-dependent checks are appended by
# `match_checks`, which is the only thing that needs the registry loaded.
_STANDALONE_CHECKS: tuple[Check, ...] = (
    _columns_match,
    _dtypes_match,
    _leakage_classification_is_complete,
    _ids_are_unique,
    _core_columns_are_populated,
    _no_team_plays_itself,
    _result_agrees_with_score,
    _results_are_known_values,
    _goals_are_plausible,
    _half_time_is_consistent,
    _shots_are_consistent,
    _table_is_chronological,
    _season_labels_are_canonical,
    _team_ids_match_their_country,
    _home_win_rate_is_plausible,
    _draw_rate_is_plausible,
    _odds_imply_a_real_book,
    _team_history_is_continuous,
)


# --- Referential (needs the registry) ----------------------------------------


def _season_window(label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The dates a season with this label may plausibly contain.

    Nominal window plus :data:`_SEASON_SLACK` at both ends. The slack is what
    keeps the check from firing on the 2020 season, which really did run long
    in most of the world, while still catching a row filed under a season two
    calendar years away from it.
    """
    year = int(label[:4])
    if "-" in label:
        start, end = pd.Timestamp(year, 7, 1), pd.Timestamp(year + 1, 6, 30)
    else:
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
    return start - _SEASON_SLACK, end + _SEASON_SLACK


def _registry_checks(registry: Registry) -> tuple[Check, ...]:
    """Checks that compare the table against the configuration it came from."""

    known = {competition.id: competition for competition in registry.competitions}

    @check("every competition is registered")
    def _competitions_are_registered(matches: pd.DataFrame) -> str | None:
        unknown = sorted(set(matches["competition_id"]) - set(known))
        return f"unregistered competition ids: {unknown}" if unknown else None

    @check("competition metadata agrees with the registry")
    def _metadata_agrees(matches: pd.DataFrame) -> str | None:
        """Country, name and tier are denormalised into every row so a single
        read needs no join. Denormalised data drifts; this is the check that
        says it has not."""
        disagreements: list[str] = []
        for competition_id, rows in matches.groupby("competition_id", observed=True):
            competition = known.get(str(competition_id))
            if competition is None:
                continue  # reported by the previous check
            for column, expected in (
                ("country", competition.country),
                ("competition", competition.name),
            ):
                seen = set(rows[column].dropna().unique())
                if seen != {expected}:
                    disagreements.append(f"{competition_id}.{column}={sorted(seen)}")
            tiers = set(rows["tier"].dropna().unique())
            if tiers != ({competition.tier} if competition.tier is not None else set()):
                disagreements.append(f"{competition_id}.tier={sorted(tiers)}")
        return f"rows disagree with the registry: {disagreements[:5]}" if disagreements else None

    @check("matches fall inside the season they are labelled with", severity=Severity.WARNING)
    def _dates_match_their_season(matches: pd.DataFrame) -> str | None:
        """A warning, not an error, because it fires on genuine upstream noise.

        Measured over the full ingest: exactly one row in 305,499 — an
        Argentinian match played 2015-01-29 and labelled season 2013-14, in the
        provider's file itself, 213 days outside the widest defensible window.
        One misfiled row does not justify refusing the dataset; silently
        carrying it into a season-level aggregate does not either, so it is
        reported and left in place.
        """
        stray: list[str] = []
        for label, rows in matches.groupby("season", observed=True):
            start, end = _season_window(str(label))
            outside = rows[(rows["date"] < start) | (rows["date"] > end)]
            if not outside.empty:
                stray.append(f"{label}: {len(outside)} row(s), e.g. {outside['date'].max().date()}")
        return f"matches outside their season window: {stray[:5]}" if stray else None

    @check("recent matches carry the statistics their feed promises", min_rows=5_000)
    def _stat_coverage_holds(matches: pd.DataFrame) -> str | None:
        """The other half of "shots on target never exceed shots".

        Nulling an impossible value is only correct while it stays rare. A rule
        that quietly removed a tenth of the shot data would pass every per-row
        check and ruin every shot-based feature, and only an aggregate can see
        it.
        """
        capable = {
            competition.id
            for competition in registry.competitions
            if Capability.MATCH_STATS in competition.capabilities
        }
        recent = matches[
            matches["competition_id"].isin(capable) & (matches["date"] >= RECENT_STATS_FROM)
        ]
        if len(recent) < 1_000:
            return None
        coverage = float(recent["home_shots"].notna().mean())
        if coverage >= RECENT_STATS_FLOOR:
            return None
        return f"shot coverage {coverage:.3f} since {RECENT_STATS_FROM}, below {RECENT_STATS_FLOOR}"

    @check(
        "every competition has enough history to model",
        severity=Severity.WARNING,
        min_rows=5_000,
    )
    def _competitions_are_substantial(matches: pd.DataFrame) -> str | None:
        counts = matches.groupby("competition_id", observed=True).size()
        thin = counts[counts < MIN_MATCHES_PER_COMPETITION]
        return f"too little data to model: {thin.to_dict()}" if not thin.empty else None

    return (
        _competitions_are_registered,
        _metadata_agrees,
        _dates_match_their_season,
        _stat_coverage_holds,
        _competitions_are_substantial,
    )


def match_checks(registry: Registry | None = None) -> tuple[Check, ...]:
    """The full suite.

    Args:
        registry: The competition registry the table should agree with.
            Defaults to the shipped ``configs/leagues.yaml``. Injectable so a
            test can validate a synthetic table against a synthetic registry
            rather than against thirty-nine real leagues.
    """
    return _STANDALONE_CHECKS + _registry_checks(registry or load_registry())
