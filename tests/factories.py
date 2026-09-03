"""Synthetic canonical tables that are valid football.

Every validation test has the same shape: build a table the whole suite passes,
break exactly one thing, assert exactly that check fails. That only works if
the starting table is genuinely valid — including its *distributions*, because
a home-win rate of 1.0 fails a check that has nothing to do with the property
under test and turns every assertion into a guess about which failure was meant.

So :func:`league_frame` produces a league that looks like football: a stable set
of clubs playing a double round robin per season, results in the measured
proportions, half-time scores below full-time, shots on target below shots, and
a bookmaker's margin of about 8%. It is deliberately large enough (14 seasons,
5,320 matches) to clear the row thresholds on the distribution checks, because
a suite that silently skips half its checks in every test is not a suite.

Kept out of ``conftest.py`` on purpose: these are constructors, not fixtures,
and several tests need one at module scope where a fixture cannot reach.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from src.ingestion.base import CANONICAL_SCHEMA
from src.ingestion.registry import Competition, Feed, Registry

# 45% home, 27% draw, 28% away — the proportions measured over the real ingest.
# Spelled as a repeating pattern of 100 rather than drawn at random so a failing
# test is reproducible without a seed to remember.
_RESULT_CYCLE = ("H",) * 45 + ("D",) * 27 + ("A",) * 28

_RESULT_STRIDE = 37
"""Walk the cycle in steps of 37 rather than 1.

A season of 380 fixtures is not a whole number of cycles, so reading the
pattern in order gives the last partial cycle entirely to its first block —
which pushed the synthetic home-win rate to 0.474. Any stride coprime with 100
spreads every partial prefix across the whole pattern, so a season of any
length lands on the intended mix."""

# One triple for every match. Implied probabilities sum to 1.080, which is a
# real bookmaker's margin and sits inside the measured median band.
_ODDS = (2.00, 3.40, 3.50)

_SCORES = {"H": (2, 1), "D": (1, 1), "A": (0, 1)}
_HALF_TIME = {"H": (1, 0), "D": (0, 0), "A": (0, 1)}


def canonical_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Coerce records into the canonical schema, filling absent columns with null."""
    built = pd.DataFrame.from_records(list(rows)).reindex(columns=list(CANONICAL_SCHEMA))
    return built.astype(CANONICAL_SCHEMA)


def season_labels(first_year: int, count: int) -> tuple[str, ...]:
    """``2012-13``-style split-season labels, ``count`` of them."""
    return tuple(f"{year}-{(year + 1) % 100:02d}" for year in range(first_year, first_year + count))


def league_frame(
    *,
    competition_id: str = "ENG_1",
    country: str = "England",
    name: str = "Premier League",
    tier: int | None = 1,
    seasons: Sequence[str] | None = None,
    teams: int = 20,
    with_stats: bool = True,
    with_odds: bool = True,
) -> pd.DataFrame:
    """One competition's full history as a canonical frame.

    Every season fields the same clubs, which keeps the rename detector quiet:
    a synthetic league whose teams change annually would trip a warning about a
    problem the test is not about.
    """
    if seasons is None:
        seasons = season_labels(2012, 14)

    prefix = competition_id.split("_")[0]
    names = [f"Team {index:02d}" for index in range(teams)]
    identifiers = {team: f"{prefix.lower()}:{team.lower().replace(' ', '-')}" for team in names}

    records: list[dict[str, Any]] = []
    for season in seasons:
        start_year = int(season[:4])
        # Fixtures are dated by their index across the season, spread over the
        # 280 days from August to May, so every date lands inside the season
        # window the validation suite checks against.
        fixtures = [(h, a) for h in names for a in names if h != a]
        span = pd.Timestamp(start_year, 8, 1)
        for index, (home, away) in enumerate(fixtures):
            result = _RESULT_CYCLE[(index * _RESULT_STRIDE) % len(_RESULT_CYCLE)]
            home_goals, away_goals = _SCORES[result]
            ht_home, ht_away = _HALF_TIME[result]
            record: dict[str, Any] = {
                "match_id": f"{competition_id}:{season}:{index:05d}",
                "provider": "stub",
                "competition_id": competition_id,
                "country": country,
                "competition": name,
                "tier": tier,
                "season": season,
                "date": span + pd.Timedelta(index * 280 // max(len(fixtures), 1), "D"),
                "kickoff": "15:00",
                "home_team": home,
                "away_team": away,
                "home_team_id": identifiers[home],
                "away_team_id": identifiers[away],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
                "ht_home_goals": ht_home,
                "ht_away_goals": ht_away,
                "ht_result": ("H" if ht_home > ht_away else "A" if ht_home < ht_away else "D"),
            }
            if with_stats:
                record |= {
                    "home_shots": 14,
                    "away_shots": 11,
                    "home_shots_on_target": 6,
                    "away_shots_on_target": 4,
                    "home_corners": 6,
                    "away_corners": 4,
                    "home_fouls": 11,
                    "away_fouls": 13,
                    "home_yellows": 1,
                    "away_yellows": 2,
                    "home_reds": 0,
                    "away_reds": 0,
                    "referee": "A Official",
                }
            if with_odds:
                record |= dict(zip(("odds_home", "odds_draw", "odds_away"), _ODDS, strict=True))
            records.append(record)

    frame = canonical_frame(records)
    # Sorted because the suite checks it, and because the real table is.
    return frame.sort_values(["date", "competition_id", "match_id"], kind="stable").reset_index(
        drop=True
    )


def league_registry(
    *,
    competition_id: str = "ENG_1",
    country: str = "England",
    name: str = "Premier League",
    tier: int | None = 1,
    feed: Feed = Feed.MAIN,
) -> Registry:
    """The registry :func:`league_frame` agrees with."""
    return Registry(
        competitions=(
            Competition(
                id=competition_id,
                country=country,
                name=name,
                tier=tier,
                feed=feed,
                code="E0",
            ),
        )
    )
