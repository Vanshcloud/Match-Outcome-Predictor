"""Tests against real downloaded provider files.

Every test here SKIPS when the data is absent, so a clean checkout stays green
and CI needs no network. Run `python scripts/fetch_data.py` first to enable
them, then `pytest -m integration`.

These exist because the unit suite's fixtures are shaped like the real files
but are not the real files, and every genuinely surprising thing this project
has hit — HTML served with a non-error status, cp1252 bytes, a header narrower
than its own rows — was found by reading real data rather than by reasoning
about the schema.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.base import CANONICAL_SCHEMA, CORE_COLUMNS
from src.ingestion.csv_reader import read_provider_csv
from src.ingestion.registry import load_registry
from src.ingestion.teams import find_single_season_teams
from src.utils.config import load_settings

pytestmark = pytest.mark.integration

SETTINGS = load_settings()
MATCHES = SETTINGS.paths.processed_dir / "matches.parquet"
RAW = SETTINGS.paths.raw_dir / "football-data"


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    if not MATCHES.is_file():
        pytest.skip(f"no ingested data at {MATCHES}; run scripts/fetch_data.py")
    return pd.read_parquet(MATCHES)


def test_the_canonical_schema_survives_a_parquet_round_trip(matches: pd.DataFrame) -> None:
    """Where a nullable Int16 quietly becomes a float64 if dtypes are not
    re-asserted after concatenation."""
    assert list(matches.columns) == list(CANONICAL_SCHEMA)
    assert {c: str(d) for c, d in matches.dtypes.items()} == CANONICAL_SCHEMA


def test_no_core_column_is_ever_null(matches: pd.DataFrame) -> None:
    """A row missing any of these is not a match this project can use."""
    nulls = {c: int(matches[c].isna().sum()) for c in sorted(CORE_COLUMNS)}
    assert not any(nulls.values()), f"nulls in core columns: {nulls}"


def test_match_ids_are_unique(matches: pd.DataFrame) -> None:
    assert matches["match_id"].is_unique


def test_the_table_is_chronological(matches: pd.DataFrame) -> None:
    """Every temporal split and rolling feature downstream assumes this."""
    assert matches["date"].is_monotonic_increasing


def test_the_stated_result_always_matches_the_score(matches: pd.DataFrame) -> None:
    """Rows that disagreed were dropped at ingest, so none should survive."""
    derived = pd.Series("D", index=matches.index)
    derived[matches["home_goals"] > matches["away_goals"]] = "H"
    derived[matches["home_goals"] < matches["away_goals"]] = "A"
    assert (derived == matches["result"]).all()


def test_no_team_plays_itself(matches: pd.DataFrame) -> None:
    assert (matches["home_team_id"] != matches["away_team_id"]).all()


def test_home_advantage_is_present_and_plausible(matches: pd.DataFrame) -> None:
    """A sanity check on the whole pipeline rather than on one function.

    Home advantage in professional football is large and stable — roughly 44%
    of matches. A rate near 33% would mean home and away had been swapped
    somewhere, which is the kind of error that produces a model that trains
    perfectly and is exactly wrong.
    """
    share = (matches["result"] == "H").mean()
    assert 0.40 <= share <= 0.52, f"home win rate {share:.3f} is not football"


def test_draw_rate_is_plausible(matches: pd.DataFrame) -> None:
    share = (matches["result"] == "D").mean()
    assert 0.20 <= share <= 0.32, f"draw rate {share:.3f} is not football"


def test_scores_are_within_sane_bounds(matches: pd.DataFrame) -> None:
    """The record top-flight scoreline is in the low teens. Anything past that
    means a column shifted."""
    assert matches["home_goals"].max() <= 15
    assert matches["away_goals"].max() <= 15
    assert matches["home_goals"].min() >= 0


def test_half_time_scores_never_exceed_full_time(matches: pd.DataFrame) -> None:
    """Goals are not un-scored. A violation means the HT and FT columns were
    read from the wrong positions."""
    known = matches.dropna(subset=["ht_home_goals", "ht_away_goals"])
    assert (known["ht_home_goals"] <= known["home_goals"]).all()
    assert (known["ht_away_goals"] <= known["away_goals"]).all()


@pytest.mark.parametrize("side", ["home", "away"])
def test_shots_on_target_never_exceed_shots(matches: pd.DataFrame, side: str) -> None:
    """Impossible pairs are nulled at ingest, so none should survive. Both
    sides are checked: an error that affected only one would mean the columns
    were read from the wrong offsets."""
    known = matches.dropna(subset=[f"{side}_shots", f"{side}_shots_on_target"])
    assert (known[f"{side}_shots_on_target"] <= known[f"{side}_shots"]).all()


def test_nulling_bad_fields_did_not_gut_the_dataset(matches: pd.DataFrame) -> None:
    """The other half of the previous test. Nulling an impossible value is only
    correct while it stays rare — a rule that quietly removed a tenth of the
    shot data would pass the check above and ruin every shot-based feature."""
    with_stats = matches[matches["home_shots"].notna()]
    assert len(with_stats) > 100_000, "shot coverage collapsed"


def test_every_competition_has_enough_history_to_model(matches: pd.DataFrame) -> None:
    """A competition with a couple of fixtures is not a league, and it reaches
    every per-competition report as a caveat. Switzerland's 'Challenge League'
    label was exactly this — two promotion-playoff matches — and was removed
    from the registry rather than carried."""
    per_competition = matches.groupby("competition_id").size()
    thin = per_competition[per_competition < 500]
    assert thin.empty, f"competitions with too little data to model: {dict(thin)}"


def test_odds_are_decimal_and_imply_a_real_book(matches: pd.DataFrame) -> None:
    """Decimal odds exceed 1.0, and a real book's implied probabilities sum
    above it — the excess is the margin.

    Triples below 1.0 are provider typos and are nulled at ingest, so none
    should survive here. The median is asserted too: a systematic column
    misread would move it, whereas 33 bad rows in 246,052 would not.
    """
    priced = matches.dropna(subset=["odds_home", "odds_draw", "odds_away"])
    assert (priced[["odds_home", "odds_draw", "odds_away"]] > 1.0).all().all()

    overround = 1 / priced["odds_home"] + 1 / priced["odds_draw"] + 1 / priced["odds_away"]
    assert overround.min() >= 0.99, "an implied book below 100% should have been nulled"
    # A real bookmaker's margin is a few per cent. Measured median: 1.074.
    assert 1.02 < overround.median() < 1.15


def test_every_ingested_competition_is_registered(matches: pd.DataFrame) -> None:
    registered = {c.id for c in load_registry().competitions}
    assert set(matches["competition_id"]) <= registered


def test_team_ids_are_stable_across_divisions(matches: pd.DataFrame) -> None:
    """A club promoted from the Championship must keep its id, or every rolling
    feature restarts its history at the promotion."""
    english = matches[matches["competition_id"].str.startswith("ENG_")]
    if english.empty:
        pytest.skip("no English competitions ingested")
    by_team = english.groupby("home_team_id")["competition_id"].nunique()
    assert (by_team > 1).any(), "expected at least one club to appear in two divisions"


def test_rename_candidates_are_reported_not_hidden(matches: pd.DataFrame) -> None:
    """Not an assertion that there are none — one-season clubs are real. The
    point is that the detector runs and returns something inspectable."""
    rows = list(zip(matches["home_team_id"], matches["season"], strict=True))
    single = find_single_season_teams(rows)
    assert isinstance(single, dict)


def test_every_cached_raw_file_still_parses() -> None:
    """The regression guard for the reader. Italian Serie B 2003/04 is the file
    that falsified the original ragged-row rule; this walks all of them."""
    if not RAW.is_dir():
        pytest.skip(f"no raw files at {RAW}; run scripts/fetch_data.py")
    files = sorted(RAW.rglob("*.csv"))
    if not files:
        pytest.skip("no raw files downloaded")
    for path in files:
        rows = read_provider_csv(path)
        assert rows, f"{path} parsed to zero rows"
