"""Every check, broken on purpose.

The shape of each test is the same: start from a table that passes all
twenty-three checks, break exactly one property, and assert that exactly one
check noticed. Asserting the *set* of failures rather than "at least one
failed" is what catches the two ways a suite rots — a check that has quietly
stopped testing anything, and a check that fires on data it should accept.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.base import CANONICAL_COLUMNS, POST_MATCH_COLUMNS, PRE_MATCH_COLUMNS
from src.ingestion.registry import Competition, Feed, Registry
from src.validation import matches as checks
from src.validation.matches import match_checks
from src.validation.report import Outcome, Severity, run_checks
from src.validation.report import names as report_names
from tests.factories import canonical_frame, league_frame, league_registry, season_labels

REGISTRY = league_registry()


def failures(frame: pd.DataFrame, registry: Registry = REGISTRY) -> set[str]:
    report = run_checks(frame, match_checks(registry))
    return {result.name for result in report.of(Outcome.FAILED)}


def assert_only(frame: pd.DataFrame, name: str, registry: Registry = REGISTRY) -> None:
    assert failures(frame, registry) == {name}


# --- the baseline ------------------------------------------------------------


def test_a_valid_league_passes_every_check() -> None:
    """The premise every other test rests on. If this stops holding, every
    'exactly one check failed' assertion below becomes meaningless."""
    report = run_checks(league_frame(), match_checks(REGISTRY))
    assert report.ok
    assert report.of(Outcome.FAILED) == ()
    assert report.of(Outcome.SKIPPED) == (), "a skipped check is an unasked question"


def test_check_names_are_unique() -> None:
    """The name is the report's key. Two checks sharing one make the report
    ambiguous about which property broke."""
    names = report_names(checks.match_checks())
    assert len(names) == len(set(names))


def test_the_default_suite_loads_the_shipped_registry() -> None:
    assert len(checks.match_checks()) == len(checks.match_checks(REGISTRY))


# --- schema ------------------------------------------------------------------


def test_a_missing_column_is_caught() -> None:
    frame = league_frame().drop(columns=["referee"])
    assert "columns are exactly the canonical schema, in order" in failures(frame)


def test_reordered_columns_are_caught() -> None:
    """Column *order* is part of the contract: a Parquet schema diff is only
    readable while it is stable."""
    frame = league_frame()
    reordered = frame[list(reversed(CANONICAL_COLUMNS))]
    result = checks._columns_match(reordered)
    assert result.outcome is Outcome.FAILED
    assert result.message == "same columns, wrong order"


def test_an_extra_column_is_reported_as_unexpected() -> None:
    frame = league_frame()
    frame["surprise"] = 1
    result = checks._columns_match(frame)
    assert "unexpected=['surprise']" in result.message


def test_a_widened_dtype_is_caught() -> None:
    """The failure this exists for: a nullable Int16 promoted to float64, where
    13 shots becomes 13.0 and 'no data' becomes NaN."""
    frame = league_frame()
    frame["home_shots"] = frame["home_shots"].astype("float64")
    assert_only(frame, "dtypes are the canonical dtypes")


def test_an_unclassified_column_is_a_leakage_gap() -> None:
    """A column nobody decided about is how the first leak gets in."""
    frame = league_frame()
    frame["expected_goals"] = 1.4
    result = checks._leakage_classification_is_complete(frame)
    assert result.outcome is Outcome.FAILED
    assert "expected_goals" in result.message


def test_a_column_classified_as_both_is_a_leakage_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checks, "PRE_MATCH_COLUMNS", PRE_MATCH_COLUMNS | {"result"})
    result = checks._leakage_classification_is_complete(league_frame())
    assert result.outcome is Outcome.FAILED
    assert "both pre- and post-match" in result.message


def test_the_classification_partitions_the_whole_schema() -> None:
    """The invariant behind the check: every canonical column has exactly one
    side of kick-off, so the feature layer can never reach for an unlabelled
    one."""
    assert set(CANONICAL_COLUMNS) == PRE_MATCH_COLUMNS | POST_MATCH_COLUMNS
    assert not PRE_MATCH_COLUMNS & POST_MATCH_COLUMNS


def test_the_score_and_the_target_are_post_match() -> None:
    """Stated as its own assertion because getting this backwards is the one
    mistake that cannot be recovered from downstream."""
    assert {"home_goals", "away_goals", "result"} <= POST_MATCH_COLUMNS
    assert {"home_shots", "away_shots", "referee"} <= POST_MATCH_COLUMNS
    assert {"date", "home_team_id", "season"} <= PRE_MATCH_COLUMNS


# --- integrity ---------------------------------------------------------------


def test_duplicate_match_ids_are_caught() -> None:
    frame = league_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    duplicated = duplicated.sort_values(["date", "competition_id", "match_id"], kind="stable")
    assert_only(duplicated.reset_index(drop=True), "match ids are unique")


def test_a_null_core_column_is_caught() -> None:
    frame = league_frame()
    frame.loc[3, "home_team_id"] = pd.NA
    assert failures(frame) == {
        "core columns are never null",
        # A null id also has no country prefix, which is the point: one hole
        # trips both checks, and the report says so rather than picking one.
        "team ids carry their competition's country",
    }


def test_a_team_playing_itself_is_caught() -> None:
    frame = league_frame()
    frame.loc[5, "away_team_id"] = frame.loc[5, "home_team_id"]
    assert_only(frame, "no team plays itself")


def test_a_result_that_disagrees_with_its_score_is_caught() -> None:
    """The error that produces a model which trains perfectly and is exactly
    wrong."""
    frame = league_frame()
    frame.loc[frame["result"] == "H", "result"] = "A"
    assert failures(frame) >= {"the stated result agrees with the score"}


def test_an_unknown_result_value_is_caught() -> None:
    frame = league_frame()
    frame.loc[1, "result"] = "X"
    assert failures(frame) == {
        "result values are H, D or A",
        "the stated result agrees with the score",
    }


def test_an_impossible_score_is_caught() -> None:
    """The record top-flight scoreline is in the low teens; past that, a column
    shifted. The result is rewritten alongside so this test is about the bound
    and not about the score agreeing with its own result."""
    frame = league_frame()
    frame.loc[2, "home_goals"] = 99
    frame.loc[2, "result"] = "H"
    assert failures(frame) == {"goals are within plausible bounds"}


def test_a_negative_score_is_caught() -> None:
    frame = league_frame()
    frame.loc[2, "away_goals"] = -1
    assert failures(frame) >= {"goals are within plausible bounds"}


def test_a_half_time_score_above_full_time_is_caught() -> None:
    frame = league_frame()
    frame.loc[4, "ht_home_goals"] = 9
    assert_only(frame, "the half-time score never exceeds full-time")


def test_shots_on_target_above_shots_are_caught() -> None:
    frame = league_frame()
    frame.loc[6, "away_shots_on_target"] = 99
    result = run_checks(frame, match_checks(REGISTRY))
    failed = {check.name: check.message for check in result.of(Outcome.FAILED)}
    assert "shots on target never exceed shots" in failed
    assert "'away': 1" in failed["shots on target never exceed shots"]


def test_the_same_fixture_under_two_competitions_is_caught() -> None:
    """`match_id` hashes the competition, so a fixture published under two
    division codes gets two ids and survives deduplication. Every copy is
    internally consistent, so no per-row check can see it — and a rolling
    feature quietly averages a team's form over fixtures it played once.

    This is what five of the provider's early files did: copies of SP1.csv
    served as P1, SC1 and SP2, which put 380 Spanish matches into Portugal's
    first season and duplicated 1,222 more into Spain's second division.
    """
    first = league_frame()
    copy = first.assign(
        competition_id="ENG_2",
        competition="Championship",
        tier=pd.array([2] * len(first), dtype="Int8"),
        match_id=first["match_id"] + "-copy",
    )
    together = (
        pd.concat([first, copy], ignore_index=True)
        .sort_values(["date", "competition_id", "match_id"], kind="stable")
        .reset_index(drop=True)
    )
    assert "no fixture appears in two competitions" in failures(together, REGISTRY)


def test_fixtures_are_matched_on_names_not_ids() -> None:
    """Ids are country-scoped, so the Portuguese copies carried `por:` ids and
    were invisible to an id-based comparison. Names are what survive the
    mislabelling."""
    first = league_frame()
    copy = first.assign(
        competition_id="POR_1",
        country="Portugal",
        competition="Primeira Liga",
        match_id=first["match_id"] + "-copy",
        home_team_id=first["home_team_id"].str.replace("eng:", "por:", regex=False),
        away_team_id=first["away_team_id"].str.replace("eng:", "por:", regex=False),
    )
    together = (
        pd.concat([first, copy], ignore_index=True)
        .sort_values(["date", "competition_id", "match_id"], kind="stable")
        .reset_index(drop=True)
    )
    assert "no fixture appears in two competitions" in failures(together, REGISTRY)


def test_an_empty_table_has_no_duplicate_fixtures() -> None:
    assert checks._fixtures_are_not_duplicated(canonical_frame([])).outcome is Outcome.PASSED


def test_an_unordered_table_is_caught() -> None:
    """Every temporal split downstream assumes this; relying on an incidental
    ordering is how a time-aware split quietly stops being one."""
    frame = league_frame().sort_values("match_id", ascending=False).reset_index(drop=True)
    assert_only(frame, "the table is chronological")


def test_a_non_canonical_season_label_is_caught() -> None:
    frame = league_frame()
    frame.loc[frame["season"] == "2012-13", "season"] = "2012/2013"
    assert failures(frame) >= {"season labels are canonical"}


def test_a_team_id_from_the_wrong_country_is_caught() -> None:
    """Team identity is scoped by country so a promoted club keeps one id. If
    the prefix and the competition disagree, two divisions no longer join."""
    frame = league_frame()
    frame.loc[7, "home_team_id"] = "fra:team-07"
    assert failures(frame) >= {"team ids carry their competition's country"}


# --- distribution ------------------------------------------------------------


def _reshape_results(frame: pd.DataFrame, home: int, draw: int) -> pd.DataFrame:
    """Re-cut the result mix, keeping every score consistent with its result.

    Rewriting the scores alongside is what isolates the distribution check:
    changing `result` alone would also break 'the result agrees with the
    score', and the test would no longer say which property was under test.
    """
    reshaped = frame.copy()
    pattern = ["H"] * home + ["D"] * draw + ["A"] * (100 - home - draw)
    results = [pattern[index % 100] for index in range(len(reshaped))]
    reshaped["result"] = pd.array(results, dtype="string")
    scores = {"H": (2, 1), "D": (1, 1), "A": (0, 1)}
    reshaped["home_goals"] = pd.array([scores[r][0] for r in results], dtype="Int16")
    reshaped["away_goals"] = pd.array([scores[r][1] for r in results], dtype="Int16")
    reshaped["ht_home_goals"] = 0
    reshaped["ht_away_goals"] = 0
    return reshaped.astype({"ht_home_goals": "Int16", "ht_away_goals": "Int16"})


def test_an_implausible_home_win_rate_is_caught() -> None:
    """The check that catches home and away being swapped in the adapter —
    invisible to every per-row check, obvious in the aggregate."""
    assert_only(_reshape_results(league_frame(), 60, 27), "the home win rate is football")


def test_an_implausible_draw_rate_is_caught() -> None:
    assert_only(_reshape_results(league_frame(), 45, 40), "the draw rate is football")


def test_odds_at_or_below_evens_are_caught() -> None:
    frame = league_frame()
    frame.loc[0, "odds_home"] = 1.0
    result = checks._odds_imply_a_real_book(frame)
    assert result.message == "decimal odds at or below 1.0"


def test_a_book_that_sums_below_one_is_caught() -> None:
    """A book paying out more than it takes does not exist; the triple is a
    provider typo and is nulled at ingest, so none should survive."""
    frame = league_frame()
    frame.loc[0, ["odds_home", "odds_draw", "odds_away"]] = [5.0, 5.0, 5.0]
    result = checks._odds_imply_a_real_book(frame)
    assert "below 0.99" in result.message


def test_an_implausible_margin_is_caught() -> None:
    """Separate from the minimum: a systematic column misread moves the median,
    where a handful of bad rows would not."""
    frame = league_frame()
    frame[["odds_home", "odds_draw", "odds_away"]] = 2.3
    result = checks._odds_imply_a_real_book(frame)
    assert "median overround" in result.message


def test_a_table_without_odds_does_not_fail_the_odds_check() -> None:
    """Half the competitions in the registry carry no prices at all. Absent is
    not wrong."""
    frame = league_frame(with_odds=False)
    assert checks._odds_imply_a_real_book(frame).outcome is Outcome.PASSED


def test_teams_that_never_recur_are_a_warning_not_an_error() -> None:
    """A rename splits one club's history into two half-length records that
    every rolling feature then computes over the wrong window. It needs a
    human, so it warns rather than refusing the dataset."""
    seasons = season_labels(2012, 14)
    frame = pd.concat(
        [
            league_frame(seasons=[season]).assign(
                home_team_id=lambda rows, s=season: rows["home_team_id"] + f"-{s}",
                away_team_id=lambda rows, s=season: rows["away_team_id"] + f"-{s}",
            )
            for season in seasons
        ],
        ignore_index=True,
    ).astype(league_frame().dtypes.to_dict())
    report = run_checks(frame, match_checks(REGISTRY))
    renamed = next(r for r in report.results if r.name == "no team appears in only one season")
    assert renamed.outcome is Outcome.FAILED
    assert renamed.severity is Severity.WARNING
    assert report.ok, "a rename warning must not block the pipeline"


def test_the_rename_check_survives_a_table_with_no_team_ids() -> None:
    """Guards the division that would otherwise be by zero."""
    frame = league_frame()
    frame["home_team_id"] = pd.NA
    assert checks._team_history_is_continuous(frame).outcome is Outcome.PASSED


# --- referential -------------------------------------------------------------


def test_an_unregistered_competition_is_caught() -> None:
    frame = league_frame(competition_id="ITA_1", country="Italy", name="Serie A")
    assert failures(frame) == {"every competition is registered"}


def test_denormalised_metadata_that_drifts_is_caught() -> None:
    """Country, name and tier are copied into every row so a read needs no
    join. Copied data drifts; this is the check that says it has not."""
    frame = league_frame()
    frame.loc[0, "country"] = "Wales"
    assert_only(frame, "competition metadata agrees with the registry")


def test_a_drifted_tier_is_caught() -> None:
    frame = league_frame()
    # astype, or the assignment widens Int8 to int64 and the dtype check fires
    # too — a second failure that would hide which property this test is about.
    frame["tier"] = pd.array([2] * len(frame), dtype="Int8")
    assert_only(frame, "competition metadata agrees with the registry")


def test_a_competition_with_no_tier_agrees_with_a_null_tier() -> None:
    """Cups have no tier. The comparison has to treat 'no tier' as a value, not
    as missing information."""
    frame = league_frame(competition_id="ENG_CUP", name="FA Cup", tier=None)
    registry = league_registry(competition_id="ENG_CUP", name="FA Cup", tier=None)
    assert failures(frame, registry) == set()


def test_a_match_outside_its_season_window_is_a_warning() -> None:
    """Measured on the real ingest: one row in 303,517, an Argentinian match
    played in 2015 and labelled 2013-14 in the provider's own file. Reported,
    not refused."""
    frame = league_frame()
    frame.loc[0, "date"] = pd.Timestamp("2019-01-01")
    frame = frame.sort_values(["date", "competition_id", "match_id"], kind="stable").reset_index(
        drop=True
    )
    report = run_checks(frame, match_checks(REGISTRY))
    stray = next(
        r
        for r in report.results
        if r.name == "matches fall inside the season they are labelled with"
    )
    assert stray.outcome is Outcome.FAILED
    assert (
        f"({frame.loc[frame['date'] == pd.Timestamp('2019-01-01'), 'competition_id'].iloc[0]})"
        in stray.message
    )
    assert report.ok


def test_the_pandemic_season_is_not_treated_as_an_error() -> None:
    """2019/20 finished in August 2020 in six competitions and the 2020
    Brasileirao finished in February 2021. A window tight enough to call those
    errors is a window that fires for no useful reason."""
    frame = league_frame(seasons=["2019-20"])
    frame.loc[frame.index[-1], "date"] = pd.Timestamp("2020-08-20")
    assert checks._registry_checks(REGISTRY)[2](frame).outcome is Outcome.PASSED

    calendar = league_frame(seasons=["2020"])
    calendar["date"] = pd.Timestamp("2021-02-25")
    assert checks._registry_checks(REGISTRY)[2](calendar).outcome is Outcome.PASSED


def test_collapsed_statistics_coverage_is_caught() -> None:
    """The other half of 'shots on target never exceed shots'. Nulling an
    impossible value is only correct while it stays rare, and only an aggregate
    can see that it stopped being."""
    frame = league_frame()
    recent = frame["date"] >= checks.RECENT_STATS_FROM
    frame.loc[recent, "home_shots"] = pd.NA
    assert_only(frame, "recent matches carry the statistics their feed promises")


def test_statistics_coverage_is_not_judged_on_too_few_recent_matches() -> None:
    """A competition mid-way through its first season has nothing to average."""
    frame = league_frame(seasons=season_labels(2000, 14))
    result = checks._registry_checks(REGISTRY)[3](frame)
    assert result.outcome is Outcome.PASSED


def test_a_competition_without_statistics_is_not_measured_for_coverage() -> None:
    """The secondary feed carries no shots at all, ever. A capability flag is a
    ceiling, and a competition below it is configuration, not corruption."""
    frame = league_frame(with_stats=False)
    registry = league_registry(feed=Feed.EXTRA)
    assert checks._registry_checks(registry)[3](frame).outcome is Outcome.PASSED


def test_a_competition_with_too_little_history_is_a_warning() -> None:
    """Switzerland's 'Challenge League' label was exactly this: two promotion
    play-off matches carried as a league."""
    big = league_frame()
    tiny = league_frame(
        competition_id="ENG_2",
        name="Championship",
        tier=2,
        seasons=["2020-21"],
        teams=4,
        team_prefix="Club",
    )
    frame = pd.concat([big, tiny], ignore_index=True).sort_values(
        ["date", "competition_id", "match_id"], kind="stable"
    )
    registry = Registry(
        competitions=(
            *REGISTRY.competitions,
            Competition(
                id="ENG_2",
                country="England",
                name="Championship",
                tier=2,
                feed=Feed.MAIN,
                code="E1",
            ),
        )
    )
    report = run_checks(frame.reset_index(drop=True), match_checks(registry))
    thin = next(
        r for r in report.results if r.name == "every competition has enough history to model"
    )
    assert thin.outcome is Outcome.FAILED
    assert "ENG_2" in thin.message
    assert report.ok


def test_metadata_drift_ignores_a_competition_that_is_not_registered() -> None:
    """Two checks, one problem: the unregistered competition is reported once,
    by the check that is about it."""
    frame = league_frame(competition_id="ITA_1", country="Italy", name="Serie A")
    assert checks._registry_checks(REGISTRY)[1](frame).outcome is Outcome.PASSED


# --- small tables ------------------------------------------------------------


def test_distribution_checks_are_skipped_on_a_single_season() -> None:
    """A single-competition ingest is a legitimate thing to do, and reporting
    a home-win rate over 380 matches as green would be a lie."""
    frame = league_frame(seasons=["2024-25"])
    report = run_checks(frame, match_checks(REGISTRY))
    skipped = {result.name for result in report.of(Outcome.SKIPPED)}
    assert "the home win rate is football" in skipped
    assert "the draw rate is football" in skipped
    assert report.ok


def test_an_empty_table_does_not_crash_the_suite() -> None:
    """The pipeline can legitimately produce nothing — one competition, offline
    provider — and a validation suite that raises there hides the real story."""
    empty = canonical_frame([])
    report = run_checks(empty, match_checks(REGISTRY))
    assert report.rows == 0
    assert report.ok
