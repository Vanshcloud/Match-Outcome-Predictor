"""Checks on the feature table.

The one worth reading is "no feature has a value before there is history": a
leakage check that works on the table rather than on the code. Every window
feature has a companion count, and a value where the count is zero came from
somewhere it should not have.
"""

from __future__ import annotations

import pandas as pd

from src.feature_engineering.registry import FEATURE_SCHEMA
from src.validation.features import feature_checks
from src.validation.report import Outcome, Severity, run_checks


def features(rows: int = 4, **overrides: object) -> pd.DataFrame:
    """A valid feature table, before whatever the test breaks."""
    base: dict[str, object] = {"match_id": [f"m{index}" for index in range(rows)]}
    for name in FEATURE_SCHEMA:
        if name == "match_id":
            continue
        if name.endswith("matches_played"):
            base[name] = [12] * rows
        elif name.endswith("matches_14d"):
            base[name] = [2] * rows
        elif name == "h2h_matches":
            base[name] = [4] * rows
        elif name.endswith("rest_days"):
            base[name] = [7] * rows
        elif "points" in name:
            base[name] = [1.5] * rows
        else:
            base[name] = [1.2] * rows
    return pd.DataFrame({**base, **overrides}).astype(FEATURE_SCHEMA)


def blank(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Null a column without changing its dtype.

    Assigning `pd.NA` to a whole column makes it `object`, which trips the
    schema check for a reason the test is not about.
    """
    out = frame.copy()
    for column in columns:
        out[column] = pd.array([None] * len(out), dtype=frame[column].dtype)
    return out


def failures(frame: pd.DataFrame, **kwargs: object) -> set[str]:
    report = run_checks(frame, feature_checks(**kwargs))  # type: ignore[arg-type]
    return {result.name for result in report.of(Outcome.FAILED)}


def test_a_valid_table_passes_everything() -> None:
    assert failures(features()) == set()


def test_an_unregistered_column_is_caught() -> None:
    """A column built but not registered is one nothing documents."""
    frame = features()
    frame["surprise"] = 1.0
    assert failures(frame) == {"the table is exactly the registered features"}


def test_a_missing_column_is_caught() -> None:
    assert "the table is exactly the registered features" in failures(
        features().drop(columns=["h2h_matches"])
    )


def test_a_wrong_dtype_is_caught() -> None:
    frame = features()
    frame["home_form_points_5"] = frame["home_form_points_5"].astype("float64")
    assert failures(frame) == {"the table is exactly the registered features"}


def test_a_value_with_no_history_behind_it_is_caught() -> None:
    """The leakage check. A form figure where the count is zero did not come
    from the team's previous matches, because there were none."""
    frame = features(home_matches_played=[0] * 4)  # values left in place
    assert failures(frame) == {"no feature has a value before there is history"}


def test_a_null_where_there_is_no_history_is_not_an_error() -> None:
    """Only the coverage warning fires, which is correct: this table really has
    no form figures, and that is worth saying rather than refusing it."""
    frame = blank(
        features(home_matches_played=[0] * 4),
        "home_form_points_5",
        "home_goals_for_5",
        "home_goals_against_5",
        "home_venue_points_5",
        "home_rest_days",
    )
    assert failures(frame) == {"most matches have a form figure"}


def test_a_head_to_head_figure_without_a_meeting_is_caught() -> None:
    frame = features(h2h_matches=[0] * 4)
    assert failures(frame) == {"no feature has a value before there is history"}


def test_a_null_count_is_caught() -> None:
    """A count of prior matches is always knowable — zero at worst — so a null
    means the walk skipped a row."""
    frame = features()
    frame.loc[0, "h2h_matches"] = pd.NA
    assert "counts are never null and never negative" in failures(frame)


def test_a_negative_count_is_caught() -> None:
    assert "counts are never null and never negative" in failures(
        features(home_matches_14d=[-1] * 4)
    )


def test_points_above_three_are_caught() -> None:
    assert "points per game never exceed three" in failures(features(home_form_points_5=[4.0] * 4))


def test_a_summed_goal_window_is_caught() -> None:
    """An error that leaves every value positive and finite, so nothing else
    would notice."""
    assert "goal rates are plausible" in failures(features(home_goals_for_5=[42.0] * 4))


def test_zero_rest_is_caught() -> None:
    """It would mean a team informed its own fixture: the window stops at the
    last strictly earlier date, so nothing can report zero."""
    assert "rest is measured in whole days, and at least one" in failures(
        features(home_rest_days=[0] * 4)
    )


def test_impossible_congestion_is_caught() -> None:
    assert "fortnight congestion is possible" in failures(features(home_matches_14d=[20] * 4))


def test_an_entirely_formless_table_is_a_warning() -> None:
    """A column that is null everywhere satisfies every rule above."""
    frame = blank(
        features(home_matches_played=[0] * 4),
        "home_form_points_5",
        "home_goals_for_5",
        "home_goals_against_5",
        "home_venue_points_5",
        "home_rest_days",
    )
    report = run_checks(frame, feature_checks())
    coverage = next(r for r in report.results if r.name == "most matches have a form figure")
    assert coverage.outcome is Outcome.FAILED
    assert coverage.severity is Severity.WARNING
    assert report.ok


def test_a_partial_build_is_not_judged_on_what_it_did_not_build() -> None:
    """`--builder head_to_head` leaves every form column null on purpose. A
    suite that called those nulls defects would fire on a legitimate command,
    which is how a report earns being ignored."""
    frame = blank(
        features(),
        "home_matches_played",
        "away_matches_played",
        "home_form_points_5",
        "away_form_points_5",
        "home_rest_days",
        "away_rest_days",
        "home_matches_14d",
        "away_matches_14d",
    )
    assert failures(frame) != set(), "a full build with null counts is a defect"
    assert failures(frame, built=["h2h_matches", "h2h_home_points"]) == set()


def test_a_missing_column_does_not_crash_the_rest_of_the_suite() -> None:
    """The schema check reports it; nothing else should raise trying to read
    it, or one broken column would hide every other finding."""
    frame = features().drop(columns=["home_rest_days", "home_matches_14d", "h2h_home_points"])
    reported = failures(frame)
    assert "the table is exactly the registered features" in reported


def test_an_empty_table_passes() -> None:
    assert failures(features(rows=0)) == set()


def test_features_must_cover_the_matches_exactly() -> None:
    frame = features()
    assert failures(frame, match_ids=[f"m{i}" for i in range(4)]) == set()
    assert failures(frame, match_ids=["m0"]) == {
        "every featured match exists, and every match is featured"
    }


def test_duplicate_feature_rows_are_caught() -> None:
    frame = pd.concat([features(), features().head(1)], ignore_index=True).astype(FEATURE_SCHEMA)
    assert "match ids are unique" in failures(frame, match_ids=[f"m{i}" for i in range(4)])
