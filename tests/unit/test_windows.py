"""The causal primitive every feature is built from.

One property carries the whole package: a window ends at the last row *strictly
earlier by date*, never at the previous row. The difference only shows up when
two rows share a date — which the obvious ``shift(1).rolling(k)`` gets wrong,
and which happens in real football often enough that the mislabelled-division
bug in `docs/DATA_SOURCES.md` was found by asking about it.

So most of these tests are about ties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.windows import (
    days_since_previous,
    group_slices,
    long_form,
    prior_counts,
    trailing_mean,
    trailing_within_days,
)
from tests.factories import canonical_frame, league_frame


def dates(*days: str) -> np.ndarray:
    return pd.to_datetime(list(days)).to_numpy()


# ---- prior_counts -----------------------------------------------------------


def test_each_row_counts_only_strictly_earlier_rows() -> None:
    assert list(prior_counts(dates("2020-01-01", "2020-01-02", "2020-01-03"))) == [0, 1, 2]


def test_rows_sharing_a_date_share_a_count() -> None:
    """The whole point. Neither of two matches on one day may see the other,
    whatever arbitrary order the sort put them in."""
    assert list(prior_counts(dates("2020-01-01", "2020-01-05", "2020-01-05", "2020-01-09"))) == [
        0,
        1,
        1,
        3,
    ]


def test_an_empty_input_produces_an_empty_result() -> None:
    assert len(prior_counts(dates())) == 0


# ---- trailing_mean ----------------------------------------------------------


def test_the_mean_covers_the_previous_window_rows() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    prior = prior_counts(
        dates("2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05")
    )
    result = trailing_mean(values, prior, 3)
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(1.0)
    assert result[2] == pytest.approx(1.5)
    assert result[4] == pytest.approx(3.0)  # 2, 3, 4


def test_fewer_than_a_full_window_is_still_answered() -> None:
    """A team that has played twice has form over two matches, not none."""
    values = np.array([3.0, 0.0])
    prior = prior_counts(dates("2020-01-01", "2020-01-02"))
    assert trailing_mean(values, prior, 5)[1] == pytest.approx(3.0)


def test_rows_sharing_a_date_do_not_feed_each_other() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    prior = prior_counts(dates("2020-01-01", "2020-01-05", "2020-01-05", "2020-01-09"))
    result = trailing_mean(values, prior, 5)
    assert result[1] == pytest.approx(1.0)
    assert result[2] == pytest.approx(1.0), "the second match of the day saw the first"
    assert result[3] == pytest.approx(2.0)  # 1, 2, 3


def test_missing_values_are_skipped_not_counted_as_zero() -> None:
    """ "No shots recorded in the last five" is not "zero shots", and a feature
    that conflated them would teach a model that competitions without shot data
    are famously bad at shooting."""
    values = np.array([10.0, np.nan, 20.0, np.nan])
    prior = prior_counts(dates("2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"))
    result = trailing_mean(values, prior, 5)
    assert result[2] == pytest.approx(10.0)
    assert result[3] == pytest.approx(15.0)


def test_a_window_with_nothing_in_it_is_null() -> None:
    values = np.array([np.nan, np.nan, 5.0])
    prior = prior_counts(dates("2020-01-01", "2020-01-02", "2020-01-03"))
    result = trailing_mean(values, prior, 5)
    assert np.isnan(result[1])
    assert np.isnan(result[2])


def test_the_window_forgets_beyond_its_length() -> None:
    values = np.array([9.0, 0.0, 0.0, 0.0])
    prior = prior_counts(dates("2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"))
    assert trailing_mean(values, prior, 2)[3] == pytest.approx(0.0)


# ---- trailing_within_days ---------------------------------------------------


def test_congestion_counts_earlier_matches_inside_the_window() -> None:
    stamps = dates("2020-01-01", "2020-01-05", "2020-01-10", "2020-01-30")
    prior = prior_counts(stamps)
    assert list(trailing_within_days(stamps, prior, 14)) == [0, 1, 2, 0]


def test_congestion_excludes_the_current_day() -> None:
    stamps = dates("2020-01-01", "2020-01-05", "2020-01-05")
    prior = prior_counts(stamps)
    assert list(trailing_within_days(stamps, prior, 14)) == [0, 1, 1]


def test_a_match_exactly_on_the_boundary_counts() -> None:
    stamps = dates("2020-01-01", "2020-01-15")
    prior = prior_counts(stamps)
    assert list(trailing_within_days(stamps, prior, 14)) == [0, 1]


# ---- days_since_previous ----------------------------------------------------


def test_rest_is_measured_from_the_last_match_played() -> None:
    stamps = dates("2020-01-01", "2020-01-08", "2020-01-11")
    prior = prior_counts(stamps)
    result = days_since_previous(stamps, prior)
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(7.0)
    assert result[2] == pytest.approx(3.0)


def test_two_matches_on_one_day_report_the_same_rest() -> None:
    """Not zero. The two are simultaneous, and neither preceded the other."""
    stamps = dates("2020-01-01", "2020-01-08", "2020-01-08")
    prior = prior_counts(stamps)
    result = days_since_previous(stamps, prior)
    assert result[1] == pytest.approx(7.0)
    assert result[2] == pytest.approx(7.0)


def test_a_first_appearance_has_no_rest() -> None:
    stamps = dates("2020-01-01")
    assert np.isnan(days_since_previous(stamps, prior_counts(stamps))[0])


# ---- long_form --------------------------------------------------------------


def test_every_match_becomes_two_rows() -> None:
    matches = league_frame(seasons=["2020-21"], teams=6)
    long = long_form(matches)
    assert len(long) == 2 * len(matches)
    assert set(long["venue"]) == {"home", "away"}


def test_each_row_is_written_from_its_own_team_s_point_of_view() -> None:
    matches = league_frame(seasons=["2020-21"], teams=6)
    long = long_form(matches).set_index(["match_id", "venue"])
    first = matches.iloc[0]
    home = long.loc[(first["match_id"], "home")]
    away = long.loc[(first["match_id"], "away")]
    assert home["team"] == first["home_team_id"]
    assert home["goals_for"] == away["goals_against"]
    assert home["points"] + away["points"] in {3.0, 2.0}


def test_points_follow_the_result() -> None:
    matches = league_frame(seasons=["2020-21"], teams=6)
    long = long_form(matches)
    won = long[long["goals_for"] > long["goals_against"]]
    drew = long[long["goals_for"] == long["goals_against"]]
    assert (won["points"] == 3.0).all()
    assert (drew["points"] == 1.0).all()


def test_a_match_with_no_score_earns_no_points() -> None:
    """Ingestion drops unplayed fixtures, so this only fires on a caller's own
    slice — but a nought there would be a loss the team never suffered."""
    matches = league_frame(seasons=["2020-21"], teams=6).copy()
    matches.loc[0, "home_goals"] = pd.NA
    assert long_form(matches)["points"].isna().sum() == 1


def test_the_long_frame_is_ordered_by_team_then_date() -> None:
    long = long_form(league_frame(seasons=["2020-21", "2021-22"], teams=6))
    assert long.sort_values(["team", "date", "match_id"], kind="stable").index.equals(long.index)


def test_an_empty_table_produces_an_empty_long_frame() -> None:
    assert long_form(canonical_frame([])).empty


# ---- group_slices -----------------------------------------------------------


def test_slices_cover_each_run_of_equal_keys() -> None:
    keys = pd.Series(["a", "a", "b", "c", "c", "c"])
    assert group_slices(keys) == [(0, 2), (2, 3), (3, 6)]


def test_a_single_group_is_one_slice() -> None:
    assert group_slices(pd.Series(["a", "a"])) == [(0, 2)]


def test_no_keys_means_no_slices() -> None:
    assert group_slices(pd.Series([], dtype="object")) == []
