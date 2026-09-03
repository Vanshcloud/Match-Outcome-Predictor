"""Head-to-head features.

The bookkeeping worth testing is the asymmetry: a pair's history is symmetric,
"points the home team took" is not, and the two are not related by any
arithmetic — a draw is one point for both, so one side's record cannot be
recovered by subtracting the other's from three.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering.head_to_head import HeadToHeadFeatures
from src.feature_engineering.registry import FeatureBuilder
from src.validation.temporal import outcome_independence, prefix_invariance
from tests.factories import canonical_frame, league_frame
from tests.unit.test_team_history import fixtures

BUILD = HeadToHeadFeatures().build
LEAGUE = league_frame(seasons=["2019-20", "2020-21", "2021-22"], teams=6)


def test_the_builder_satisfies_the_protocol() -> None:
    assert isinstance(HeadToHeadFeatures(), FeatureBuilder)


def test_the_output_is_one_row_per_match_in_input_order() -> None:
    built = BUILD(LEAGUE)
    assert list(built["match_id"]) == list(LEAGUE["match_id"])
    assert {c: str(d) for c, d in built.dtypes.items()} == {
        "match_id": "string",
        "h2h_matches": "Int16",
        "h2h_home_points": "Float64",
    }


def test_a_first_meeting_has_no_history() -> None:
    built = BUILD(fixtures([("2020-08-01", "alpha", "beta", 1, 0)]))
    assert built.loc[0, "h2h_matches"] == 0
    assert pd.isna(built.loc[0, "h2h_home_points"])


def test_the_reverse_fixture_shares_the_history() -> None:
    """The pair key is order-independent, so a fixture and its reverse are the
    same rivalry."""
    matches = fixtures(
        [("2020-08-01", "alpha", "beta", 2, 0), ("2020-12-01", "beta", "alpha", 0, 0)]
    )
    built = BUILD(matches)
    assert built.loc[1, "h2h_matches"] == 1


def test_the_points_belong_to_this_match_s_home_team() -> None:
    """Alpha beat beta at home. In the return fixture beta is at home, and
    beta's record against alpha is nought — not three minus alpha's."""
    matches = fixtures(
        [("2020-08-01", "alpha", "beta", 2, 0), ("2020-12-01", "beta", "alpha", 1, 1)]
    )
    built = BUILD(matches)
    assert built.loc[1, "h2h_home_points"] == pytest.approx(0.0)


def test_a_draw_gives_both_sides_a_point() -> None:
    """The case that rules out recovering one side's record from the other's."""
    matches = fixtures(
        [
            ("2020-08-01", "alpha", "beta", 1, 1),
            ("2020-12-01", "beta", "alpha", 0, 0),
            ("2021-04-01", "alpha", "beta", 0, 0),
        ]
    )
    built = BUILD(matches)
    assert built.loc[1, "h2h_home_points"] == pytest.approx(1.0)
    assert built.loc[2, "h2h_home_points"] == pytest.approx(1.0)


def test_the_record_averages_over_previous_meetings() -> None:
    matches = fixtures(
        [
            ("2020-08-01", "alpha", "beta", 3, 0),
            ("2020-12-01", "beta", "alpha", 2, 0),
            ("2021-04-01", "alpha", "beta", 0, 0),
        ]
    )
    built = BUILD(matches)
    # alpha won at home, lost away: 3 + 0 from two meetings.
    assert built.loc[2, "h2h_home_points"] == pytest.approx(1.5)


def test_other_pairs_do_not_contribute() -> None:
    matches = fixtures(
        [
            ("2020-08-01", "alpha", "gamma", 3, 0),
            ("2020-09-01", "beta", "gamma", 3, 0),
            ("2020-12-01", "alpha", "beta", 0, 0),
        ]
    )
    built = BUILD(matches)
    assert built.loc[2, "h2h_matches"] == 0
    assert pd.isna(built.loc[2, "h2h_home_points"])


def test_meetings_on_one_day_do_not_feed_each_other() -> None:
    matches = fixtures(
        [
            ("2020-08-01", "alpha", "beta", 3, 0),
            ("2020-12-01", "alpha", "beta", 0, 3),
            ("2020-12-01", "beta", "alpha", 3, 0),
        ]
    )
    built = BUILD(matches)
    assert built.loc[1, "h2h_matches"] == built.loc[2, "h2h_matches"] == 1
    assert built.loc[1, "h2h_home_points"] == pytest.approx(3.0)
    assert built.loc[2, "h2h_home_points"] == pytest.approx(0.0)


def test_a_match_with_no_score_contributes_no_points() -> None:
    matches = fixtures(
        [("2020-08-01", "alpha", "beta", 1, 0), ("2020-12-01", "alpha", "beta", 0, 0)]
    ).copy()
    matches.loc[0, "home_goals"] = pd.NA
    built = BUILD(matches)
    assert built.loc[1, "h2h_matches"] == 1
    assert pd.isna(built.loc[1, "h2h_home_points"])


def test_an_empty_table_produces_an_empty_frame() -> None:
    built = BUILD(canonical_frame([]))
    assert built.empty
    assert "h2h_matches" in built.columns


def test_the_builder_is_prefix_invariant() -> None:
    assert prefix_invariance(BUILD, LEAGUE, name="head_to_head").ok


def test_the_builder_does_not_read_the_meeting_it_describes() -> None:
    assert outcome_independence(BUILD, LEAGUE, name="head_to_head").ok
