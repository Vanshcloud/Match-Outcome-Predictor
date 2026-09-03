"""Form and schedule features.

The arithmetic is checked against a hand-built fixture where the right answer
can be worked out on paper, and the causality against the probes. The tests
that matter most are the ones about *what a window is allowed to see*: its own
match, a match later the same day, and a match in another competition are three
separate questions with three different right answers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering.registry import FeatureBuilder
from src.feature_engineering.team_history import TeamHistoryFeatures
from src.validation.temporal import outcome_independence, prefix_invariance
from tests.factories import canonical_frame, league_frame

BUILD = TeamHistoryFeatures().build


def fixtures(
    rows: list[tuple[str, str, str, int, int]], competition: str = "ENG_1"
) -> pd.DataFrame:
    """Matches from (date, home, away, home_goals, away_goals) tuples."""
    records = []
    for index, (day, home, away, home_goals, away_goals) in enumerate(rows):
        records.append(
            {
                "match_id": f"{competition}-{index:03d}",
                "provider": "stub",
                "competition_id": competition,
                "country": "England",
                "competition": "Premier League",
                "tier": 1,
                "season": "2020-21",
                "date": pd.Timestamp(day),
                "home_team": home,
                "away_team": away,
                "home_team_id": f"eng:{home}",
                "away_team_id": f"eng:{away}",
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": (
                    "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
                ),
            }
        )
    frame = canonical_frame(records)
    return frame.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)


LEAGUE = league_frame(seasons=["2019-20", "2020-21", "2021-22"], teams=8)


# ---- the contract -----------------------------------------------------------


def test_the_builder_satisfies_the_protocol() -> None:
    assert isinstance(TeamHistoryFeatures(), FeatureBuilder)


def test_the_output_is_one_row_per_match_in_input_order() -> None:
    built = BUILD(LEAGUE)
    assert list(built["match_id"]) == list(LEAGUE["match_id"])


def test_the_dtypes_come_from_the_registry() -> None:
    built = BUILD(LEAGUE)
    for feature in TeamHistoryFeatures.features:
        assert str(built[feature.name].dtype) == feature.dtype, feature.name


def test_an_empty_table_produces_an_empty_frame() -> None:
    built = BUILD(canonical_frame([]))
    assert built.empty
    assert "home_form_points_5" in built.columns


# ---- what a window may see --------------------------------------------------


def test_a_first_appearance_has_no_history() -> None:
    """A team with no previous matches has no form. Calling that a form of zero
    would teach a model that every debutant is terrible."""
    built = BUILD(fixtures([("2020-08-01", "alpha", "beta", 1, 0)]))
    assert built.loc[0, "home_matches_played"] == 0
    assert pd.isna(built.loc[0, "home_form_points_5"])
    assert pd.isna(built.loc[0, "home_rest_days"])
    assert built.loc[0, "home_matches_14d"] == 0


def test_form_is_the_mean_of_previous_results() -> None:
    """Alpha wins, draws, then loses. Going into the fourth match it has taken
    3 + 1 + 0 = 4 points from 3 games."""
    matches = fixtures(
        [
            ("2020-08-01", "alpha", "beta", 2, 0),
            ("2020-08-08", "alpha", "gamma", 1, 1),
            ("2020-08-15", "alpha", "delta", 0, 2),
            ("2020-08-22", "alpha", "beta", 0, 0),
        ]
    )
    built = BUILD(matches)
    assert built.loc[3, "home_matches_played"] == 3
    assert built.loc[3, "home_form_points_5"] == pytest.approx(4 / 3)
    assert built.loc[3, "home_goals_for_5"] == pytest.approx(1.0)
    assert built.loc[3, "home_goals_against_5"] == pytest.approx(1.0)


def test_a_match_never_sees_itself() -> None:
    """Going into its second match, alpha's form is its first result alone."""
    matches = fixtures(
        [("2020-08-01", "alpha", "beta", 3, 0), ("2020-08-08", "alpha", "gamma", 0, 3)]
    )
    built = BUILD(matches)
    assert built.loc[1, "home_form_points_5"] == pytest.approx(3.0)


def test_two_matches_on_one_day_do_not_feed_each_other() -> None:
    """A league fixture and a cup tie on the same date. Neither preceded the
    other, and a row-based shift would let the first inform the second."""
    matches = fixtures(
        [
            ("2020-08-01", "alpha", "beta", 3, 0),
            ("2020-08-08", "alpha", "gamma", 0, 3),
            ("2020-08-08", "alpha", "delta", 0, 3),
        ]
    )
    built = BUILD(matches)
    assert built.loc[1, "home_form_points_5"] == pytest.approx(3.0)
    assert built.loc[2, "home_form_points_5"] == pytest.approx(3.0)
    assert built.loc[1, "home_matches_played"] == built.loc[2, "home_matches_played"] == 1


def test_form_counts_matches_in_other_competitions() -> None:
    """A club's last five are its last five, league or cup. A per-competition
    window restarts a promoted team at zero and pretends the midweek tie never
    happened."""
    league = fixtures([("2020-08-01", "alpha", "beta", 3, 0)], competition="ENG_1")
    cup = fixtures([("2020-08-05", "alpha", "gamma", 2, 0)], competition="ENG_2")
    cup = cup.assign(competition="Championship", tier=pd.array([2] * len(cup), dtype="Int8"))
    later = fixtures([("2020-08-10", "alpha", "delta", 0, 0)], competition="ENG_1")
    later["match_id"] = "later-000"
    matches = (
        pd.concat([league, cup, later], ignore_index=True)
        .sort_values(["date", "match_id"], kind="stable")
        .reset_index(drop=True)
    )
    built = BUILD(matches)
    assert built.loc[2, "home_matches_played"] == 2
    assert built.loc[2, "home_form_points_5"] == pytest.approx(3.0)


def test_the_window_forgets_beyond_five_matches() -> None:
    rows = [("2020-08-01", "alpha", "beta", 3, 0)]
    rows += [(f"2020-09-{day:02d}", "alpha", "gamma", 0, 1) for day in range(1, 7)]
    built = BUILD(fixtures(rows))
    assert built.loc[6, "home_form_points_5"] == pytest.approx(0.0), "an old win still counted"


# ---- venue form -------------------------------------------------------------


def test_venue_form_only_counts_matches_at_that_venue() -> None:
    """Alpha wins at home and loses away. Its home form is 3, its overall form
    is 1.5, and a single figure would average exactly that signal away."""
    matches = fixtures(
        [
            ("2020-08-01", "alpha", "beta", 2, 0),
            ("2020-08-08", "gamma", "alpha", 2, 0),
            ("2020-08-15", "alpha", "delta", 0, 0),
        ]
    )
    built = BUILD(matches)
    assert built.loc[2, "home_venue_points_5"] == pytest.approx(3.0)
    assert built.loc[2, "home_form_points_5"] == pytest.approx(1.5)


def test_away_venue_form_is_the_away_team_s_away_record() -> None:
    matches = fixtures(
        [
            ("2020-08-01", "beta", "alpha", 0, 3),
            ("2020-08-08", "alpha", "gamma", 0, 3),
            ("2020-08-15", "delta", "alpha", 0, 0),
        ]
    )
    built = BUILD(matches)
    assert built.loc[2, "away_venue_points_5"] == pytest.approx(3.0)


def test_venue_form_is_null_before_a_team_has_played_at_that_venue() -> None:
    matches = fixtures(
        [("2020-08-01", "beta", "alpha", 0, 3), ("2020-08-08", "alpha", "gamma", 1, 0)]
    )
    built = BUILD(matches)
    assert pd.isna(built.loc[1, "home_venue_points_5"])
    assert built.loc[1, "home_matches_played"] == 1


def test_venue_form_is_aligned_to_the_right_rows() -> None:
    """It is computed on a second ordering and written back by position, which
    is exactly the operation that misaligns a column if it is done carelessly."""
    built = BUILD(LEAGUE)
    warmed = built[built["home_matches_played"] > 10]
    assert warmed["home_venue_points_5"].notna().all()
    assert warmed["home_venue_points_5"].between(0, 3).all()


# ---- schedule ---------------------------------------------------------------


def test_rest_is_days_since_the_team_last_played() -> None:
    matches = fixtures(
        [("2020-08-01", "alpha", "beta", 1, 0), ("2020-08-08", "gamma", "alpha", 0, 1)]
    )
    built = BUILD(matches)
    assert built.loc[1, "away_rest_days"] == 7


def test_congestion_counts_the_previous_fortnight() -> None:
    rows = [
        ("2020-08-01", "alpha", "beta", 1, 0),
        ("2020-08-05", "alpha", "gamma", 1, 0),
        ("2020-08-09", "alpha", "delta", 1, 0),
        ("2020-09-20", "alpha", "beta", 1, 0),
    ]
    built = BUILD(fixtures(rows))
    assert list(built["home_matches_14d"]) == [0, 1, 2, 0]


def test_shots_are_null_where_the_feed_carries_none() -> None:
    """Most of the secondary feed, and every season before 2000/01."""
    matches = league_frame(seasons=["2019-20", "2020-21"], teams=6, with_stats=False)
    built = BUILD(matches)
    assert built["home_shots_for_5"].isna().all()
    assert built["home_form_points_5"].notna().any()


# ---- causality --------------------------------------------------------------


def test_the_builder_is_prefix_invariant() -> None:
    assert prefix_invariance(BUILD, LEAGUE, name="team_history").ok


def test_the_builder_does_not_read_the_match_it_describes() -> None:
    assert outcome_independence(BUILD, LEAGUE, name="team_history").ok


def test_rewriting_a_result_does_change_the_team_s_next_form() -> None:
    """The other half. A builder that ignored results entirely would pass
    outcome independence trivially.

    The comparison is against that team's *next* appearance, not the last row
    of the table: a five-match window has long forgotten a result 160 fixtures
    later, so asserting on the tail would pass for the wrong reason.
    """
    position = 5
    team = LEAGUE.loc[position, "home_team_id"]
    plays_again = LEAGUE.index[
        (LEAGUE.index > position)
        & ((LEAGUE["home_team_id"] == team) | (LEAGUE["away_team_id"] == team))
    ][0]

    rewritten = LEAGUE.copy()
    rewritten.loc[position, "home_goals"] = 9
    rewritten.loc[position, "result"] = "H"

    before = BUILD(LEAGUE).loc[plays_again]
    after = BUILD(rewritten).loc[plays_again]
    assert not before.equals(after)
