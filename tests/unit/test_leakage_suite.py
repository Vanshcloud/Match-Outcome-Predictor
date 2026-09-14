"""The leakage suite: every producer, probed, on every run of the test suite.

The ratings and features pipelines each probe their own producers.
This file probes whatever *exists*, discovered by walking the packages, which
is the difference between "every builder we remembered" and "every builder".
It is a unit test rather than a shell step in the workflow on purpose: CI runs
the suite, so a producer added on a branch is probed by the same command the
author already runs locally.

The fixture is two competitions in two countries, because several of the
answers are only visible when the column varies. A single-competition sample
cannot show that Dixon-Coles partitions on `competition_id`, and would report
that it does not read it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering.registry import FEATURE_COLUMNS
from src.pipelines.features import default_builders
from src.pipelines.ratings import default_models
from src.ratings.base import RATINGS_SCHEMA
from src.utils.paths import PROJECT_ROOT
from src.validation.leakage import (
    KEY_COLUMN,
    Producer,
    audit,
    benchmark_leaks,
    check_defaults_are_complete,
    producers,
    run_suite,
)
from src.validation.temporal import observed_reads, split_boundary
from tests.factories import league_frame

SEASONS = ("2018-19", "2019-20", "2020-21")


def _two_countries() -> pd.DataFrame:
    england = league_frame(seasons=SEASONS, teams=8)
    spain = league_frame(
        competition_id="ESP_1",
        country="Spain",
        name="La Liga",
        seasons=SEASONS,
        teams=8,
        team_prefix="Club",
    )
    return (
        pd.concat([england, spain], ignore_index=True)
        .sort_values(["date", "competition_id", "match_id"], kind="stable")
        .reset_index(drop=True)
    )


LEAGUE = _two_countries()
FOUND = producers()
AUDIT = audit(LEAGUE, FOUND)

MODEL_VISIBLE = set(FEATURE_COLUMNS) | (set(RATINGS_SCHEMA) - {KEY_COLUMN})


# ---- discovery --------------------------------------------------------------


def test_every_producer_in_the_codebase_is_found() -> None:
    assert {producer.name for producer in FOUND} == {
        "elo",
        "dixon_coles",
        "team_history",
        "head_to_head",
    }


def test_the_contracts_themselves_are_not_mistaken_for_producers() -> None:
    """`RatingModel` has a `rate` and `FeatureBuilder` has a `build`.

    Both are Protocols, and instantiating one raises. Discovering them would
    turn the whole suite red for a reason that has nothing to do with leakage.
    """
    assert "RatingModel" not in {producer.name for producer in FOUND}


def test_every_found_producer_is_in_a_pipeline_default() -> None:
    """A producer nobody runs is a column the model layer will never see."""
    defaults = [model.name for model in default_models()]
    defaults += [builder.name for builder in default_builders()]
    assert check_defaults_are_complete(defaults) == ()


def test_a_producer_missing_from_the_defaults_is_named() -> None:
    assert check_defaults_are_complete(["elo"]) == (
        "dixon_coles",
        "head_to_head",
        "team_history",
    )


# ---- the probes, over everything -------------------------------------------


@pytest.mark.parametrize("producer", FOUND, ids=lambda producer: producer.name)
def test_every_producer_survives_both_temporal_probes(producer: Producer) -> None:
    for result in run_suite(LEAGUE, [producer]):
        assert result.ok, result.summary()


def test_the_suite_runs_two_probes_per_producer() -> None:
    assert len(run_suite(LEAGUE, FOUND)) == 2 * len(FOUND)


# ---- what each column actually reads ---------------------------------------


def test_every_derived_column_is_traced() -> None:
    assert {row.column for row in AUDIT} == MODEL_VISIBLE


@pytest.mark.parametrize("row", AUDIT, ids=lambda row: row.column)
def test_no_feature_reads_more_than_it_declares(row: object) -> None:
    """Over-declaring is safe; under-declaring reclassifies a leak as safe."""
    assert not row.understated, f"{row.column} reads undeclared {sorted(row.understated)}"  # type: ignore[attr-defined]


def test_every_form_column_admits_reading_a_result() -> None:
    """The columns with something to prove are the ones the probes then prove.

    A form feature whose measured inputs contained no post-match column would
    not be a safe feature; it would be a feature that is not computing what it
    says it computes.
    """
    forms = [row for row in AUDIT if row.column.endswith("_form_points_5")]
    assert forms and all(row.post_match for row in forms)


def test_a_count_of_fixtures_reads_no_result_at_all() -> None:
    played = next(row for row in AUDIT if row.column == "home_matches_played")
    assert not played.post_match


def test_nothing_reads_the_bookmakers_price() -> None:
    """Odds are the benchmark. A feature reading them compares the project to itself."""
    assert benchmark_leaks(AUDIT) == ()


def test_a_producer_reading_the_odds_is_caught() -> None:
    def cheat(matches: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {KEY_COLUMN: matches[KEY_COLUMN], "tip": matches["odds_home"].to_numpy()}
        )

    planted = Producer(
        name="cheat", kind="feature", compute=cheat, outputs=("tip",), declared_reads=None
    )
    assert benchmark_leaks(audit(LEAGUE, [planted], columns=("odds_home",))) == ("tip",)


def test_observed_reads_finds_a_dependence_the_declaration_would_hide() -> None:
    def copies_the_score(matches: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {KEY_COLUMN: matches[KEY_COLUMN], "leaked": matches["home_goals"].to_numpy()}
        )

    measured = observed_reads(
        copies_the_score, LEAGUE, ("home_goals", "away_goals"), key=KEY_COLUMN
    )
    assert measured["leaked"] == frozenset({"home_goals"})


def test_a_column_that_is_entirely_null_cannot_be_perturbed() -> None:
    """No perturbation exists, so nothing is claimed about it either way."""
    blank = LEAGUE.assign(referee=pd.NA)
    measured = observed_reads(
        lambda matches: pd.DataFrame(
            {KEY_COLUMN: matches[KEY_COLUMN], "anything": matches["referee"].to_numpy()}
        ),
        blank,
        ("referee",),
        key=KEY_COLUMN,
    )
    assert measured["anything"] == frozenset()


# ---- the split boundary -----------------------------------------------------


def _halves(frame: pd.DataFrame, at: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return frame.iloc[:at], frame.iloc[at:]


def test_a_split_on_a_date_gap_holds() -> None:
    cutoff = pd.Timestamp("2020-01-01")
    train = LEAGUE[LEAGUE["date"] < cutoff]
    evaluate = LEAGUE[LEAGUE["date"] >= cutoff]
    assert split_boundary(train, evaluate, name="chronological").ok


def test_a_shuffled_split_is_caught() -> None:
    shuffled = LEAGUE.sample(frac=1.0, random_state=0)
    train, evaluate = _halves(shuffled, len(shuffled) // 2)
    result = split_boundary(train, evaluate, name="shuffled")
    assert not result.ok
    assert "training runs to" in result.violations[0]


def test_a_split_that_cuts_through_a_matchday_is_caught() -> None:
    """Ties fail: the 3pm results are not available to the 5.30 kick-off."""
    same_day = LEAGUE[LEAGUE["date"] == LEAGUE["date"].iloc[100]]
    assert len(same_day) > 1
    train, evaluate = _halves(same_day, 1)
    assert not split_boundary(train, evaluate, name="matchday").ok


def test_a_match_in_both_halves_is_caught() -> None:
    train = LEAGUE.iloc[:100]
    evaluate = LEAGUE.iloc[99:]
    result = split_boundary(train, evaluate, name="overlapping")
    assert not result.ok
    assert any("in both halves" in violation for violation in result.violations)


def test_an_empty_half_leaves_nothing_to_compare() -> None:
    assert split_boundary(LEAGUE, LEAGUE.iloc[:0], name="empty").ok


def test_the_split_probe_reports_its_own_name() -> None:
    assert "split boundary" in split_boundary(LEAGUE, LEAGUE.iloc[:0], name="empty").summary()


# ---- the written audit ------------------------------------------------------


def test_the_documented_audit_names_every_model_visible_column() -> None:
    """The doc is generated from the same call. This is what stops it rotting."""
    written = (PROJECT_ROOT / "docs" / "LEAKAGE.md").read_text(encoding="utf-8")
    missing = sorted(column for column in MODEL_VISIBLE if f"`{column}`" not in written)
    assert not missing, f"docs/LEAKAGE.md does not mention: {missing}"
