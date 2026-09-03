"""The ratings pipeline coordinates; it must not compute.

The property worth protecting is the one that would otherwise be a claim in a
README: the build runs the causality probes and *fails* when one does not hold.
So the suite includes a deliberately leaky model, and asserts the pipeline
catches it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.manifest import read_manifest, verify_manifest
from src.pipelines.ratings import (
    RATINGS_FILENAME,
    build_ratings,
    choose_verification_sample,
    default_models,
    run_ratings,
)
from src.ratings.base import ELO_COLUMNS, RATINGS_SCHEMA, RatingError
from src.ratings.dixon_coles import DEFAULT as DC_DEFAULT
from src.ratings.dixon_coles import DixonColesRatings
from src.ratings.elo import EloRatings
from tests.factories import canonical_frame, league_frame

FAST_DC = DixonColesRatings(
    replace(DC_DEFAULT, window_days=540, refit_days=60, min_matches=60, min_teams=6)
)
MODELS = (EloRatings(), FAST_DC)

LEAGUE = league_frame(seasons=["2018-19", "2019-20", "2020-21", "2021-22"], teams=10)


class LeakyRatings:
    """A model that reads the match it is rating. Deliberately."""

    name = "leaky"
    feature_columns = tuple(ELO_COLUMNS)

    def rate(self, matches: pd.DataFrame) -> pd.DataFrame:
        goals = matches["home_goals"].astype("float64").to_numpy()
        return pd.DataFrame(
            {
                "match_id": matches["match_id"].to_numpy(),
                "elo_home": 1500.0 + 100.0 * goals,
                "elo_away": 1500.0,
                "elo_expected_home": 0.5,
                "elo_home_played": 1,
                "elo_away_played": 1,
            }
        ).astype({"match_id": "string", **ELO_COLUMNS})


# ---- assembly ---------------------------------------------------------------


def test_the_table_has_one_row_per_match_in_input_order() -> None:
    built = build_ratings(LEAGUE, MODELS)
    assert list(built["match_id"]) == list(LEAGUE["match_id"])
    assert {c: str(d) for c, d in built.dtypes.items()} == RATINGS_SCHEMA


def test_a_partial_build_widens_the_same_schema() -> None:
    """A consumer should not have to discover which columns exist before it can
    read any of them."""
    built = build_ratings(LEAGUE, (EloRatings(),))
    assert list(built.columns) == list(RATINGS_SCHEMA)
    assert built["dc_prob_home"].isna().all()
    assert built["elo_home"].notna().all()


def test_an_unsorted_table_is_refused() -> None:
    with pytest.raises(RatingError, match="sorted by date"):
        build_ratings(LEAGUE.sort_values("match_id", ascending=False), MODELS)


def test_the_default_models_are_elo_and_dixon_coles() -> None:
    assert [model.name for model in default_models()] == ["elo", "dixon_coles"]


# ---- verification -----------------------------------------------------------


def test_the_sample_is_the_competition_closest_to_the_target() -> None:
    """Not the smallest available: a sample too thin for Dixon-Coles to fit at
    all would pass both probes trivially, which is the one way a leakage check
    can be worse than useless."""
    small = league_frame(seasons=["2020-21"], teams=6)
    big = league_frame(
        competition_id="ESP_1",
        country="Spain",
        name="La Liga",
        seasons=["2018-19", "2019-20", "2020-21"],
        teams=12,
    )
    together = pd.concat([small, big], ignore_index=True).sort_values("date", kind="stable")
    assert choose_verification_sample(together, target=30) == "ENG_1"
    assert choose_verification_sample(together, target=10_000) == "ESP_1"


def test_no_sample_from_an_empty_table() -> None:
    assert choose_verification_sample(canonical_frame([])) is None


def test_a_clean_build_reports_causal(tmp_path: Path) -> None:
    report = run_ratings(LEAGUE, tmp_path, models=MODELS)
    assert report.causal
    assert report.verified_on == "ENG_1"
    assert len(report.temporal) == 4  # two probes per model


def test_a_leaky_model_fails_the_build(tmp_path: Path) -> None:
    """The claim this pipeline exists to make. A rating that can see its own
    match does not produce a worse model — it produces a better-looking one."""
    report = run_ratings(LEAGUE, tmp_path, models=(LeakyRatings(),))
    assert not report.causal
    assert any(
        "outcome independence" in result.probe for result in report.temporal if not result.ok
    )


def test_verification_can_be_skipped(tmp_path: Path) -> None:
    report = run_ratings(LEAGUE, tmp_path, models=MODELS, verify=False)
    assert report.temporal == ()
    assert report.verified_on is None
    assert report.causal, "no probe ran, so there is nothing to have failed"


# ---- reporting and output ---------------------------------------------------


def test_coverage_is_reported_per_model(tmp_path: Path) -> None:
    report = run_ratings(LEAGUE, tmp_path, models=MODELS, verify=False)
    assert report.coverage["elo"] == 1.0
    assert 0.0 < report.coverage["dixon_coles"] < 1.0


def test_the_elo_error_is_reported(tmp_path: Path) -> None:
    report = run_ratings(LEAGUE, tmp_path, models=MODELS, verify=False)
    assert report.elo_mse is not None
    assert 0.0 < report.elo_mse < 0.5


def test_the_output_and_its_manifest_verify(tmp_path: Path) -> None:
    report = run_ratings(LEAGUE, tmp_path, models=MODELS, verify=False)
    assert report.output == tmp_path / RATINGS_FILENAME
    written = pd.read_parquet(report.output)
    assert len(written) == len(LEAGUE)

    manifest_path = report.output.with_suffix(".manifest.json")
    manifest = read_manifest(manifest_path)
    assert manifest["kind"] == "ratings"
    assert manifest["models"] == ["elo", "dixon_coles"]
    assert verify_manifest(manifest, tmp_path).ok


def test_validation_is_reported_not_raised(tmp_path: Path) -> None:
    """Same reasoning as ingestion: a table you can inspect beats one the
    pipeline refused to save."""

    class ImpossibleRatings(LeakyRatings):
        """Emits an expected score of exactly 1, which is not a score."""

        name = "impossible"

        def rate(self, matches: pd.DataFrame) -> pd.DataFrame:
            rated = super().rate(matches)
            rated["elo_expected_home"] = 1.0
            return rated

    report = run_ratings(LEAGUE, tmp_path, models=(ImpossibleRatings(),), verify=False)
    assert report.validation is not None
    assert not report.validation.ok
    assert "the expected score is a score" in {r.name for r in report.validation.blocking}
    assert (tmp_path / RATINGS_FILENAME).is_file()


def test_the_summary_names_what_was_verified(tmp_path: Path) -> None:
    """'Verified' and 'verified on ENG_1' are different claims."""
    report = run_ratings(LEAGUE, tmp_path, models=(EloRatings(),))
    assert "causality ok on ENG_1" in report.summary()
    assert "elo 100.0%" in report.summary()


def test_an_empty_table_produces_an_empty_build(tmp_path: Path) -> None:
    report = run_ratings(canonical_frame([]), tmp_path, models=MODELS)
    assert report.matches == 0
    assert report.coverage == {"elo": 0.0, "dixon_coles": 0.0}
    assert report.elo_mse is None
    assert report.verified_on is None
