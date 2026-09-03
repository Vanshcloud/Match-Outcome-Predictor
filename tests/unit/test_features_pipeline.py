"""The feature pipeline, and the shared machinery under it.

Same shape as the ratings pipeline, and for the same reason: the property worth
protecting is that the build *fails* when a producer can see the future, so the
suite includes a deliberately leaky builder and asserts it is caught.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.feature_engineering.head_to_head import HeadToHeadFeatures
from src.feature_engineering.registry import FEATURE_SCHEMA, FEATURES, Feature
from src.ingestion.manifest import read_manifest, verify_manifest
from src.pipelines.derived import DerivedReport, choose_verification_sample
from src.pipelines.features import (
    FEATURES_FILENAME,
    build_features,
    default_builders,
    run_features,
)
from src.ratings.base import RatingError
from tests.factories import canonical_frame, league_frame

LEAGUE = league_frame(seasons=["2018-19", "2019-20", "2020-21", "2021-22"], teams=10)


class LeakyFeatures:
    """A builder that reads the match it is describing. Deliberately."""

    name = "leaky"
    features: tuple[Feature, ...] = tuple(
        feature for feature in FEATURES if feature.group == "head_to_head"
    )

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        # A record with no meetings behind it, taken from this very match. It
        # fails both defences at once: the probes catch that it read its own
        # result, and the table check catches a value with no history.
        goals = matches["home_goals"].astype("Float64").clip(upper=3.0)
        return pd.DataFrame(
            {
                "match_id": matches["match_id"].to_numpy(),
                "h2h_matches": pd.array([0] * len(matches), dtype="Int16"),
                "h2h_home_points": goals.to_numpy(),
            }
        ).astype({"match_id": "string", "h2h_matches": "Int16", "h2h_home_points": "Float64"})


# ---- assembly ---------------------------------------------------------------


def test_the_table_has_one_row_per_match_in_input_order() -> None:
    built = build_features(LEAGUE)
    assert list(built["match_id"]) == list(LEAGUE["match_id"])
    assert {c: str(d) for c, d in built.dtypes.items()} == FEATURE_SCHEMA


def test_a_partial_build_widens_the_same_schema() -> None:
    """A consumer should not have to discover which columns exist before it can
    read any of them."""
    built = build_features(LEAGUE, (HeadToHeadFeatures(),))
    assert list(built.columns) == list(FEATURE_SCHEMA)
    assert built["home_form_points_5"].isna().all()
    assert built["h2h_matches"].notna().all()


def test_an_unsorted_table_is_refused() -> None:
    with pytest.raises(RatingError, match="sorted by date"):
        build_features(LEAGUE.sort_values("match_id", ascending=False))


def test_the_default_builders_cover_every_registered_feature() -> None:
    """A feature registered but never built is a promise to a consumer that
    nothing keeps."""
    produced = {feature.name for builder in default_builders() for feature in builder.features}
    assert produced == {feature.name for feature in FEATURES}


def test_no_two_builders_claim_the_same_feature() -> None:
    claimed = [feature.name for builder in default_builders() for feature in builder.features]
    assert len(claimed) == len(set(claimed))


# ---- verification -----------------------------------------------------------


def test_a_clean_build_reports_causal(tmp_path: Path) -> None:
    report = run_features(LEAGUE, tmp_path)
    assert report.causal
    assert report.verified_on == "ENG_1"
    assert len(report.temporal) == 4  # two probes per builder


def test_a_leaky_builder_fails_the_build(tmp_path: Path) -> None:
    """The claim this pipeline exists to make."""
    report = run_features(LEAGUE, tmp_path, builders=(LeakyFeatures(),))
    assert not report.causal
    assert any(not result.ok and "outcome" in result.probe for result in report.temporal)


def test_verification_can_be_skipped(tmp_path: Path) -> None:
    report = run_features(LEAGUE, tmp_path, verify=False)
    assert report.temporal == ()
    assert report.verified_on is None
    assert report.causal, "no probe ran, so there is nothing to have failed"


# ---- reporting and output ---------------------------------------------------


def test_the_report_counts_the_features_built(tmp_path: Path) -> None:
    report = run_features(LEAGUE, tmp_path, verify=False)
    assert report.features == len(FEATURES)
    assert "20 features" in report.summary()


def test_coverage_is_reported_per_builder(tmp_path: Path) -> None:
    report = run_features(LEAGUE, tmp_path, verify=False)
    assert report.coverage["team_history"] == 1.0
    assert report.coverage["head_to_head"] == 1.0


def test_the_output_and_its_manifest_verify(tmp_path: Path) -> None:
    report = run_features(LEAGUE, tmp_path, verify=False)
    assert report.output == tmp_path / FEATURES_FILENAME
    assert len(pd.read_parquet(report.output)) == len(LEAGUE)

    manifest = read_manifest(report.output.with_suffix(".manifest.json"))
    assert manifest["kind"] == "features"
    assert manifest["builders"] == ["team_history", "head_to_head"]
    assert verify_manifest(manifest, tmp_path).ok


def test_validation_is_reported_not_raised(tmp_path: Path) -> None:
    report = run_features(LEAGUE, tmp_path, builders=(LeakyFeatures(),), verify=False)
    assert report.validation is not None
    assert not report.validation.ok, "the leaky builder invents a record with no meetings"
    assert "no feature has a value before there is history" in {
        result.name for result in report.validation.blocking
    }
    assert (tmp_path / FEATURES_FILENAME).is_file()


def test_an_empty_table_produces_an_empty_build(tmp_path: Path) -> None:
    report = run_features(canonical_frame([]), tmp_path)
    assert report.matches == 0
    assert report.coverage == {"team_history": 0.0, "head_to_head": 0.0}
    assert report.verified_on is None


# ---- the shared report ------------------------------------------------------


def test_a_report_with_no_probes_is_vacuously_causal() -> None:
    assert DerivedReport().causal


def test_the_sample_is_the_competition_closest_to_the_target() -> None:
    small = league_frame(seasons=["2020-21"], teams=6, team_prefix="Small")
    big = league_frame(
        competition_id="ESP_1",
        country="Spain",
        name="La Liga",
        seasons=["2018-19", "2019-20", "2020-21"],
        teams=12,
        team_prefix="Big",
    )
    together = pd.concat([small, big], ignore_index=True).sort_values("date", kind="stable")
    assert choose_verification_sample(together, target=30) == "ENG_1"
    assert choose_verification_sample(together, target=10_000) == "ESP_1"


def test_no_sample_from_an_empty_table() -> None:
    assert choose_verification_sample(canonical_frame([])) is None
