"""The feature registry.

Its job is to make the schema's leakage classification enforceable rather
than documented: each feature declares what it reads, and whether it *can* leak
follows from that rather than from a second field somebody might set wrongly.
"""

from __future__ import annotations

import pytest

from src.feature_engineering.registry import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    FEATURES,
    KEY_COLUMN,
    Feature,
    RegistryError,
    by_name,
    groups,
    validate,
)
from src.ingestion.base import CANONICAL_COLUMNS, POST_MATCH_COLUMNS, PRE_MATCH_COLUMNS


def test_every_feature_reads_canonical_columns_only() -> None:
    """`reads` is hand-written, and a typo in it would silently reclassify a
    leaking feature as safe."""
    for feature in FEATURES:
        assert feature.reads <= set(CANONICAL_COLUMNS), feature.name


def test_feature_names_are_unique() -> None:
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_the_schema_is_derived_from_the_registry() -> None:
    """One source of truth, or the table's shape ends up defined differently in
    two places."""
    assert list(FEATURE_SCHEMA) == [KEY_COLUMN, *FEATURE_COLUMNS]


def test_a_feature_that_reads_a_result_can_leak() -> None:
    assert by_name("home_form_points_5").can_leak
    assert by_name("h2h_home_points").can_leak


def test_a_feature_that_reads_only_the_fixture_list_cannot() -> None:
    """Rest days and congestion are facts about the calendar. They are computed
    from past matches, but nothing they read is a result."""
    for name in ("home_rest_days", "away_matches_14d", "home_matches_played", "h2h_matches"):
        assert not by_name(name).can_leak, name
        assert by_name(name).reads <= PRE_MATCH_COLUMNS


def test_can_leak_is_exactly_reading_a_post_match_column() -> None:
    """Derived rather than declared: there is no field to set wrongly."""
    for feature in FEATURES:
        assert feature.can_leak == bool(feature.reads & POST_MATCH_COLUMNS)


def test_both_kinds_of_feature_exist() -> None:
    """A registry where everything can leak, or nothing can, would mean the
    classification had stopped distinguishing anything."""
    assert any(feature.can_leak for feature in FEATURES)
    assert any(not feature.can_leak for feature in FEATURES)


def test_every_feature_is_grouped_and_described() -> None:
    for feature in FEATURES:
        assert feature.group
        assert len(feature.description) > 20, feature.name


def test_groups_partition_the_registry() -> None:
    assert sum(len(members) for members in groups().values()) == len(FEATURES)


def test_home_and_away_features_come_in_pairs() -> None:
    """They are the same computation seen from two ends, and writing them out
    twice is how the two drift apart."""
    sided = [name for name in FEATURE_COLUMNS if name.startswith(("home_", "away_"))]
    suffixes = {name.split("_", 1)[1] for name in sided}
    for suffix in suffixes:
        assert f"home_{suffix}" in FEATURE_COLUMNS
        assert f"away_{suffix}" in FEATURE_COLUMNS


def test_an_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        by_name("no_such_feature")


# ---- the validator ----------------------------------------------------------


def _feature(name: str, reads: frozenset[str]) -> Feature:
    return Feature(name=name, dtype="Float64", group="test", reads=reads, description="x" * 30)


def test_duplicate_names_are_refused() -> None:
    twice = _feature("same", frozenset({"date"}))
    with pytest.raises(RegistryError, match="duplicate feature names"):
        validate([twice, twice])


def test_a_non_canonical_column_is_refused() -> None:
    with pytest.raises(RegistryError, match="non-canonical"):
        validate([_feature("typo", frozenset({"home_gaols"}))])


def test_a_feature_with_no_inputs_is_refused() -> None:
    """A feature that reads nothing is either a constant or a lie about what it
    reads, and the second is the dangerous one."""
    with pytest.raises(RegistryError, match="declares no inputs"):
        validate([_feature("empty", frozenset())])


def test_the_shipped_registry_validates() -> None:
    validate(FEATURES)
