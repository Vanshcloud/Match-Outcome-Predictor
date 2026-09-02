"""The registry's promise is that adding a league is a config entry, not code.

These tests load the file this project actually ships, because a registry that
validates in the abstract and is malformed on disk protects nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.ingestion.base import Capability
from src.ingestion.registry import (
    Competition,
    Feed,
    Registry,
    load_registry,
    normalise_season,
    season_code_to_label,
    season_label_to_code,
    season_start_year,
)

REGISTRY = load_registry()


# ---- the shipped file -------------------------------------------------------


def test_the_shipped_registry_loads() -> None:
    assert len(REGISTRY.competitions) >= 38


def test_every_competition_id_is_unique() -> None:
    ids = [competition.id for competition in REGISTRY.competitions]
    assert len(ids) == len(set(ids))


def test_a_duplicate_id_is_refused() -> None:
    """Two leagues sharing a key would merge silently at every join."""
    entry = {"id": "ENG_1", "country": "England", "name": "A", "feed": "main", "code": "E0"}
    with pytest.raises(ValidationError, match="duplicate competition id"):
        Registry.model_validate({"competitions": [entry, {**entry, "name": "B"}]})


def test_multi_competition_country_files_are_filtered() -> None:
    """Argentina's file mixes a league with a cup, so both entries must name
    which rows they want. Unfiltered they merge into one impossible
    competition."""
    for competition_id in ("ARG_1", "ARG_CUP", "SUI_1"):
        assert REGISTRY.by_id(competition_id).league_filter is not None


def test_the_swiss_playoff_label_is_not_registered_as_a_league() -> None:
    """Switzerland's file carries a 'Challenge League' label that is not the
    second tier: two matches, Thun v Sion home and away in May 2021, the
    promotion/relegation playoff. Registered it would be a league with one
    season and two fixtures."""
    ids = {competition.id for competition in REGISTRY.competitions}
    assert "SUI_2" not in ids


def test_competitions_sharing_a_country_file_have_distinct_filters() -> None:
    by_code: dict[str, set[str | None]] = {}
    for competition in REGISTRY.for_feed(Feed.EXTRA):
        by_code.setdefault(competition.code, set()).add(competition.league_filter)
    for code, filters in by_code.items():
        if len(filters) > 1:
            assert None not in filters, f"{code}: an unfiltered entry would swallow the others"


def test_unknown_id_lookup_lists_what_is_available() -> None:
    with pytest.raises(KeyError, match="ENG_1"):
        REGISTRY.by_id("NOPE")


# ---- validation -------------------------------------------------------------


def test_id_format_is_enforced() -> None:
    """The country prefix is what scopes team identity, so a malformed id would
    silently produce teams that never join."""
    with pytest.raises(ValidationError):
        Competition(id="england1", country="England", name="PL", feed=Feed.MAIN, code="E0")


def test_whitespace_in_names_is_stripped() -> None:
    """The provider's own data holds 'Ireland ' alongside 'Ireland' and
    ' J1 League' alongside 'J1 League'."""
    competition = Competition(
        id="IRL_1",
        country=" Ireland ",
        name="Premier Division",
        feed=Feed.EXTRA,
        code="IRL",
        league_filter=" Premier Division ",
    )
    assert competition.country == "Ireland"
    assert competition.league_filter == "Premier Division"


def test_capabilities_follow_the_feed() -> None:
    """The primary feed carries shots; the secondary one never does."""
    assert REGISTRY.by_id("ENG_1").supports(Capability.MATCH_STATS)
    assert not REGISTRY.by_id("BRA_1").supports(Capability.MATCH_STATS)
    assert REGISTRY.by_id("BRA_1").supports(Capability.RESULTS)


def test_an_unknown_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "leagues.yaml"
    path.write_text(
        "competitions:\n  - {id: ENG_1, country: England, name: PL, feed: main, "
        "code: E0, tyre: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="tyre"):
        load_registry(path)


# ---- season labels ----------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "label"),
    [("9394", "1993-94"), ("9900", "1999-00"), ("0001", "2000-01"), ("2425", "2024-25")],
)
def test_season_codes_convert_both_ways(code: str, label: str) -> None:
    """1999-00 is the case a naive implementation gets wrong: the two-digit
    years wrap and the century pivot has to hold."""
    assert season_code_to_label(code) == label
    assert season_label_to_code(label) == code


@pytest.mark.parametrize("code", ["24", "2426", "abcd", "20245"])
def test_malformed_season_codes_are_refused(code: str) -> None:
    with pytest.raises(ValueError, match="season code"):
        season_code_to_label(code)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024", "2024"),  # calendar: Brazil, USA, Japan through 2025
        ("2012/2013", "2012-13"),  # split: Austria, Poland, Romania
        ("2024/25", "2024-25"),
        ("2026/2027", "2026-27"),  # Japan's switch away from calendar years
        (" 2024 ", "2024"),
    ],
)
def test_secondary_feed_seasons_normalise(raw: str, expected: str) -> None:
    assert normalise_season(raw) == expected


@pytest.mark.parametrize("raw", ["24", "2024-2025", "2024/2026", "spring"])
def test_unrecognised_seasons_raise_rather_than_guess(raw: str) -> None:
    with pytest.raises(ValueError):
        normalise_season(raw)


def test_season_labels_sort_chronologically_as_text() -> None:
    """This is why labels are strings shaped this way: a temporal split becomes
    a comparison, with no parsing at the point of use."""
    labels = ["2024-25", "1993-94", "2000-01", "1999-00"]
    assert sorted(labels) == ["1993-94", "1999-00", "2000-01", "2024-25"]


def test_start_year_works_for_both_label_forms() -> None:
    assert season_start_year("2024-25") == 2024
    assert season_start_year("2024") == 2024


@pytest.mark.parametrize("label", ["2024", "24-25", "not-a-season"])
def test_only_split_labels_convert_back_to_a_code(label: str) -> None:
    """Calendar seasons have no URL code — they live in the secondary feed,
    which is addressed by country rather than by season."""
    with pytest.raises(ValueError, match="not a split-season label"):
        season_label_to_code(label)


@pytest.mark.parametrize("label", ["24", "2024/25", "next year"])
def test_start_year_refuses_an_unrecognised_label(label: str) -> None:
    with pytest.raises(ValueError, match="unrecognised season label"):
        season_start_year(label)
