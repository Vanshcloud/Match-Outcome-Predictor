"""Team identity is exact, not fuzzy, and that was a measurement.

Across the 34 Premier League seasons the provider publishes — 12,724 matches —
it uses only 51 distinct team strings. The test that matters most here is the
one showing why a similarity threshold would be actively harmful on exactly
that vocabulary.
"""

from __future__ import annotations

import difflib

import pytest

from src.ingestion.teams import country_code, find_single_season_teams, slugify, team_id

# ---- slugs ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Man United", "man-united"),
        ("Aston Villa", "aston-villa"),
        ("Nott'm Forest", "nott-m-forest"),
        ("1. FC Koln", "1-fc-koln"),
        ("  Arsenal  ", "arsenal"),
        ("Sheffield   Weds", "sheffield-weds"),
    ],
)
def test_slugify(name: str, expected: str) -> None:
    assert slugify(name) == expected


def test_accents_fold_to_their_base_letters() -> None:
    """The Turkish, Greek and Portuguese divisions are not consistently
    transliterated by the provider, so both spellings must land on one id."""
    assert slugify("Beşiktaş") == slugify("Besiktas")
    assert slugify("Málaga") == slugify("Malaga")


def test_a_name_that_reduces_to_nothing_raises() -> None:
    """An empty id would join to every other broken row, which is worse than
    a loud failure at the point the parsing actually went wrong."""
    with pytest.raises(ValueError, match="empty slug"):
        slugify("---")


# ---- identity ---------------------------------------------------------------


def test_a_club_keeps_one_id_across_divisions() -> None:
    """Identity is scoped to the country, not the competition. Promotion and
    relegation must not restart a club's history — which is exactly the
    continuity every rolling form feature depends on."""
    assert team_id("ENG_1", "Luton") == team_id("ENG_2", "Luton")
    assert team_id("ENG_1", "Luton") == team_id("ENG_5", "Luton")


def test_same_name_in_different_countries_is_a_different_club() -> None:
    """Arsenal of England and Arsenal de Sarandí of Argentina are not the
    same team, and a global name index would merge them."""
    assert team_id("ENG_1", "Arsenal") != team_id("ARG_1", "Arsenal")


def test_ids_are_readable() -> None:
    """These appear in logs, error messages and dashboard URLs. An opaque
    digest turns every debugging session into a lookup."""
    assert team_id("ENG_1", "Man United") == "eng:man-united"


def test_aliases_map_a_second_provider_onto_the_canonical_spelling() -> None:
    aliases = {("ENG", "Manchester United"): "Man United"}
    assert team_id("ENG_1", "Manchester United", aliases) == team_id("ENG_1", "Man United")


def test_alias_lookup_tolerates_surrounding_whitespace() -> None:
    aliases = {("ENG", "Manchester United"): "Man United"}
    assert team_id("ENG_1", "  Manchester United  ", aliases) == "eng:man-united"


@pytest.mark.parametrize("bad", ["england1", "E_1", "eng_1", "ENGLAND_1", "ENG"])
def test_a_malformed_competition_id_raises(bad: str) -> None:
    with pytest.raises(ValueError, match="ENG_1"):
        country_code(bad)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # The two most similar distinct pairs in the real Premier League
        # vocabulary. Both score exactly 0.800.
        ("Sheffield United", "Sheffield Weds"),
        ("Barnsley", "Burnley"),
    ],
)
def test_a_similarity_threshold_would_merge_real_clubs(first: str, second: str) -> None:
    """The measured justification for exact matching.

    These are different clubs — Sheffield United and Sheffield Wednesday share
    a city and have played each other — and they sit at exactly the similarity
    anyone would pick as a threshold. Fuzzy matching this vocabulary has
    nothing to find and a club merger to cause.
    """
    ratio = difflib.SequenceMatcher(None, first.lower(), second.lower()).ratio()
    assert ratio >= 0.80, "these pairs are the reason the threshold is unusable"
    assert team_id("ENG_1", first) != team_id("ENG_1", second)


# ---- rename detection -------------------------------------------------------


def test_single_season_teams_are_reported() -> None:
    """A rename presents as one club stopping and another starting. This cannot
    tell that apart from a genuine one-season promotion — nothing can, without
    a human — so it reports both rather than guessing."""
    rows = [
        ("eng:arsenal", "2023-24"),
        ("eng:arsenal", "2024-25"),
        ("eng:luton", "2023-24"),
    ]
    assert find_single_season_teams(rows) == {"eng:luton": "2023-24"}


def test_nothing_is_reported_when_every_team_persists() -> None:
    rows = [("a", "1"), ("a", "2"), ("b", "1"), ("b", "2")]
    assert find_single_season_teams(rows) == {}
