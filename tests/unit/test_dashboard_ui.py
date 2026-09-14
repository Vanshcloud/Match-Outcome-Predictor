"""The components, as the strings they produce.

Every one is a pure function from a fixture to HTML, which is what makes them
testable without a browser and what makes these tests worth having: the three
things that would be invisible in a screenshot are that names are escaped, that
the probability segments sum to the width of the bar, and that the crest colour
for a club does not change between runs.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from dashboard import theme, ui
from dashboard.domain import competition as catalogue
from dashboard.domain.match import Fixture, MatchStatus

WIDTH = re.compile(r"width:([0-9.]+)%")


def fixture(**overrides: object) -> Fixture:
    fields: dict[str, object] = {
        "match_id": "abc",
        "competition_id": "ENG_1",
        "date": dt.date(2026, 8, 31),
        "home_team": "Aston Villa",
        "away_team": "Arsenal",
    }
    fields.update(overrides)
    return Fixture(**fields)  # type: ignore[arg-type]


# ---- escaping ----------------------------------------------------------------


def test_a_club_name_is_escaped_before_it_reaches_the_markup() -> None:
    """Club names come out of a provider's CSV and this module is the only
    place in the project that puts them inside markup."""
    card = ui.match_card(fixture(home_team="<script>alert(1)</script>"))
    assert "<script>" not in card
    assert "&lt;script&gt;" in card


def test_a_match_id_is_escaped_inside_the_link() -> None:
    card = ui.match_card(fixture(match_id='a" onclick="x'))
    assert 'onclick="x"' not in card
    assert "&quot;" in card


def test_a_crest_url_is_escaped_too() -> None:
    """No source fills this field today. It is the one a badge provider starts
    writing, and it will be a URL somebody else chose."""
    assert "&quot;" in ui.crest_html("Arsenal", '"><img onerror=x')


# ---- the probability bar -----------------------------------------------------


def test_the_three_segments_fill_the_bar_exactly() -> None:
    bar = ui.probability_bar({"home": 0.178, "draw": 0.241, "away": 0.581})
    assert sum(float(one) for one in WIDTH.findall(bar)) == pytest.approx(100.0)


def test_a_distribution_that_does_not_sum_to_one_is_renormalised() -> None:
    """A bar built from percentages summing to 99.7 has a visible gap at its
    right edge, and a caller may hand this a rounded distribution."""
    bar = ui.probability_bar({"home": 0.3, "draw": 0.3, "away": 0.3})
    # abs rather than relative: the widths are written to four decimal places,
    # so three thirds sum to 99.9999 and that is a rounding decision, not a gap.
    assert sum(float(one) for one in WIDTH.findall(bar)) == pytest.approx(100.0, abs=1e-3)


def test_an_empty_distribution_is_three_equal_segments_rather_than_a_crash() -> None:
    bar = ui.probability_bar({"home": 0.0, "draw": 0.0, "away": 0.0})
    assert [float(one) for one in WIDTH.findall(bar)] == pytest.approx([100 / 3] * 3, abs=1e-3)


def test_a_negative_probability_cannot_produce_a_negative_width() -> None:
    bar = ui.probability_bar({"home": -0.2, "draw": 0.6, "away": 0.6})
    assert all(float(one) >= 0 for one in WIDTH.findall(bar))


def test_the_legend_can_be_left_off_for_a_card() -> None:
    assert "mop-legend" not in ui.probability_bar(
        {"home": 0.5, "draw": 0.3, "away": 0.2}, legend=False
    )


def test_the_segments_are_in_the_order_the_model_returns_them() -> None:
    """The same order as `CLASSES`. A page that showed them differently is a
    page where a reader checking one against the other reads the wrong leg."""
    bar = ui.probability_bar({"home": 0.5, "draw": 0.3, "away": 0.2})
    assert bar.index(theme.HOME) < bar.index(theme.DRAW) < bar.index(theme.AWAY)


# ---- confidence --------------------------------------------------------------


@pytest.mark.parametrize(
    ("largest", "band"),
    [(0.72, "high"), (0.60, "high"), (0.52, "moderate"), (0.45, "moderate"), (0.38, "low")],
)
def test_the_confidence_band_starts_where_a_three_way_forecast_starts(
    largest: float, band: str
) -> None:
    """The floor is a third, not zero. A forecast whose largest leg is 0.34 has
    said almost nothing, and "low" is the honest label for it."""
    rest = (1 - largest) / 2
    outcome, found, value = ui.confidence({"home": largest, "draw": rest, "away": rest})
    assert (outcome, found) == ("home", band)
    assert value == pytest.approx(largest)


def test_the_confidence_pill_names_the_outcome_and_the_band() -> None:
    pill = ui.confidence_pill({"home": 0.178, "draw": 0.241, "away": 0.581})
    assert "Away 58%" in pill
    assert "moderate" in pill


# ---- crests ------------------------------------------------------------------


def test_a_clubs_colour_is_the_same_on_every_page_and_between_runs() -> None:
    """A reader scanning forty cards navigates by colour before they read a
    word, and `hash` is salted per process."""
    assert ui.crest_html("Arsenal") == ui.crest_html("Arsenal")
    assert ui.crest_html("Arsenal") != ui.crest_html("Chelsea")


@pytest.mark.parametrize(
    ("club", "expected"),
    [
        ("Manchester United", "MU"),
        ("Paris Saint-Germain", "PS"),
        ("Arsenal", "ARS"),
        ("Bayern Munich", "BM"),
        ("", "?"),
        ("!!!", "?"),
    ],
)
def test_initials_identify_a_club_at_twenty_six_pixels(club: str, expected: str) -> None:
    assert ui.initials(club) == expected


def test_a_supplied_badge_replaces_the_generated_one() -> None:
    assert "<img" in ui.crest_html("Arsenal", "https://example.test/badge.png")


# ---- the card ----------------------------------------------------------------


def test_a_finished_card_shows_the_score_and_dims_the_beaten_side() -> None:
    card = ui.match_card(fixture(status=MatchStatus.FINISHED, home_goals=0, away_goals=1))
    assert card.count("mop-row dim") == 1
    assert '<span class="mop-score">1</span>' in card


def test_a_draw_dims_neither_side() -> None:
    card = ui.match_card(fixture(status=MatchStatus.FINISHED, home_goals=1, away_goals=1))
    assert "mop-row dim" not in card


def test_a_card_with_no_score_shows_none() -> None:
    assert "mop-score" not in ui.match_card(fixture())


def test_a_live_card_carries_the_minute() -> None:
    card = ui.match_card(fixture(status=MatchStatus.LIVE, minute=63))
    assert "mop-pill live" in card
    assert "63&#x27;" in card or "63'" in card


def test_a_live_card_with_no_minute_still_says_live() -> None:
    card = ui.match_card(fixture(status=MatchStatus.LIVE))
    assert "mop-pill live" in card


def test_the_card_links_to_the_match_page_by_id() -> None:
    assert 'href="match?match=abc"' in ui.match_card(fixture())


def test_the_kickoff_is_shown_when_the_provider_recorded_one() -> None:
    assert "15:00" in ui.match_card(fixture(kickoff="15:00"))


@pytest.mark.parametrize("recorded", ["00:00", "", None])
def test_the_providers_own_placeholder_kickoff_is_not_shown(recorded: str | None) -> None:
    """Midnight is in the table about one time in ninety and is a placeholder.
    Showing it would be a kick-off time that is wrong for all of them."""
    card = ui.match_card(fixture(kickoff=recorded))
    assert "00:00" not in card
    assert "Mon 31 Aug 2026" in card


def test_a_card_prefers_the_label_it_is_handed_then_the_name_then_the_id() -> None:
    assert "Premier League" in ui.match_card(fixture(), competition_label="Premier League")
    assert "La Liga" in ui.match_card(fixture(competition="La Liga"))
    assert "ENG_1" in ui.match_card(fixture())


def test_a_card_holds_only_inline_elements_so_the_browser_keeps_it_one_link() -> None:
    """Streamlit's markdown wraps a card in ``<p>``, and a ``<p>`` cannot hold a
    block element: a ``<div>`` inside it closes the anchor, and the browser
    re-opens one around every block, so the card renders as separate fragments.
    Every test that read the card's text passed while every browser showed it
    broken."""
    card = ui.match_card(
        fixture(status=MatchStatus.LIVE, minute=10, home_goals=1, away_goals=0),
        probabilities={"home": 0.5, "draw": 0.3, "away": 0.2},
    )
    assert not re.search(r"<(div|p|h[1-6]|ul|ol|li|table|section)\b", card)
    assert card.count("<a ") == 1


def test_probabilities_turn_a_result_card_into_a_forecast_card() -> None:
    card = ui.match_card(fixture(), probabilities={"home": 0.5, "draw": 0.3, "away": 0.2})
    assert "mop-bar" in card
    assert "mop-bar" not in ui.match_card(fixture())


# ---- small pieces ------------------------------------------------------------


def test_a_form_string_is_one_square_per_result() -> None:
    assert ui.form_string(["W", "D", "L"]).count("<i") == 3


def test_an_empty_form_says_so_rather_than_rendering_nothing() -> None:
    assert "no earlier" in ui.form_string([])


def test_a_form_string_ignores_a_letter_that_is_not_a_result() -> None:
    assert ui.form_string(["W", "X"]).count("<i") == 1


def test_a_link_stays_inside_the_tab() -> None:
    assert 'target="_self"' in ui.link("Open", "competitions?competition=ENG_1")


def test_the_kick_off_time_moves_to_its_own_line_whole_rather_than_breaking() -> None:
    """A narrow card used to break "Sun 13 Sep 2026 · 00:30 IST" mid-time. The
    header may wrap, but only between its pieces."""
    assert '<span class="mop-when">' in ui.match_card(fixture())
    assert ".mop-card-top .mop-when { white-space: nowrap; }" in theme.STYLESHEET
    assert "flex-wrap: wrap;" in theme.STYLESHEET


def test_the_card_grid_sizes_its_columns_to_the_width_rather_than_a_fixed_count() -> None:
    """Three fixed Streamlit columns at 1024px were 143px cards. The grid is
    one element whose column count follows the width."""
    assert "repeat(\n    auto-fill,\n    minmax(max(var(--mop-min)," in theme.STYLESHEET


def test_the_card_grid_carries_its_column_limit_and_minimum_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written: list[str] = []
    monkeypatch.setattr(ui.st, "markdown", lambda body, **_: written.append(body))
    ui.card_grid(['<a class="mop-card"></a>'], columns=4, min_rem=11)
    assert written == [
        '<span class="mop-grid" style="--mop-cols:4;--mop-min:11.0rem"><a class="mop-card"></a></span>'
    ]


def test_a_competition_name_wraps_rather_than_being_cut_off() -> None:
    assert (
        "text-overflow: ellipsis"
        not in theme.STYLESHEET.split("a.mop-league {", 1)[1].split("}", 1)[0]
    )


def test_an_anchor_pill_is_styled_or_the_browser_draws_it_blue_and_underlined() -> None:
    """`ui.link` wears the pill class, and Streamlit styles every anchor inside
    markdown as a document link. Without an `a.mop-pill` rule the "Open" on
    Competitions and the shortcuts on Search render underlined and blue inside
    a pill border — which no test that read their text could see."""
    assert "mop-pill" in ui.link("Open", "competitions?competition=ENG_1")
    assert "a.mop-pill {" in theme.STYLESHEET


# ---- the competition row -----------------------------------------------------


def test_a_competition_row_is_one_link_with_its_countrys_flag() -> None:
    row = ui.competition_row("Serie A", "competitions?competition=ITA_1", country="Italy")
    assert row.count("<a ") == 1
    assert 'href="competitions?competition=ITA_1"' in row
    assert 'target="_self"' in row
    assert "/flags/4x3/it.svg" in row and 'alt="Italy"' in row
    assert not re.search(r"<(div|p|h[1-6]|ul|ol|li|table|section)\b", row)


def test_a_country_with_no_flag_keeps_the_frame_so_names_stay_aligned() -> None:
    row = ui.competition_row("League", "#", country="Atlantis")
    assert "<img" not in row and 'class="mop-flag"' in row


def test_every_registry_country_has_a_flag() -> None:
    assert {one.country for one in catalogue.competitions()} <= set(ui.FLAG_CODES)


def test_a_competition_name_is_escaped_before_it_reaches_the_markup() -> None:
    row = ui.competition_row("<script>x</script>", "#", country="Italy")
    assert "<script>" not in row
