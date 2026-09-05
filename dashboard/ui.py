"""The components every page is built from.

A card, a probability bar, a crest, a form string, a section header. Written as
functions returning HTML rather than as Streamlit calls, for one reason worth
stating: a match card is a *link*, and a grid of forty of them has to be forty
anchors the browser can follow — not forty `st.button` widgets, each of which
is a round trip to the server and a rerun of the whole script.

**Every string that came from data is escaped.** Club and competition names
come out of a provider's CSV, and this module is the only place in the project
that puts them inside markup. `html.escape` on all of them, once, here, rather
than a rule the next component has to remember.

**Nothing here computes.** A card renders the probabilities it is handed; it
does not ask the service for them and it does not decide which fixtures to
show. That belongs to the views, which is what makes every component testable
by handing it a fixture and reading the string back.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Final

import streamlit as st

from dashboard import theme
from dashboard.domain.match import Fixture, MatchStatus

MATCH_PAGE: Final = "match"
"""The url path of the prediction detail page. Every card links to it."""

UNKNOWN_KICKOFF: Final = frozenset({"", "00:00"})
"""Kick-off values that mean "the provider did not record one".

Midnight is in the table about one time in ninety and is the provider's own
placeholder rather than a fixture played at midnight. Showing it would be a
kick-off time that is wrong for every one of those matches.
"""

CREST_COLOURS: Final[tuple[str, ...]] = (
    "#f87171",
    "#fb923c",
    "#fbbf24",
    "#a3e635",
    "#34d399",
    "#22d3ee",
    "#60a5fa",
    "#a78bfa",
    "#f472b6",
    "#e2e8f0",
)
"""The palette a generated crest picks from.

There are no club badges in this repository and there is no licence to ship
any, so a crest is the club's initials on a colour derived from its name. The
colour is stable — the same club is the same colour on every page and between
sessions — because a reader scanning a column of forty cards navigates by
colour before they read a word, and a palette that reshuffled per render would
be worse than no colour at all.
"""


# ---- small pieces ------------------------------------------------------------


def crest_html(team: str, url: str | None = None) -> str:
    """A club's badge: the provider's image where there is one, initials where not.

    ``url`` is filled by no source in this repository. It is the field a badge
    provider starts writing, and this is the one function that has to change
    when it does.
    """
    if url:
        return f'<img class="mop-crest" src="{html.escape(url, quote=True)}" alt="">'
    colour = CREST_COLOURS[_stable_index(team, len(CREST_COLOURS))]
    return (
        f'<span class="mop-crest" style="background:{colour}" aria-hidden="true">'
        f"{html.escape(initials(team))}</span>"
    )


def initials(team: str) -> str:
    """Up to three letters that identify a club at 26 pixels.

    The first letter of each word, which is what a reader recognises — "MU" for
    Man United, "PSG" for Paris Saint-Germain. A one-word club keeps its first
    three letters rather than a lone initial, because a column of cards where
    six clubs are all "B" identifies nothing.
    """
    words = [word for word in team.split() if word[:1].isalnum()]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:3].upper()
    return "".join(word[0] for word in words[:3]).upper()


def _stable_index(text: str, modulus: int) -> int:
    """A deterministic bucket for a string.

    Summed code points rather than :func:`hash`, which is salted per process:
    the same club would be a different colour on every restart, and "stable"
    is the entire property this is for.
    """
    return sum(ord(character) for character in text) % modulus


def pill(text: str, *, live: bool = False) -> str:
    """A small label — a competition, a date, a status."""
    classes = "mop-pill live" if live else "mop-pill"
    return f'<span class="{classes}">{html.escape(text)}</span>'


def probability_bar(probabilities: Mapping[str, float], *, legend: bool = True) -> str:
    """Home, draw and away as one bar, in the order the model returns them.

    A bar rather than three numbers because the question a reader asks of a
    forecast first is "how one-sided", and that is a length. The numbers are
    under it, because the question they ask second is "how one-sided exactly".
    """
    widths = _percentages(probabilities)
    segments = "".join(
        f'<span style="width:{width:.4f}%;background:{theme.OUTCOME_COLOURS[key]}"'
        f' title="{key} {width:.1f}%"></span>'
        for key, width in widths.items()
    )
    bar = f'<div class="mop-bar">{segments}</div>'
    if not legend:
        return bar
    labels = "".join(
        f"<span>{key.capitalize()} <b>{width:.0f}%</b></span>" for key, width in widths.items()
    )
    return f'{bar}<div class="mop-legend">{labels}</div>'


def _percentages(probabilities: Mapping[str, float]) -> dict[str, float]:
    """The three probabilities as percentages that sum to a hundred.

    Renormalised rather than trusted. The service returns three floats that sum
    to one, but a bar built from percentages that sum to 99.7 has a visible
    gap at its right edge, and a caller may legitimately hand this a filtered
    or rounded distribution.
    """
    values = {key: max(float(probabilities.get(key, 0.0)), 0.0) for key in theme.OUTCOME_COLOURS}
    total = sum(values.values())
    if total <= 0:
        return dict.fromkeys(values, 100.0 / len(values))
    return {key: value * 100.0 / total for key, value in values.items()}


def confidence(probabilities: Mapping[str, float]) -> tuple[str, str, float]:
    """How firmly the model committed: the outcome, a band, and the probability.

    **This is not a calibrated confidence and must not be read as one.** It is
    the largest of three probabilities, banded for legibility. What that
    probability is *worth* — how often forecasts stated near it actually happen
    — is a measurement, it is on the detail page, and it comes from the
    reliability tables the model card is generated from.

    The bands start at 0.45 and 0.60 because the floor here is a third, not
    zero: a three-way forecast whose largest leg is 0.34 has said almost
    nothing, and calling that "low" rather than "34%" is the honest label.
    """
    outcome = max(theme.OUTCOME_COLOURS, key=lambda key: float(probabilities.get(key, 0.0)))
    value = float(probabilities.get(outcome, 0.0))
    if value >= 0.60:
        band = "high"
    elif value >= 0.45:
        band = "moderate"
    else:
        band = "low"
    return outcome, band, value


def confidence_pill(probabilities: Mapping[str, float]) -> str:
    """The confidence band as a coloured pill."""
    outcome, band, value = confidence(probabilities)
    colour = theme.CONFIDENCE_COLOURS[band]
    text = html.escape(f"{outcome.capitalize()} {value:.0%} · {band}")
    return f'<span class="mop-pill" style="color:{colour};border-color:{colour}33">{text}</span>'


def form_string(results: Sequence[str]) -> str:
    """A run of W/D/L as coloured squares, oldest first."""
    if not results:
        return '<span class="mop-legend">no earlier matches</span>'
    squares = "".join(f'<i class="{letter}">{letter}</i>' for letter in results if letter in "WDL")
    return f'<span class="mop-form">{squares}</span>'


def link(label: str, href: str) -> str:
    """An in-app link.

    An anchor rather than :func:`streamlit.page_link`, which takes a page and
    cannot carry a query parameter — and a query parameter is how every deep
    link in this application names what it is a link *to*. ``target="_self"``
    keeps it inside the tab; without it Streamlit's markdown renderer opens a
    new one.
    """
    return (
        f'<a class="mop-pill" href="{html.escape(href, quote=True)}" target="_self">'
        f"{html.escape(label)} →</a>"
    )


def section(title: str, note: str = "") -> None:
    """A section header, with the sentence that says what is in it."""
    suffix = f"<span>{html.escape(note)}</span>" if note else ""
    st.markdown(
        f'<div class="mop-section"><h3>{html.escape(title)}</h3>{suffix}</div>',
        unsafe_allow_html=True,
    )


def placeholder(title: str, body: str) -> None:
    """What a section that has no provider yet says.

    A rendered, deliberate sentence rather than an empty column. Milestone 12
    ships the architecture for live fixtures and not the fixtures, and a screen
    that was simply blank there would read as a bug rather than as a boundary.
    """
    st.markdown(
        f'<div class="mop-placeholder"><b>{html.escape(title)}</b><br>{body}</div>',
        unsafe_allow_html=True,
    )


# ---- the card ----------------------------------------------------------------


def match_card(
    fixture: Fixture,
    *,
    probabilities: Mapping[str, float] | None = None,
    competition_label: str | None = None,
) -> str:
    """One match, as the link a reader clicks to open its prediction page.

    ``probabilities`` is optional and is what separates a result card from a
    forecast card. A page listing two hundred finished matches does not ask the
    service to price all of them — that would be two hundred HTTP calls behind
    one scroll — so it renders the same card without a bar.
    """
    top = _card_top(fixture, competition_label)
    body = _card_sides(fixture)
    bar = probability_bar(probabilities) if probabilities is not None else ""
    href = f"{MATCH_PAGE}?match={html.escape(fixture.match_id, quote=True)}"
    return f'<a class="mop-card" href="{href}" target="_self">' f"{top}{body}{bar}</a>"


def _card_top(fixture: Fixture, competition_label: str | None) -> str:
    """The strip above the teams: competition, then when."""
    label = competition_label or fixture.competition or fixture.competition_id
    parts = [pill(label)]
    if fixture.status is MatchStatus.LIVE:
        minute = f"{fixture.minute}'" if fixture.minute is not None else "live"
        parts.append(pill(minute, live=True))
    parts.append('<span class="spacer"></span>')
    parts.append(f"<span>{html.escape(_when(fixture))}</span>")
    return f'<div class="mop-card-top">{"".join(parts)}</div>'


def _when(fixture: Fixture) -> str:
    """The date, with the kick-off time when the provider recorded one."""
    day = fixture.date.strftime("%a %d %b %Y")
    kickoff = (fixture.kickoff or "").strip()
    return f"{day} · {kickoff}" if kickoff not in UNKNOWN_KICKOFF else day


def _card_sides(fixture: Fixture) -> str:
    """The two clubs and, where there is one, the score.

    The losing side is dimmed rather than the winner emboldened: on a card that
    already carries a coloured bar, a second weight of type is one signal too
    many, and dimming reads correctly for a draw with no special case.
    """
    home_beaten = fixture.has_score and _int(fixture.home_goals) < _int(fixture.away_goals)
    away_beaten = fixture.has_score and _int(fixture.away_goals) < _int(fixture.home_goals)
    return _side(
        fixture.home_team, fixture.home_goals, fixture.home_crest_url, dim=home_beaten
    ) + _side(fixture.away_team, fixture.away_goals, fixture.away_crest_url, dim=away_beaten)


def _side(team: str, goals: int | None, crest: str | None, *, dim: bool) -> str:
    score = f'<span class="mop-score">{goals}</span>' if goals is not None else ""
    classes = "mop-row dim" if dim else "mop-row"
    return (
        f'<div class="{classes}"><span class="mop-side">{crest_html(team, crest)}'
        f'<span class="mop-team">{html.escape(team)}</span></span>{score}</div>'
    )


def _int(value: int | None) -> int:
    return 0 if value is None else value


def card_grid(
    cards: Sequence[str], *, columns: int = 3, empty: str = "Nothing to show here."
) -> None:
    """Lay cards out in columns, down each column in turn.

    Streamlit's columns are independent vertical stacks, so a grid is built by
    dealing cards into them. Round-robin rather than in blocks, because cards
    have different heights and blocks put every tall one in the same column.
    """
    if not cards:
        st.caption(empty)
        return
    lanes = st.columns(min(columns, len(cards)))
    for position, card in enumerate(cards):
        with lanes[position % len(lanes)]:
            st.markdown(card, unsafe_allow_html=True)
