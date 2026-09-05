"""The shell: the theme, the sidebar, and the six pages the reader moves between.

``streamlit run dashboard/app.py``

This file holds no content. It sets the page up, renders the one thing every
page shares — what the reader follows, and whether the service and the tables
are there — and hands over to :mod:`dashboard.views`.

**Pages are declared, not discovered.** ``st.navigation`` over explicit
:class:`streamlit.Page` objects rather than Streamlit's ``pages/`` directory
convention, for two reasons that both matter later: a page gets a stable
``url_path`` that a card can deep-link to with a query parameter, and every
view stays an ordinary function that a test can run directly instead of a
script only Streamlit knows how to execute.

**Nothing here computes and nothing here predicts.** The rule the whole package
is arranged around: tables come from :mod:`src.pipelines`, results from the
match table, probabilities from the service over HTTP. ``dashboard`` imports
``src``; it does not import ``api``, and CI enforces both halves.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run dashboard/app.py` puts **this file's directory** on `sys.path`,
# not the project root — so `import dashboard` has nothing to resolve against
# and the first session dies with `ModuleNotFoundError: No module named
# 'dashboard'`. The container never saw it because its `PYTHONPATH=/app` says
# the same thing this line does, and the test suite never saw it because pytest
# puts the rootdir on the path before any of it runs.
#
# Fixed here rather than in the Makefile because `streamlit run
# dashboard/app.py` is the documented command and people type it directly; a
# target that exported `PYTHONPATH` would fix the one invocation that already
# had a wrapper and leave the bare one broken.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from dashboard import context, theme  # noqa: E402
from dashboard.domain import competition as catalogue  # noqa: E402
from dashboard.domain import favourites  # noqa: E402
from dashboard.services import history, matchday  # noqa: E402
from dashboard.views import competitions, home, match, performance, search  # noqa: E402

TITLE = "Match Outcome Predictor"
ICON = "⚽"

PAGES = (
    (home.render, "Home", ":material/home:", ""),
    (home.render_live, "Live centre", ":material/sensors:", "live"),
    (competitions.render, "Competitions", ":material/trophy:", "competitions"),
    (match.render, "Match", ":material/analytics:", "match"),
    (search.render, "Search", ":material/search:", "search"),
    (performance.render, "Model", ":material/query_stats:", "model"),
)
"""Every page, in the order the sidebar lists them.

The first is the default and owns ``/``. ``match`` and ``competitions`` are
reached by a card or a link carrying a query parameter far more often than by
the sidebar, and they are listed anyway: a page nobody can navigate to directly
is a page that is unreachable the moment a link is wrong.
"""


def sidebar(ctx: context.Context) -> None:
    """What the reader follows, and whether the two data paths are answering.

    Rendered on every page, from here rather than from each view, because it is
    the application's chrome and six copies of it would drift on the first
    change.
    """
    with st.sidebar:
        st.markdown(f"### {ICON} {TITLE}")
        st.caption("Calibrated home / draw / away probabilities for football.")

        followed = st.multiselect(
            "Competitions you follow",
            options=[one.id for one in catalogue.competitions()],
            default=favourites.leagues(),
            format_func=catalogue.short_label,
            placeholder="All competitions",
        )
        if followed != favourites.leagues():
            favourites.remember_leagues(followed)
            st.rerun()

        clubs = favourites.teams()
        if clubs:
            st.markdown("**Clubs you follow**")
            for club in clubs:
                if st.button(f"✕  {club}", key=f"unfollow-{club}", width="stretch"):
                    favourites.toggle_team(club)
                    st.rerun()
        else:
            st.caption("Follow a club from Search to see it first everywhere.")

        st.divider()
        _status(ctx)
        st.caption(
            "Favourites last as long as this browser tab. Accounts and saved "
            "preferences are Milestone 14."
        )


def _status(ctx: context.Context) -> None:
    """The two things that can be missing, each naming the command that fixes it.

    A dashboard whose sections are quietly empty is a dashboard a reader
    debugs. There are three reasons for an empty section here — no match table,
    no service, no fixture feed — and each is one line carrying the provider's
    own words rather than this module's guess at them.
    """
    if ctx.has_matches:
        latest = history.latest_date(ctx.matches_path)
        st.caption(f"✓ Match table · through {latest:%d %b %Y}" if latest else "✓ Match table")
    else:
        st.caption("✕ No match table — run `make data`")

    if ctx.predictions.available:
        st.caption("✓ Prediction service")
    else:
        st.caption(f"✕ Prediction service — {ctx.predictions.error or 'not answering'}")

    if ctx.fixtures.available:
        st.caption(f"✓ Fixture feed · {ctx.fixtures.name}")
    else:
        st.caption(f"✕ No fixture feed — {matchday.reason(ctx.fixtures)}")


def main() -> None:
    """Set the page up, render the chrome, run the page the reader is on."""
    st.set_page_config(
        page_title=TITLE,
        page_icon=ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    theme.inject()
    sidebar(context.resolve())
    st.navigation(
        [
            st.Page(render, title=title, icon=icon, url_path=path or None, default=not path)
            for render, title, icon, path in PAGES
        ]
    ).run()


main()
