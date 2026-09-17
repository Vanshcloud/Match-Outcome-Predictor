"""The look: colour tokens, and the stylesheet Streamlit has no setting for.

Streamlit's own theme lives in ``.streamlit/config.toml`` and owns the chrome —
sidebar, widgets, dataframe grid. It has no setting for a match card, so this
module supplies one stylesheet and the tokens that stylesheet is written in.

**Dark first, and only.** Not a preference: a page somebody opens on a Saturday
evening to watch scores is a page that should not flash white, and a second
palette is a second set of contrast ratios to check. The light variant is the
place to add one when there is a reader asking for it.

**The colours carry meaning and are checked against a colour-blind reader.**
Home, draw and away are green, slate and amber — not the red/green pair the
outcome labels invite, which is the single most common way a football
visualisation becomes unreadable for one viewer in twelve. The three are
distinguishable by lightness alone, so the probability bar survives being
printed in greyscale.
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from dashboard.domain.match import OUTCOMES

# ---- tokens ------------------------------------------------------------------
# Repeated in .streamlit/config.toml, which Streamlit reads for its chrome. The
# two must agree; there is no third place that feeds both.

INK: Final = "#e2e8f0"
"""Body text on the page background."""

MUTED: Final = "#94a3b8"
"""Captions, labels and anything a reader should be able to skip."""

SURFACE: Final = "#111a2e"
"""A card. One step up from the page, never a border alone — a card that is
only an outline disappears at the bottom of a long scroll."""

SURFACE_RAISED: Final = "#18233c"
"""A card under the cursor, and the strip a section header sits on."""

LINE: Final = "#233150"

HOME: Final = "#22c55e"
DRAW: Final = "#64748b"
AWAY: Final = "#f59e0b"
"""The three outcomes, in one order everywhere: home, draw, away.

The same order as ``src.evaluation.metrics.CLASSES``, deliberately. A page that
showed them in a different order from the array the model returns is a page
where a reader checking one against the other reads the wrong number.
"""

OUTCOME_COLOURS: Final[dict[str, str]] = dict(zip(OUTCOMES, (HOME, DRAW, AWAY), strict=True))

CONFIDENCE_COLOURS: Final[dict[str, str]] = {
    "high": "#22c55e",
    "moderate": "#38bdf8",
    "low": "#94a3b8",
}
"""How firmly the model committed, not how often it is right.

Three steps rather than a continuous scale, because the underlying quantity —
the largest of three probabilities — is bounded below by a third and a reader
reading a continuous scale over that range reads it as "0 to 1".
"""

LIVE: Final = "#dc2626"
"""The live pill, which is white text on this colour.

A step darker than the obvious red: white on ``#ef4444`` is 3.8:1, and the pill
is small bold text, which WCAG AA asks 4.5:1 of. This is 4.8:1, and still 3.9:1
against the page behind it — a pill is a graphic, and 3:1 is the bar for one.
"""

STYLESHEET: Final = f"""
<style>
:root {{
  --ink: {INK};
  --muted: {MUTED};
  --surface: {SURFACE};
  --raised: {SURFACE_RAISED};
  --line: {LINE};
  --home: {HOME};
  --draw: {DRAW};
  --away: {AWAY};
  --live: {LIVE};
}}

/* Streamlit's default block padding is built for a document. This is a board
   of cards, and the extra 3rem at the top is a screenful of nothing on a
   laptop. */
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1400px; }}

/* ---- cards ---------------------------------------------------------------- */

a.mop-card {{
  display: block;
  text-decoration: none;
  color: var(--ink);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 0.85rem 1rem 0.95rem;
  margin-bottom: 0.7rem;
  transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
}}
a.mop-card:hover {{
  background: var(--raised);
  border-color: #33456d;
  transform: translateY(-1px);
}}
/* Keyboard users get the same affordance the mouse gets. Streamlit's own
   focus ring does not reach an anchor written into markdown. */
a.mop-card:focus-visible {{ outline: 2px solid var(--home); outline-offset: 2px; }}

/* Up to --mop-cols columns, none narrower than --mop-min: for match cards,
   three beside the sidebar on a wide screen, two at 1280px and 1024px, one on
   a phone. `auto-fill` keeps empty tracks, so a country with one competition
   gets one tile rather than one stretched across the page. A row's cards
   stretch to the same height, so a header that needs two lines does not
   stagger the row. */
.mop-grid {{
  --mop-cols: 3;
  --mop-min: 17rem;
  display: grid;
  gap: 0.7rem;
  grid-template-columns: repeat(
    auto-fill,
    minmax(max(var(--mop-min), calc((100% - (var(--mop-cols) - 1) * 0.7rem) / var(--mop-cols))), 1fr)
  );
}}
.mop-grid > a.mop-card {{ margin-bottom: 0; }}

/* When the competition, the live pill and the kick-off do not fit on one line,
   the kick-off moves to the next line whole rather than breaking mid-time. */
.mop-card-top {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.3rem 0.5rem;
  font-size: 0.72rem;
  color: var(--muted);
  margin-bottom: 0.6rem;
}}
.mop-card-top .spacer {{ flex: 1; }}
.mop-card-top .mop-when {{ white-space: nowrap; }}

.mop-side {{
  display: flex;
  align-items: center;
  gap: 0.55rem;
  min-width: 0;
}}
.mop-team {{
  font-size: 0.95rem;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.mop-score {{
  font-variant-numeric: tabular-nums;
  font-size: 1.05rem;
  font-weight: 700;
  margin-left: auto;
  padding-left: 0.6rem;
}}
.mop-row {{ display: flex; align-items: center; margin: 0.18rem 0; }}
.mop-row.dim {{ opacity: 0.62; }}

/* ---- crest --------------------------------------------------------------- */

.mop-crest {{
  flex: 0 0 auto;
  width: 26px;
  height: 26px;
  border-radius: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  color: #0b1120;
}}

/* ---- pills --------------------------------------------------------------- */

.mop-pill {{
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  border-radius: 999px;
  padding: 0.1rem 0.5rem;
  font-size: 0.68rem;
  font-weight: 600;
  border: 1px solid var(--line);
  color: var(--muted);
  white-space: nowrap;
}}
/* `ui.link` is an anchor wearing the pill class, and Streamlit styles every
   anchor inside markdown blue and underlined — so without this the "Open →"
   on Competitions and the shortcuts on Search render as raw links inside a
   pill border instead of as the quiet chips every other pill is. */
a.mop-pill {{
  text-decoration: none;
  color: var(--ink);
  background: var(--surface);
  transition: background 120ms ease, border-color 120ms ease;
}}
a.mop-pill:hover {{ background: var(--raised); border-color: #33456d; }}
a.mop-pill:focus-visible {{ outline: 2px solid var(--home); outline-offset: 2px; }}

.mop-pill.live {{
  color: #fff;
  background: var(--live);
  border-color: var(--live);
}}
.mop-pill.live::before {{
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #fff;
  animation: mop-pulse 1.4s ease-in-out infinite;
}}
@keyframes mop-pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.25; }} }}
/* A reader who has asked their system for less motion is asking about this. */
@media (prefers-reduced-motion: reduce) {{
  .mop-pill.live::before {{ animation: none; }}
  a.mop-card, a.mop-pill {{ transition: none; }}
}}

/* ---- competition list ----------------------------------------------------- */

/* A flag and a name per row, no card: the list is the browser, two columns
   across the page's width (one on a phone). Wraps rather than truncating:
   "Copa de la Liga Profesional" is the name. */
.mop-leagues {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.6rem 1.5rem;
  margin-top: 1.5rem;
}}
@media (max-width: 640px) {{ .mop-leagues {{ grid-template-columns: 1fr; }} }}
a.mop-league {{
  display: flex;
  align-items: center;
  gap: 1.5rem;
  padding: 1.1rem 1.5rem;
  border-radius: 14px;
  background: var(--surface);
  border: 1px solid var(--line);
  color: var(--ink);
  text-decoration: none;
  font-size: 1.6rem;
  font-weight: 700;
  overflow-wrap: anywhere;
  transition: background 120ms ease, border-color 120ms ease;
}}
a.mop-league:hover {{ background: var(--raised); border-color: #33456d; }}
a.mop-league:focus-visible {{ outline: 2px solid var(--home); outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ a.mop-league {{ transition: none; }} }}
/* A flat flag in a rounded white frame. */
.mop-flag {{
  flex: 0 0 auto;
  width: 3.4rem;
  height: 2.4rem;
  object-fit: cover;
  border: 3px solid #f8fafc;
  border-radius: 9px;
  background: var(--raised);
}}

/* ---- sidebar menu --------------------------------------------------------- */

/* Streamlit's page menu, as one rounded panel of large bold entries. Its test
   ids are the only hooks it offers; if a release renames them the menu falls
   back to Streamlit's own look rather than breaking. */
[data-testid="stSidebarNavItems"] {{
  background: var(--raised);
  border-radius: 16px;
  padding: 0.5rem;
}}
/* The header above the menu only holds the collapse arrow; Streamlit gives it
   a logo's height, which left a blank band over Home. */
[data-testid="stSidebarHeader"] {{ height: 2.5rem; min-height: 0; margin-bottom: 0; padding-bottom: 0; }}
/* The panels' gaps separate the sections, so the rules between them go. */
[data-testid="stSidebarNavSeparator"] {{ display: none; }}
[data-testid="stSidebarNav"] {{ border-bottom: none; padding-bottom: 0; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 1.2rem; }}
.st-key-mop-panel-you, .st-key-mop-panel-status {{
  background: var(--raised);
  border-radius: 16px;
  padding: 1rem;
}}
[data-testid="stSidebarNavLink"] {{
  gap: 0.9rem;
  padding: 0.7rem 0.9rem;
  border-radius: 10px;
}}
[data-testid="stSidebarNavLink"] span {{
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--ink);
}}
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] {{
  font-size: 1.5rem;
  color: var(--muted);
}}

/* ---- probability bar ------------------------------------------------------ */

.mop-bar {{
  display: flex;
  height: 8px;
  border-radius: 999px;
  overflow: hidden;
  margin-top: 0.7rem;
  background: #0b1120;
}}
.mop-bar span {{ display: block; height: 100%; }}
.mop-legend {{
  display: flex;
  justify-content: space-between;
  font-size: 0.7rem;
  font-variant-numeric: tabular-nums;
  color: var(--muted);
  margin-top: 0.3rem;
}}
.mop-legend b {{ color: var(--ink); font-weight: 600; }}

/* ---- form string ---------------------------------------------------------- */

.mop-form {{ display: inline-flex; gap: 3px; }}
.mop-form i {{
  font-style: normal;
  width: 18px;
  height: 18px;
  border-radius: 5px;
  font-size: 0.62rem;
  font-weight: 700;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #0b1120;
}}
.mop-form i.W {{ background: var(--home); }}
/* White rather than `--ink` on the draw swatch, for the reason `LIVE` is a
   step darker than the obvious red: this is 10px bold text, which WCAG AA asks
   4.5:1 of, and `--ink` on `--draw` is 3.9:1. White on it is 4.8:1. The
   near-black the other two squares use is 4.0:1 here, so neither inherited
   colour clears the bar and this one is stated. */
.mop-form i.D {{ background: var(--draw); color: #fff; }}
.mop-form i.L {{ background: var(--away); }}

/* ---- section head --------------------------------------------------------- */

.mop-section {{
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  margin: 1.4rem 0 0.7rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--line);
}}
.mop-section h3 {{ margin: 0; font-size: 1.15rem; font-weight: 700; }}
.mop-section span {{ font-size: 0.76rem; color: var(--muted); }}

/* ---- placeholder ---------------------------------------------------------- */

.mop-placeholder {{
  border: 1px dashed var(--line);
  border-radius: 14px;
  padding: 1.1rem 1.2rem;
  color: var(--muted);
  font-size: 0.84rem;
  line-height: 1.55;
}}
.mop-placeholder b {{ color: var(--ink); }}
</style>
"""


def inject() -> None:
    """Put the stylesheet on the page. Called once, by the shell."""
    st.markdown(STYLESHEET, unsafe_allow_html=True)
