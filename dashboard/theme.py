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

LIVE: Final = "#ef4444"

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

.mop-card-top {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.72rem;
  color: var(--muted);
  margin-bottom: 0.6rem;
}}
.mop-card-top .spacer {{ flex: 1; }}

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
  a.mop-card {{ transition: none; }}
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
.mop-form i.D {{ background: var(--draw); color: var(--ink); }}
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
.mop-section h3 {{ margin: 0; font-size: 1.02rem; font-weight: 700; }}
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
