"""What each team brings to a match: recent form, and how hard its month has been.

Two families, one computation. Both are windows over a team's own earlier
appearances, so both fall out of the same pass over the long frame — and doing
them together means the frame is built and sorted once rather than twice.

**Form spans competitions.** A club's last five matches are its last five,
whether they were league or cup fixtures. The alternative — a per-competition
window — restarts a promoted team's form at zero and pretends a midweek cup tie
never happened, which is not what a form table means and not what the players
experienced.

**Venue form is separate on purpose.** Home and away form differ by more than
the average home advantage: some sides are transformed by their own ground and
some travel well, and a single form figure averages exactly that signal away.
Elo already carries a *global* home-advantage constant, so this is the part it
cannot express.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering.registry import FEATURES, KEY_COLUMN, Feature
from src.feature_engineering.windows import (
    days_since_previous,
    group_slices,
    long_form,
    prior_counts,
    trailing_mean,
    trailing_within_days,
)

FORM_WINDOW = 5
"""Matches in a form window. Five is what a form table shows, and it is short
enough to mean "lately" — Elo already carries the long memory, so a second long
window here would mostly restate it."""

CONGESTION_DAYS = 14
"""A fortnight: long enough to contain a midweek round, short enough that a
quiet period reads as one."""

# Which long-frame column each rolling feature averages, and the suffix it
# lands under. Written once so a new form feature is one entry rather than
# three edits in three places.
_ROLLING: tuple[tuple[str, str], ...] = (
    ("points", "form_points_5"),
    ("goals_for", "goals_for_5"),
    ("goals_against", "goals_against_5"),
    ("shots_for", "shots_for_5"),
    ("shots_against", "shots_against_5"),
)


class TeamHistoryFeatures:
    """Form and schedule features for both sides of every match."""

    name = "team_history"
    features: tuple[Feature, ...] = tuple(
        feature for feature in FEATURES if feature.group in {"form", "schedule"}
    )

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return one row per match, keyed by ``match_id``."""
        if matches.empty:
            return pd.DataFrame({column: [] for column in _SCHEMA}).astype(_SCHEMA)

        long = long_form(matches)
        self._add_team_windows(long)
        self._add_venue_windows(long)

        # Split back into two halves and join, rather than pivot: the names the
        # registry knows are `home_form_points_5`, and building them directly
        # keeps the mapping from long column to feature name in one readable
        # place instead of inside a MultiIndex.
        sides = []
        for venue in ("home", "away"):
            side = long[long["venue"] == venue].set_index(KEY_COLUMN)[list(_SIDED_SUFFIXES)]
            side.columns = pd.Index(f"{venue}_{suffix}" for suffix in _SIDED_SUFFIXES)
            sides.append(side)

        wide = sides[0].join(sides[1])
        wide = wide.reindex(matches[KEY_COLUMN]).rename_axis(KEY_COLUMN).reset_index()
        return wide.reindex(columns=list(_SCHEMA)).astype(_SCHEMA)

    @staticmethod
    def _add_team_windows(long: pd.DataFrame) -> None:
        """Everything computed over a team's own earlier appearances."""
        dates = long["date"].to_numpy()
        played = np.empty(len(long), dtype="int64")
        rest = np.empty(len(long), dtype="float64")
        congestion = np.empty(len(long), dtype="int64")
        rolled = {name: np.empty(len(long), dtype="float64") for _, name in _ROLLING}
        sources = {name: long[column].to_numpy(dtype="float64") for column, name in _ROLLING}

        for start, stop in group_slices(long["team"]):
            window_dates = dates[start:stop]
            prior = prior_counts(window_dates)
            played[start:stop] = prior
            rest[start:stop] = days_since_previous(window_dates, prior)
            congestion[start:stop] = trailing_within_days(window_dates, prior, CONGESTION_DAYS)
            for name, values in sources.items():
                rolled[name][start:stop] = trailing_mean(values[start:stop], prior, FORM_WINDOW)

        long["matches_played"] = played
        long["rest_days"] = rest
        long["matches_14d"] = congestion
        for name, values in rolled.items():
            long[name] = values

    @staticmethod
    def _add_venue_windows(long: pd.DataFrame) -> None:
        """Form restricted to the venue this match is at.

        Computed on a second ordering and written back by position, rather than
        by re-sorting the frame: every other column is already aligned to the
        team ordering, and re-sorting twice is how two columns end up describing
        different rows.
        """
        order = long.sort_values(["team", "venue", "date", "match_id"], kind="stable").index
        reordered = long.loc[order]
        dates = reordered["date"].to_numpy()
        points = reordered["points"].to_numpy(dtype="float64")
        venue_points = np.empty(len(reordered), dtype="float64")

        keys = reordered["team"].astype(str) + "|" + reordered["venue"].astype(str)
        for start, stop in group_slices(keys.reset_index(drop=True)):
            prior = prior_counts(dates[start:stop])
            venue_points[start:stop] = trailing_mean(points[start:stop], prior, FORM_WINDOW)

        long["venue_points_5"] = pd.Series(venue_points, index=order).reindex(long.index)


# The long-frame column names that become `{venue}_{suffix}` on the way out.
_SIDED_SUFFIXES: tuple[str, ...] = (
    "matches_played",
    *(name for _, name in _ROLLING),
    "venue_points_5",
    "rest_days",
    "matches_14d",
)


_SCHEMA: dict[str, str] = {
    KEY_COLUMN: "string",
    **{feature.name: feature.dtype for feature in TeamHistoryFeatures.features},
}
"""What this builder produces, taken from the registry rather than restated."""
