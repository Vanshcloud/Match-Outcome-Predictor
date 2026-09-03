"""What these two teams have done to each other before.

A small family — two columns — and the one whose bookkeeping is easiest to get
wrong, because a pair's history is symmetric and the feature is not. "Points
the home team took" depends on which end of the fixture that team was at last
time, and the two are not related by any arithmetic: a draw is one point for
both, so you cannot recover one side's record by subtracting the other's from
three.

So both sides' records are carried through the window and the right one is
selected per row. The alternative — rolling only the home team's points — would
silently mean "points taken by whichever team happens to be home *this* time,
in matches where it may have been away", which is a different and wrong thing.

Meetings count across competitions, like form: two clubs that met in a cup last
month have met, and a per-competition window would say otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering.registry import FEATURES, KEY_COLUMN, Feature
from src.feature_engineering.windows import group_slices, prior_counts, trailing_mean

H2H_WINDOW = 5
"""Meetings in the window. Deliberately the same as the form window, and
deliberately small: two clubs meet twice a season, so five meetings already
reach back two or three years, and further than that is a different squad."""


def _points(goals_for: np.ndarray, goals_against: np.ndarray) -> np.ndarray:
    points = np.where(
        goals_for > goals_against, 3.0, np.where(goals_for == goals_against, 1.0, 0.0)
    )
    return np.where(np.isnan(goals_for) | np.isnan(goals_against), np.nan, points)


class HeadToHeadFeatures:
    """Prior meetings between the two teams, and how the home side fared."""

    name = "head_to_head"
    features: tuple[Feature, ...] = tuple(
        feature for feature in FEATURES if feature.group == "head_to_head"
    )

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        if matches.empty:
            return pd.DataFrame({column: [] for column in _SCHEMA}).astype(_SCHEMA)

        home = matches["home_team_id"].to_numpy(dtype=object)
        away = matches["away_team_id"].to_numpy(dtype=object)
        # The pair key is order-independent, so a fixture and its reverse share
        # a history. Which of the two is "first" is arbitrary and only has to be
        # stable.
        first = np.minimum(home, away)
        second = np.maximum(home, away)
        home_is_first = home == first

        goals_home = matches["home_goals"].astype("Float64").to_numpy(dtype="float64")
        goals_away = matches["away_goals"].astype("Float64").to_numpy(dtype="float64")
        home_points = _points(goals_home, goals_away)
        away_points = _points(goals_away, goals_home)

        frame = pd.DataFrame(
            {
                KEY_COLUMN: matches[KEY_COLUMN].to_numpy(),
                "date": matches["date"].to_numpy(),
                "pair": pd.Series(first, dtype="string") + "|" + pd.Series(second, dtype="string"),
                "home_is_first": home_is_first,
                "first_points": np.where(home_is_first, home_points, away_points),
                "second_points": np.where(home_is_first, away_points, home_points),
            }
        ).sort_values(["pair", "date", KEY_COLUMN], kind="stable")

        dates = frame["date"].to_numpy()
        first_values = frame["first_points"].to_numpy(dtype="float64")
        second_values = frame["second_points"].to_numpy(dtype="float64")
        meetings = np.empty(len(frame), dtype="int64")
        first_form = np.empty(len(frame), dtype="float64")
        second_form = np.empty(len(frame), dtype="float64")

        for start, stop in group_slices(frame["pair"].reset_index(drop=True)):
            prior = prior_counts(dates[start:stop])
            meetings[start:stop] = prior
            first_form[start:stop] = trailing_mean(first_values[start:stop], prior, H2H_WINDOW)
            second_form[start:stop] = trailing_mean(second_values[start:stop], prior, H2H_WINDOW)

        frame["h2h_matches"] = meetings
        frame["h2h_home_points"] = np.where(
            frame["home_is_first"].to_numpy(), first_form, second_form
        )

        out = frame.set_index(KEY_COLUMN)[["h2h_matches", "h2h_home_points"]]
        out = out.reindex(matches[KEY_COLUMN]).rename_axis(KEY_COLUMN).reset_index()
        return out.astype(_SCHEMA)


_SCHEMA: dict[str, str] = {
    KEY_COLUMN: "string",
    **{feature.name: feature.dtype for feature in HeadToHeadFeatures.features},
}
