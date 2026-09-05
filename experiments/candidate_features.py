"""Candidate features from canonical columns the design matrix does not read.

Milestone 5 built twenty features and Milestone 8 measured what each block was
worth. Nothing since has asked whether the *canonical table* still holds signal
the thirty design columns ignore. It does: half-time scores (71% coverage),
shots on target (42%), a longer form window, the competition's tier, and how
far into a season a match falls are all knowable before kick-off and none of
them reaches the model.

Every column here is a window over rows strictly earlier **by date**, built
with :mod:`src.feature_engineering.windows` rather than a second implementation
of it, so :mod:`src.validation.temporal`'s probes apply unchanged and were run:
prefix invariance holds over 4 cutoffs and outcome independence over 3
rewrites.

**One of these leaked on the first attempt and the probe caught it.** The
original ``season_progress`` divided a match's position in its season by the
season's *total* length — a whole-group statistic, unknowable at kick-off. It
looked causal, it passed review, and prefix invariance rejected it at all four
cutoffs on the first run. It is ``season_days`` below instead: elapsed days
since the season's first match, which is in the past at every later kick-off.
That is the single best argument in this repository for testing a derivation
rather than reading it, and it is recorded here rather than quietly fixed.

This module is deliberately **not** in ``src``. Adopting these features means
registry entries, a rebuilt feature table and every reported number
regenerated; until that milestone runs, this is evidence, not production code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering.windows import (
    group_slices,
    long_form,
    prior_counts,
    trailing_mean,
)

ROLL: dict[str, tuple[str, int]] = {
    "ht_points_5": ("ht_points", 5),
    "ht_goals_for_5": ("ht_for", 5),
    "sot_for_5": ("sot_for", 5),
    "sot_against_5": ("sot_against", 5),
    "form_points_20": ("points", 20),
    "goal_diff_20": ("goal_diff", 20),
}
"""Rolling windows to build per team-appearance, as ``name -> (source, window)``."""

CANDIDATE_COLUMNS: tuple[str, ...] = tuple(
    [f"{venue}_{name}" for venue in ("home", "away") for name in ROLL] + ["tier", "season_days"]
)
"""The fourteen columns, in the order the experiment adds them."""


def _long_extra(matches: pd.DataFrame) -> pd.DataFrame:
    """``long_form``, plus half-time and shots-on-target per team-appearance.

    Half-time is folded into points the same way full-time is, because "led at
    the break" is the comparable statement and a scoreline is two numbers.
    """
    long = long_form(matches)
    sides = []
    for venue, ht_for, ht_against, sot_for, sot_against in (
        ("home", "ht_home_goals", "ht_away_goals", "home_shots_on_target", "away_shots_on_target"),
        ("away", "ht_away_goals", "ht_home_goals", "away_shots_on_target", "home_shots_on_target"),
    ):
        sides.append(
            pd.DataFrame(
                {
                    "match_id": matches["match_id"].to_numpy(),
                    "venue": venue,
                    "ht_for": matches[ht_for].astype("Float64").to_numpy(dtype="float64"),
                    "ht_against": matches[ht_against].astype("Float64").to_numpy(dtype="float64"),
                    "sot_for": matches[sot_for].astype("Float64").to_numpy(dtype="float64"),
                    "sot_against": matches[sot_against].astype("Float64").to_numpy(dtype="float64"),
                }
            )
        )
    extra = pd.concat(sides, ignore_index=True)
    extra["ht_points"] = np.select(
        [extra["ht_for"] > extra["ht_against"], extra["ht_for"] == extra["ht_against"]],
        [3.0, 1.0],
        default=0.0,
    )
    # A match with no half-time score is not a half-time result to learn from.
    extra.loc[extra["ht_for"].isna(), "ht_points"] = np.nan
    return long.merge(extra, on=["match_id", "venue"], how="left")


def _season_days(matches: pd.DataFrame) -> pd.Series:
    """Days elapsed since the first match of this competition-season.

    Causal, where the obvious version is not. Dividing by the season's total
    length would consult how many matches the season *will* contain, which is
    the leak :mod:`src.validation.temporal` rejected — see the module
    docstring. The first match of a season is in the past at every later
    kick-off, so a gap measured from it is knowable.
    """
    frame = matches[["match_id", "competition_id", "season", "date"]].sort_values(
        ["competition_id", "season", "date", "match_id"], kind="stable"
    )
    dates = frame["date"].to_numpy()
    elapsed = np.empty(len(frame), dtype="float64")
    keys = frame["competition_id"].astype(str) + "|" + frame["season"].astype(str)
    for start, stop in group_slices(keys.reset_index(drop=True)):
        window = dates[start:stop]
        elapsed[start:stop] = (window - window[0]) / np.timedelta64(1, "D")
    return pd.Series(elapsed, index=frame["match_id"])


def build_candidates(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per ``match_id``, keyed like every other feature table.

    Args:
        matches: The canonical table, sorted by date.
    """
    if matches.empty:
        return pd.DataFrame({"match_id": []})

    long = _long_extra(matches)
    long["goal_diff"] = long["goals_for"] - long["goals_against"]
    long = long.sort_values(["team", "date", "match_id"], kind="stable").reset_index(drop=True)

    dates = long["date"].to_numpy()
    rolled = {name: np.empty(len(long), dtype="float64") for name in ROLL}
    sources = {name: long[column].to_numpy(dtype="float64") for name, (column, _) in ROLL.items()}
    for start, stop in group_slices(long["team"]):
        prior = prior_counts(dates[start:stop])
        for name, (_, window) in ROLL.items():
            rolled[name][start:stop] = trailing_mean(sources[name][start:stop], prior, window)
    for name, values in rolled.items():
        long[name] = values

    sides = []
    for venue in ("home", "away"):
        side = long[long["venue"] == venue].set_index("match_id")[list(ROLL)]
        side.columns = pd.Index(f"{venue}_{name}" for name in ROLL)
        sides.append(side)
    wide = sides[0].join(sides[1])

    # Both match-level and knowable before kick-off: a competition's tier is a
    # fact about the competition, and the season's start is in the past.
    wide["tier"] = matches.set_index("match_id")["tier"].astype("Float64").reindex(wide.index)
    wide["season_days"] = _season_days(matches).reindex(wide.index)

    return wide.reindex(matches["match_id"]).rename_axis("match_id").reset_index()
