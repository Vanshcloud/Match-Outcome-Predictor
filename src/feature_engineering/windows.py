"""Looking backwards, exactly once, and never past kick-off.

Every feature in this package is some summary of what happened *before* a
match. That makes them all the same computation with different inputs, and it
makes the correctness question a single one: does the window stop in time?

The obvious implementation is ``shift(1).rolling(k)``, and it is subtly wrong.
``shift`` counts *rows*, so two matches on the same date let the earlier row —
earlier only by an arbitrary tiebreak in the sort — inform the later one. That
is not a hypothetical: the mislabelled-division bug recorded in
``docs/DATA_SOURCES.md`` was found by asking whether a team ever plays twice on
one date, and before it was fixed the answer was 2,444 times.

So the primitive here cuts on the **date**, not on the row. For every row it
finds ``prior``, the number of that group's matches strictly earlier than this
row's date, and every window ends there. Same-day matches cannot see each
other, whatever order they happen to be sorted in — which is also what makes
these functions survive :mod:`src.validation.temporal`'s probes, since a
truncation that removes a same-day sibling then changes nothing.

Windows are counted in **matches, not days**: "the last five" is what a form
table means, and a five-match window over a fixture-congested fortnight and a
quiet month should cover the same amount of football.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prior_counts(dates: np.ndarray) -> np.ndarray:
    """For each row, how many rows of ``dates`` fall strictly earlier.

    ``dates`` must be sorted ascending. The result doubles as the exclusive end
    of every window: rows sharing a date all get the same value, which is what
    stops them seeing each other.
    """
    return np.searchsorted(dates, dates, side="left").astype("int64")


def trailing_mean(values: np.ndarray, prior: np.ndarray, window: int) -> np.ndarray:
    """Mean of up to ``window`` values from the rows strictly before each row.

    Missing values are skipped rather than counted as zero, and a row with no
    usable history returns NaN. Those are different facts — "no shots recorded
    in the last five matches" is not "zero shots" — and a feature that conflated
    them would teach a model that competitions without shot data are famously
    bad at shooting.

    Args:
        values: One value per row, aligned with ``prior``. NaN where absent.
        window: How many earlier rows to average over, at most.
    """
    start = np.maximum(0, prior - window)
    present = ~np.isnan(values)
    # Prefix sums with a leading zero, so a window is one subtraction. The
    # alternative — a rolling object per group — is an order of magnitude
    # slower over six hundred thousand rows and no clearer.
    totals = np.concatenate(([0.0], np.nancumsum(values)))
    counts = np.concatenate(([0], np.cumsum(present)))
    covered = counts[prior] - counts[start]
    summed = totals[prior] - totals[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(covered > 0, summed / np.maximum(covered, 1), np.nan)


def trailing_within_days(dates: np.ndarray, prior: np.ndarray, days: int) -> np.ndarray:
    """How many earlier rows fall within ``days`` before each row's date.

    Counts matches, so it answers "how congested has this team's fortnight
    been" without any of the current day's fixtures in it.
    """
    cutoff = dates - np.timedelta64(days, "D")
    return (prior - np.searchsorted(dates, cutoff, side="left")).astype("int64")


def days_since_previous(dates: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Days between each row's date and the most recent strictly earlier one.

    NaN for a group's first appearance, and for every row that shares a date
    with it. Rest is measured from the last match actually played, so a second
    fixture on the same day reports the same rest as the first rather than
    zero — the two are simultaneous, and neither preceded the other.
    """
    out = np.full(len(dates), np.nan)
    has_history = prior > 0
    previous = dates[np.maximum(prior - 1, 0)]
    gap = (dates - previous) / np.timedelta64(1, "D")
    out[has_history] = gap[has_history]
    return out


def long_form(matches: pd.DataFrame) -> pd.DataFrame:
    """Recast one row per match into one row per team-appearance.

    Every per-team feature is easier to state this way — a team's form is a
    window over its own rows, whether it was at home or away — and recombining
    is a join on ``(match_id, venue)``.

    The frame is sorted by ``(team, date, match_id)``. The tiebreak on
    ``match_id`` never decides anything, because the windows cut on date; it is
    there so the ordering is deterministic and two runs produce identical
    output.
    """
    columns = (
        "match_id",
        "date",
        "venue",
        "team",
        "opponent",
        "goals_for",
        "goals_against",
        "shots_for",
        "shots_against",
        "points",
    )
    if matches.empty:
        # Concatenating two empty frames with a datetime column asks pandas for
        # a NaT of a bare unit, which it deprecates and this suite treats as an
        # error. The builders short-circuit before they reach here, so this
        # guard is about the helper being usable on its own.
        return pd.DataFrame({column: [] for column in columns})

    sides = []
    for venue, team, opponent, goals_for, goals_against, shots_for, shots_against in (
        (
            "home",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
            "home_shots",
            "away_shots",
        ),
        (
            "away",
            "away_team_id",
            "home_team_id",
            "away_goals",
            "home_goals",
            "away_shots",
            "home_shots",
        ),
    ):
        sides.append(
            pd.DataFrame(
                {
                    "match_id": matches["match_id"].to_numpy(),
                    "date": matches["date"].to_numpy(),
                    "venue": venue,
                    "team": matches[team].to_numpy(),
                    "opponent": matches[opponent].to_numpy(),
                    "goals_for": matches[goals_for].astype("Float64").to_numpy(dtype="float64"),
                    "goals_against": matches[goals_against]
                    .astype("Float64")
                    .to_numpy(dtype="float64"),
                    "shots_for": matches[shots_for].astype("Float64").to_numpy(dtype="float64"),
                    "shots_against": matches[shots_against]
                    .astype("Float64")
                    .to_numpy(dtype="float64"),
                }
            )
        )
    long = pd.concat(sides, ignore_index=True)
    long["points"] = np.select(
        [long["goals_for"] > long["goals_against"], long["goals_for"] == long["goals_against"]],
        [3.0, 1.0],
        default=0.0,
    )
    # A match with no score is not a result to learn from. Ingestion drops
    # unplayed fixtures, so this only fires on a caller's own slice.
    long.loc[long["goals_for"].isna(), "points"] = np.nan
    return long.sort_values(["team", "date", "match_id"], kind="stable").reset_index(drop=True)


def group_slices(keys: pd.Series) -> list[tuple[int, int]]:
    """Start and stop offsets of each run of equal keys in a sorted column.

    The windows above are all per-group, and a Python loop over these slices
    beats ``groupby.apply`` by a wide margin here: the work inside is already
    vectorised, so the only thing a groupby adds is its own bookkeeping.
    """
    if keys.empty:
        return []
    values = keys.to_numpy()
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    edges = np.concatenate(([0], boundaries, [len(values)]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
