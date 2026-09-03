"""Walk-forward folds. Mostly: what is on which side of the boundary.

Every assertion here is about an inequality, because every leak available at
this layer is an inequality that was written the other way round.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.splits import (
    DEFAULT_FOLDS,
    Fold,
    SplitError,
    boundaries,
    walk_forward,
)
from src.validation.temporal import TemporalResult, split_boundary
from tests.factories import canonical_frame, league_frame, season_labels

LEAGUE = league_frame(seasons=season_labels(2012, 14), teams=20)


def _folds(frame: pd.DataFrame = LEAGUE, **kwargs: object) -> list[Fold]:
    return list(walk_forward(frame, **kwargs))  # type: ignore[arg-type]


# ---- the boundary -----------------------------------------------------------


def test_the_default_is_five_yearly_folds() -> None:
    assert len(_folds()) == DEFAULT_FOLDS


def test_no_training_match_is_dated_at_or_after_an_evaluation_match() -> None:
    """The property. Everything else in this file is a way of it failing."""
    for fold in _folds():
        assert fold.train["date"].max() < fold.evaluate["date"].min()


def test_every_fold_passes_the_boundary_probe() -> None:
    for fold in _folds():
        assert split_boundary(fold.train, fold.evaluate, name="check").ok


def test_a_day_is_never_split_across_the_boundary() -> None:
    """The cut is a timestamp, so a full Saturday programme stays together."""
    for fold in _folds():
        assert not set(fold.train["date"]) & set(fold.evaluate["date"])


def test_folds_are_yielded_oldest_first_and_do_not_overlap() -> None:
    folds = _folds()
    for earlier, later in zip(folds[:-1], folds[1:], strict=True):
        assert earlier.end == later.start
        assert not set(earlier.evaluate["match_id"]) & set(later.evaluate["match_id"])


def test_the_window_expands_rather_than_slides() -> None:
    folds = _folds()
    for earlier, later in zip(folds[:-1], folds[1:], strict=True):
        assert len(later.train) > len(earlier.train)
        assert set(earlier.train["match_id"]) <= set(later.train["match_id"])


def test_the_last_fold_contains_the_last_match() -> None:
    """An inclusive end would drop it; an exclusive bound one day later does not."""
    assert LEAGUE["date"].max() in set(_folds()[-1].evaluate["date"])


def test_the_boundaries_are_evenly_spaced_and_anchored_at_the_end() -> None:
    cuts = boundaries(LEAGUE, folds=3, horizon_days=100)
    assert len(cuts) == 4
    assert cuts[-1] == LEAGUE["date"].max() + pd.Timedelta(1, "D")
    steps = zip(cuts[:-1], cuts[1:], strict=True)
    assert all(later - earlier == pd.Timedelta(100, "D") for earlier, later in steps)


# ---- what it refuses --------------------------------------------------------


def test_an_unsorted_table_is_refused() -> None:
    """Sorting silently would produce folds that look chronological and are not."""
    with pytest.raises(SplitError, match="sorted by date"):
        _folds(LEAGUE.sort_values("match_id", ascending=False))


def test_an_empty_table_is_refused() -> None:
    with pytest.raises(SplitError, match="no matches to split"):
        boundaries(canonical_frame([]))


def test_zero_folds_is_refused() -> None:
    with pytest.raises(SplitError, match="at least one fold"):
        boundaries(LEAGUE, folds=0)


def test_a_fold_spanning_no_time_is_refused() -> None:
    with pytest.raises(SplitError, match="at least one day"):
        boundaries(LEAGUE, horizon_days=0)


def test_a_table_too_short_to_split_is_refused() -> None:
    """One season cannot train a fold and be scored by it at the same time."""
    with pytest.raises(SplitError, match="no fold had"):
        _folds(league_frame(seasons=["2020-21"], teams=6))


def test_a_fold_with_too_little_training_history_is_skipped() -> None:
    """Skipped, not yielded empty: a fold of forty matches is not evidence."""
    # Every fold but the last trains on less than this, so only the last runs.
    folds = _folds(min_train=len(LEAGUE) - 400)
    assert [fold.index for fold in folds] == [4]


def test_a_fold_with_nothing_to_score_is_skipped() -> None:
    """A gap in the fixture list is a fold with no evaluation matches."""
    gapped = LEAGUE[(LEAGUE["date"] < "2024-01-01") | (LEAGUE["date"] >= "2025-06-01")].reset_index(
        drop=True
    )
    assert all(len(fold.evaluate) > 0 for fold in _folds(gapped))


def test_a_fold_that_fails_the_probe_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate itself. The arithmetic above is four lines and obviously correct,
    which is exactly the kind of code that is quietly wrong after an edit."""
    monkeypatch.setattr(
        "src.models.splits.split_boundary",
        lambda *_, **__: TemporalResult(
            name="fold 0", probe="split boundary", checks=2, violations=("planted",)
        ),
    )
    with pytest.raises(SplitError, match="planted"):
        _folds()


# ---- reporting --------------------------------------------------------------


def test_a_fold_says_what_it_trained_on_and_what_it_scores() -> None:
    summary = _folds()[0].summary()
    assert "fold 0" in summary
    assert "train" in summary and "evaluate" in summary
