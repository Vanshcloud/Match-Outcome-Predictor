"""The rating contract, and the schema every rating model writes into.

A rating is the first thing in this project that has *memory*: its value for a
match depends on every match before it. That makes it the first place a leak
can hide, and the reason this module leads with the contract rather than with
an algorithm.

**One rule, stated once.** A model's output for match *n* may depend on matches
1..*n*-1 and on nothing else — not on match *n*'s own result, and not on any
statistic computed over the whole table. Two consequences follow, and both are
enforced rather than asserted in a docstring:

- :func:`require_chronological` — a model that is handed unsorted matches
  produces ratings that look fine and are silently wrong, so the input order is
  checked rather than assumed.
- :mod:`src.ratings.causality` — a model is *tested* by truncating its input
  and checking the answers for the surviving matches do not move.

**Ratings are pre-match by construction**, so every column here belongs to
:data:`~src.ingestion.base.PRE_MATCH_COLUMNS`' side of kick-off even though it
is derived from post-match data. That is not a contradiction: yesterday's goals
are knowable this morning. It is only true while the derivation is causal,
which is what the two mechanisms above exist to guarantee.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

KEY_COLUMN = "match_id"
"""Ratings join back to the canonical table on this and nothing else."""

# The columns each model contributes. Kept here rather than in the model
# modules so the assembled table has one definition, and so a consumer can ask
# what a partial run left null without importing an implementation.
ELO_COLUMNS: dict[str, str] = {
    "elo_home": "Float64",
    "elo_away": "Float64",
    "elo_expected_home": "Float64",
    "elo_home_played": "Int32",
    "elo_away_played": "Int32",
}

DIXON_COLES_COLUMNS: dict[str, str] = {
    "dc_home_lambda": "Float64",
    "dc_away_lambda": "Float64",
    "dc_prob_home": "Float64",
    "dc_prob_draw": "Float64",
    "dc_prob_away": "Float64",
}

RATINGS_SCHEMA: dict[str, str] = {
    KEY_COLUMN: "string",
    **ELO_COLUMNS,
    **DIXON_COLES_COLUMNS,
}

# Every rating column is nullable, and a null means the model could not produce
# a value at all rather than that it produced a poor one. Dixon-Coles before
# its first fit has nothing to say and says nothing; Elo before a team's first
# match has a prior (the initial rating) and reports it alongside the count of
# matches backing it, because "1500 because we have never seen them" and "1500
# because they are exactly average" are different facts and a consumer needs
# both distinguishable.


class RatingError(RuntimeError):
    """A rating model was asked for something it cannot honestly produce."""


def require_chronological(matches: pd.DataFrame) -> None:
    """Raise unless ``matches`` is ordered by date.

    Checked rather than sorted-for-you on purpose. Sorting silently would hide
    that a caller's pipeline had lost its ordering somewhere upstream, and the
    ordering is load-bearing for far more than ratings — every rolling feature
    and every temporal split downstream assumes it. Failing here names the
    problem at the point it is still cheap to find.

    Raises:
        RatingError: If the dates are not non-decreasing.
    """
    if matches.empty:
        return
    if not matches["date"].is_monotonic_increasing:
        raise RatingError(
            "matches must be sorted by date before rating; an unsorted input "
            "produces ratings that look plausible and are computed from the future"
        )


@runtime_checkable
class RatingModel(Protocol):
    """What every rating model must offer.

    A Protocol, matching :class:`~src.ingestion.base.MatchProvider` and
    :class:`~src.storage.base.MatchStore`: Elo and Dixon-Coles share a shape
    and no implementation at all — one is an online update, the other a
    periodic maximum-likelihood refit — so inheritance would buy nothing and
    structural typing costs nothing.
    """

    @property
    def name(self) -> str:
        """Short identifier, used in logs and in the ratings report."""
        ...

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """The columns of :data:`RATINGS_SCHEMA` this model fills."""
        ...

    def rate(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return one row of pre-match ratings per input match.

        Args:
            matches: The canonical table, sorted by date.

        Returns:
            A frame with :data:`KEY_COLUMN` and :attr:`feature_columns`, one
            row per input row, in input order. Values are what was knowable
            *before* each match kicked off. Where a model has nothing to say
            yet it returns null rather than a number nobody could have had.
        """
        ...
