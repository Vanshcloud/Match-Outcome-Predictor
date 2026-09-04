"""The storage contract.

Everything downstream — validation, features, evaluation, the API — reads
matches through this protocol rather than through ``pd.read_parquet``. The
difference matters exactly once: when the matches stop living in a file. At
that point a caller that opened a path has to be rewritten, and a caller that
asked a store for rows does not.

**Read-shaped on purpose.** There is no ``write_matches`` here, because nothing
but :mod:`src.pipelines.ingest` writes matches, and a protocol method with no
caller is a promise nobody is holding anyone to. The writing counterpart is
:class:`PredictionLog`, added by Milestone 11 — the milestone that first has
something to persist. It is a *separate* protocol rather than two more methods
on this one: a served prediction is application state with a different shape,
a different lifetime and a different store, and a store that had to implement
both to satisfy either would be a store nobody could write.

A Protocol rather than an abstract base class, matching
:class:`src.ingestion.base.MatchProvider`: implementations share a shape, not
code, and structural typing means a store does not have to import this module
to satisfy it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import date

    import pandas as pd


class StorageError(RuntimeError):
    """A store could not answer. Distinct from an empty result, which is data."""


@runtime_checkable
class MatchStore(Protocol):
    """Read access to the canonical match table."""

    def read_matches(
        self,
        *,
        competitions: Sequence[str] | None = None,
        since: date | str | None = None,
        until: date | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return matches in :data:`src.ingestion.base.CANONICAL_SCHEMA` order.

        ``since`` and ``until`` are inclusive bounds on ``date``. They exist on
        the interface rather than being left to the caller's ``df[df.date < x]``
        because a point-in-time read is the operation every temporal split and
        every backtest performs, and a store that can push the bound down to
        the query engine should be allowed to.

        Implementations must re-assert the canonical dtypes on the way out. A
        query engine returns whatever its own type system produces, and for
        DuckDB that is a nullable ``Int16`` for a column that happens to
        contain a null and a NumPy ``int16`` for one that does not — so the
        dtype of ``home_goals`` would depend on the rows selected.
        """
        ...

    def count(self) -> int:
        """Number of matches held, without materialising them."""
        ...


@runtime_checkable
class PredictionLog(Protocol):
    """Append-only record of what the service said, and what it said it from.

    The reason this exists is stated in the plan and is worth restating here:
    the backtest measures the model over folds, and a *served* model is a
    different thing — a different training window, a different population of
    requests, and outcomes that arrive later. Writing every prediction down
    with the inputs it was made from is what lets the served calibration be
    measured against what happened rather than assumed to match the backtest.

    The outcome is deliberately not a column. It lives in the canonical match
    table already, and a second copy is a second thing to keep in step; a
    consumer joins on ``match_id``.
    """

    enabled: bool
    """Whether this log actually stores anything.

    On the protocol rather than discovered with ``getattr`` at the call site,
    because "is the audit trail on?" is a question the API answers in
    ``/health`` and ``/version`` and one an implementation should have to
    answer rather than be inspected for.
    """

    def ensure_schema(self) -> None:
        """Create whatever the log needs, if it is not there. Idempotent."""
        ...

    def record(self, predictions: Sequence[Mapping[str, Any]]) -> int:
        """Append predictions. Returns how many rows were written."""
        ...

    def recent(self, limit: int = 20) -> pd.DataFrame:
        """The most recently written predictions, newest first."""
        ...

    def close(self) -> None:
        """Release the connection. Idempotent."""
        ...
