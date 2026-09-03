"""The storage contract.

Everything downstream — validation, features, evaluation, the API — reads
matches through this protocol rather than through ``pd.read_parquet``. The
difference matters exactly once: when the matches stop living in a file. At
that point a caller that opened a path has to be rewritten, and a caller that
asked a store for rows does not.

**Read-shaped on purpose.** There is no ``write_matches`` here, because nothing
but :mod:`src.pipelines.ingest` writes matches, and a protocol method with no
caller is a promise nobody is holding anyone to. The writing counterpart
arrives with the store that needs it — PostgreSQL, in the milestone that first
persists a prediction. Declaring it now would mean two implementations of a
method that is only ever called by tests.

A Protocol rather than an abstract base class, matching
:class:`src.ingestion.base.MatchProvider`: implementations share a shape, not
code, and structural typing means a store does not have to import this module
to satisfy it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
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
