"""The shipped model as a *fitted* object, for a process that serves it.

Every backtested forecaster fits inside its own ``forecast(train,
evaluate)``. That is what makes the backtest causal by construction — a model
cannot see a match it was not handed, and the split layer decides what it is
handed — and it is exactly the wrong shape for a service, which has one process
answering many requests and no training half to hand it.

So this module holds the other half of the pair: the same composition, fitted
once. Members from :data:`~src.models.ensemble.MEMBERS`, blended by the mean
:class:`~src.models.ensemble.Ensemble` already takes, with the single
temperature :class:`~src.models.calibration.Calibrated` already fits, applied
by the same :func:`~src.models.calibration.apply`. Nothing about the arithmetic
is re-derived here — a served probability that differed from a backtested one
would make every number in docs/MODEL_CARD.md a claim about a different model.

**Still frames in, arrays out.** No path, no store, no format. Persisting one
of these is :mod:`src.pipelines.serving`'s job, which is the layer allowed to
know where things live; this file stays testable against a frame in memory and
stays inside the invariant CI enforces on ``src/models``.

**It knows what it was trained on, and says so.** ``trained_through`` is the
last date the fit could see. A fixture dated at or before it was in the
training half, so the probability for it is in-sample and is not the thing the
walk-forward folds measured — a distinction the caller is entitled to and
:meth:`ServableModel.is_in_sample` answers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.calibration import NEUTRAL, Calibrated, apply
from src.models.dataset import DESIGN_COLUMNS, DatasetError
from src.models.ensemble import MEMBERS, SHIPPED, ensemble
from src.models.zoo import Estimator, TrainedForecaster, build
from src.utils.logging import get_logger

logger = get_logger(__name__)

DATE_COLUMN = "date"


class ArtifactError(ValueError):
    """A servable model cannot be fitted or used as asked."""


@dataclass(frozen=True, slots=True)
class FittedMember:
    """One family of the blend, by name, with the estimator it fitted.

    **The name and the columns are stored; the forecaster is not.**
    :class:`~src.models.zoo.TrainedForecaster` holds its ``build`` as a lambda
    — which is the right shape for a zoo that wants a fresh estimator per fold
    and an unpicklable one for an artefact. Pickling it fails outright rather
    than quietly, which is how this was found; storing the name instead is both
    smaller and more honest, because ``build`` is a *factory* and an artefact
    that carried one would be carrying the ability to discard its own fit.

    :attr:`forecaster` rebuilds it from the zoo on demand. That is what keeps
    the two things the estimator does not know — which columns the matrix is
    made of, and the scatter that puts ``predict_proba`` back into
    :data:`~src.evaluation.metrics.CLASSES` order — as one implementation
    rather than two.
    """

    name: str
    columns: tuple[str, ...]
    estimator: Estimator

    @property
    def forecaster(self) -> TrainedForecaster:
        """The zoo's wrapper for this family, over this member's columns.

        Rebuilt rather than stored. It is a dataclass holding a lambda and two
        tuples, so constructing one costs nothing, and rebuilding means a
        family that has been removed from the zoo fails on load with
        :class:`~src.models.zoo.ModelError` naming it — rather than serving
        from an estimator nothing in the codebase describes any more.
        """
        return build(self.name, self.columns)

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        return self.forecaster.predict(self.estimator, fixtures)


@dataclass(frozen=True, slots=True)
class ServableModel:
    """A fitted blend with its temperature, and what it was fitted on.

    Frozen, because a model that can be mutated after loading is a model whose
    answer depends on which request arrived first.
    """

    name: str
    members: tuple[FittedMember, ...]
    temperature: float
    trained_matches: int
    trained_from: pd.Timestamp
    trained_through: pd.Timestamp

    @property
    def columns(self) -> tuple[str, ...]:
        """The design columns every member was fitted on.

        Read off the members rather than stored again: two copies of a column
        list is one copy that can disagree with the matrix the estimator
        actually saw, and the disagreement is silent — a model handed the wrong
        thirty columns in the wrong order returns a forecast that sums to one.
        """
        return self.members[0].columns

    @property
    def member_names(self) -> tuple[str, ...]:
        return tuple(member.name for member in self.members)

    def is_in_sample(self, when: pd.Timestamp) -> bool:
        """Whether a fixture on ``when`` was inside the training history.

        Not a guard — the caller may well want the number anyway, and a
        service that refused would be unable to price the very matches the
        backtest reports on. It is reported instead, because an in-sample
        probability is not what the walk-forward folds measured and a consumer
        that cannot tell the two apart will quote the wrong one.
        """
        return bool(pd.Timestamp(when) <= self.trained_through)

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        """Calibrated probabilities, one row per fixture, in CLASSES order.

        The mean of the members, then ``p ** (1/T)`` renormalised — the same
        two steps :class:`~src.models.ensemble.Ensemble` and
        :class:`~src.models.calibration.Calibrated` compose in the backtest,
        called rather than repeated.

        Raises:
            DatasetError: If a design column is missing from ``fixtures``. Left
                to propagate from :func:`~src.models.dataset.design_matrix`,
                which names the columns; catching it to re-raise something
                vaguer would lose the only detail worth having.
        """
        if fixtures.empty:
            raise ArtifactError("no fixtures to price")
        stated = np.mean([member.predict(fixtures) for member in self.members], axis=0)
        return apply(stated, self.temperature)


def fit_servable(
    matches: pd.DataFrame,
    *,
    members: Sequence[str] = MEMBERS,
    columns: Sequence[str] = DESIGN_COLUMNS,
    name: str = SHIPPED,
) -> ServableModel:
    """Fit every member on ``matches``, then the one temperature over them.

    The temperature comes from :meth:`~src.models.calibration.Calibrated.temperature`
    unchanged: it cuts the last year off the training history by *date*, refits
    the blend on what is left, scores the year it held back, and minimises log
    loss over one scalar. Refitting is the cost of the guarantee — a
    temperature fitted on predictions the members had already seen would come
    out near one and mean nothing.

    Args:
        matches: The joined modelling frame — canonical columns, ratings and
            features — sorted by date. Every row is trained on.
        members: Family names from the zoo. Defaults to the blend the model
            card is written about.
        columns: The design columns. Defaults to all thirty.
        name: What the served model calls itself.

    Raises:
        ArtifactError: If ``matches`` is empty, or carries no usable date.
    """
    if matches.empty:
        raise ArtifactError("cannot fit a servable model on an empty table")
    if DATE_COLUMN not in matches.columns:
        raise ArtifactError(f"the modelling frame has no {DATE_COLUMN!r} column")

    dates = pd.to_datetime(matches[DATE_COLUMN])
    if dates.isna().all():
        raise ArtifactError("every match has a null date; the training window is undefined")

    fitted = tuple(
        FittedMember(
            name=forecaster.name,
            columns=forecaster.columns,
            estimator=forecaster.fit(matches),
        )
        for forecaster in (build(one, columns) for one in members)
    )
    logger.info("fitted %d member(s) on %d matches", len(fitted), len(matches))

    temperature = Calibrated(ensemble(members, columns)).temperature(matches)
    if temperature == NEUTRAL:
        logger.warning(
            "temperature is %.1f: the holdout was too thin to fit one, so the "
            "served probabilities are the blend's own",
            NEUTRAL,
        )
    logger.info("temperature %.4f", temperature)

    return ServableModel(
        name=name,
        members=fitted,
        temperature=temperature,
        trained_matches=len(matches),
        trained_from=pd.Timestamp(dates.min()),
        trained_through=pd.Timestamp(dates.max()),
    )


__all__ = [
    "ArtifactError",
    "DatasetError",
    "FittedMember",
    "ServableModel",
    "fit_servable",
]
