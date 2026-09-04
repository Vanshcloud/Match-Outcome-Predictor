"""What answers a request: one loaded model, one indexed table, one log.

Built once at startup and held on the application, because all three are
expensive to obtain and none of them changes while the process runs. A handler
that loaded a model per request would spend a second unpickling it and would
still be answering from the same bytes.

**No arithmetic here.** Every probability comes out of
:meth:`src.models.artifact.ServableModel.predict` and every lookup out of
:class:`src.pipelines.serving.FixtureIndex`. This class decides which rows to
ask about and what to do when there are none — routing, not modelling.

**Degraded is a state, not a crash.** A process with no artefact and no feature
table is exactly what a clean checkout produces, and it starts: ``/health``
says which component is missing and ``/predict`` returns 503 naming it. The
alternative — refusing to start — makes the first thing a new reader does fail
with a stack trace instead of a sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from api.schemas import (
    ComponentHealth,
    HealthResponse,
    LimitationsResponse,
    VersionResponse,
)
from src import __version__
from src.evaluation.model_card import LIMITATIONS
from src.models.dataset import DESIGN_COLUMNS
from src.pipelines.serving import (
    FixtureIndex,
    LoadedModel,
    Prediction,
    ServingError,
    predict_fixtures,
)
from src.storage.base import PredictionLog, StorageError
from src.storage.predictions import NullPredictionLog
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_COMPONENT = "model"
FIXTURES_COMPONENT = "fixtures"
LOG_COMPONENT = "prediction_log"


class ServiceUnavailableError(RuntimeError):
    """The service cannot answer because something it needs is not loaded."""


class FixtureNotFoundError(LookupError):
    """No row in the feature table matches the fixture that was asked for."""


class BatchTooLargeError(ValueError):
    """More fixtures were asked for in one request than the configured limit.

    Its own class rather than a bare ``ValueError`` so the handler that turns
    it into a 422 cannot also catch an unrelated bug and report it as a client
    error — which is how a real fault ends up looking like a bad request in
    every dashboard that counts them.
    """


@dataclass(frozen=True, slots=True)
class Component:
    """One dependency and whether it came up."""

    name: str
    ready: bool
    detail: str | None = None


@dataclass
class PredictionService:
    """Everything one process needs to answer a prediction request."""

    model: LoadedModel | None = None
    index: FixtureIndex | None = None
    log: PredictionLog = field(default_factory=NullPredictionLog)
    max_batch: int = 50

    model_error: str | None = None
    index_error: str | None = None

    # ---- readiness ---------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True when the service can price a fixture.

        The log is deliberately not part of this. It is optional by design, and
        a readiness probe that went red because an audit trail was switched off
        would take a working service out of a load balancer for a reason that
        has nothing to do with whether it can answer.
        """
        return self.model is not None and self.index is not None

    def components(self) -> list[Component]:
        """Each dependency, in the order a reader would check them."""
        return [
            Component(
                name=MODEL_COMPONENT,
                ready=self.model is not None,
                detail=self.model_error if self.model is None else self.model.model.name,
            ),
            Component(
                name=FIXTURES_COMPONENT,
                ready=self.index is not None,
                detail=(
                    self.index_error
                    if self.index is None
                    else f"{len(self.index):,} fixtures indexed"
                ),
            ),
            Component(
                name=LOG_COMPONENT,
                ready=True,
                detail=self.log_kind,
            ),
        ]

    @property
    def log_kind(self) -> str:
        """``"postgres"`` or ``"disabled"``, for ``/health`` and ``/version``."""
        return "postgres" if self.log.enabled else "disabled"

    # ---- answering ---------------------------------------------------------

    def _require(self) -> tuple[LoadedModel, FixtureIndex]:
        """The model and the index, or the reason there is not one.

        Raises:
            ServiceUnavailableError: Naming the components that are missing and the
                command that produces them. A 503 whose body says "not ready"
                and nothing else costs the reader the next twenty minutes.
        """
        if self.model is None or self.index is None:
            missing = [component.name for component in self.components() if not component.ready]
            reasons = [
                detail for detail in (self.model_error, self.index_error) if detail is not None
            ]
            raise ServiceUnavailableError(
                f"not ready: {', '.join(missing)} unavailable"
                + (f" ({'; '.join(reasons)})" if reasons else "")
                + ". Run `make reproduce` to build the tables, then `make model`."
            )
        return self.model, self.index

    def resolve(self, lookups: list[dict[str, str | None]]) -> tuple[pd.DataFrame, list[int]]:
        """Rows for the fixtures that exist, and the positions of those that do not.

        One frame rather than a list of frames: the model prices a matrix, and
        concatenating once here is what keeps a batch of fifty from costing
        fifty passes through three estimators.
        """
        _, index = self._require()
        found: list[pd.DataFrame] = []
        missing: list[int] = []
        for position, lookup in enumerate(lookups):
            row = index.resolve(**lookup)
            if row is None:
                missing.append(position)
            else:
                found.append(row)
        frame = pd.concat(found, ignore_index=False) if found else pd.DataFrame()
        return frame, missing

    def predict(self, lookups: list[dict[str, str | None]]) -> tuple[list[Prediction], list[int]]:
        """Price every fixture that resolves; report the positions that did not.

        Raises:
            ServiceUnavailableError: If the model or the table is not loaded.
            FixtureNotFoundError: If *nothing* resolved. A single-fixture request
                takes this path to a 404; a batch handles the partial case
                itself, because forty-nine forecasts and one named gap is a better
                answer than an error.
            ServingError: If the rows cannot be priced — a design column
                missing from the table, which means the feature build and the
                artefact disagree about what a match looks like.
        """
        loaded, _ = self._require()
        rows, missing = self.resolve(lookups)
        if rows.empty:
            raise FixtureNotFoundError("no fixture in the table matches this request")
        return predict_fixtures(loaded.model, rows), missing

    def record(self, predictions: list[Prediction]) -> int:
        """Write predictions to the log, and never fail a request because of it.

        A prediction that was answered and not logged is a gap in an audit
        trail. A prediction that was *not answered* because the audit trail was
        down is an outage. The first is the lesser harm, so a log failure is
        logged at error level and the response goes out with ``recorded: 0``,
        which is how a caller can tell.
        """
        if not predictions:
            return 0
        try:
            return self.log.record([prediction.as_record() for prediction in predictions])
        except StorageError as error:
            logger.error("prediction log rejected %d row(s): %s", len(predictions), error)
            return 0

    def fixtures(
        self,
        *,
        competition_id: str | None = None,
        team: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        """Fixtures matching the filters, so a caller can name one."""
        _, index = self._require()
        return index.search(
            competition_id=competition_id, team=team, since=since, until=until, limit=limit
        )

    # ---- provenance --------------------------------------------------------

    def health(self) -> HealthResponse:
        """Liveness and readiness as one document.

        ``status`` is ``"ok"`` only when every component the service needs to
        answer is up, so a load balancer reads one field and a person reads the
        rest.
        """
        return HealthResponse(
            status="ok" if self.ready else "degraded",
            version=__version__,
            components=[
                ComponentHealth(name=part.name, ready=part.ready, detail=part.detail)
                for part in self.components()
            ],
        )

    def version(self) -> VersionResponse:
        """What is running, for ``/version``.

        Assembled here rather than in the route, so the route is a call and a
        return and this stays the one place that knows what a loaded model is.
        """
        if self.model is None:
            return VersionResponse(
                version=__version__,
                design_columns=len(DESIGN_COLUMNS),
                fixtures_indexed=len(self.index) if self.index is not None else 0,
                prediction_log=self.log_kind,
            )
        model = self.model.model
        return VersionResponse(
            version=__version__,
            model=model.name,
            members=list(model.member_names),
            design_columns=len(model.columns),
            temperature=model.temperature,
            trained_matches=model.trained_matches,
            trained_from=model.trained_from.date(),
            trained_through=model.trained_through.date(),
            fixtures_indexed=len(self.index) if self.index is not None else 0,
            prediction_log=self.log_kind,
            library_mismatches=list(self.model.library_mismatches),
        )

    def limitations(self) -> LimitationsResponse:
        """The model card's limitations, as sentences a client can render.

        The leading ``- `` and the card's line wrapping are undone, so a
        consumer gets four statements rather than fourteen lines of Markdown
        wrapped at the width a document happens to be written at.
        """
        bullets: list[str] = []
        for line in LIMITATIONS:
            if line.startswith("- "):
                bullets.append(line[2:].strip())
            else:
                bullets[-1] = f"{bullets[-1]} {line.strip()}"
        return LimitationsResponse(
            model=self.model.model.name if self.model is not None else None,
            limitations=bullets,
        )

    def close(self) -> None:
        """Release what was opened at startup. Called from the lifespan's exit."""
        self.log.close()


__all__ = [
    "BatchTooLargeError",
    "Component",
    "FixtureNotFoundError",
    "PredictionService",
    "ServiceUnavailableError",
    "ServingError",
]
