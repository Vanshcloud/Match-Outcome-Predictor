"""The endpoints.

Thin on purpose. Each handler resolves the service, calls one method on it and
returns what came back; the decisions — what is ready, which rows to price,
what to do about a fixture that is not there — live in :mod:`api.service`,
where they can be tested without a client. A route that grew a branch would be
a decision whose only tests are HTTP round trips.

Failures are raised, not returned. :mod:`api.main` maps the four service
exceptions onto 503, 404, 422 and 500 in one place, so every handler reports a
missing model the same way and none of them has to remember a status code.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import PlainTextResponse

from api.metrics import CONTENT_TYPE, Metrics, render
from api.schemas import (
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    FixtureListResponse,
    FixtureRequest,
    FixtureSummary,
    HealthResponse,
    LimitationsResponse,
    PredictionResponse,
    VersionResponse,
)
from api.service import BatchTooLargeError, PredictionService
from src.pipelines.serving import Prediction
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

MAX_FIXTURE_RESULTS = 200
"""The ceiling on one ``/fixtures`` page.

A bound rather than a preference, for the same reason the batch has one: the
index holds three hundred thousand rows and an unbounded ``limit`` is a request
that serialises all of them.
"""


def get_service(request: Request) -> PredictionService:
    """The service built by the lifespan, off the application state.

    A dependency rather than a module global, so a test can build an app around
    a service of its own and two apps in one process cannot end up sharing a
    model.
    """
    service: PredictionService = request.app.state.service
    return service


Service = Annotated[PredictionService, Depends(get_service)]


def get_metrics(request: Request) -> Metrics:
    """The counters the middleware writes to, off the application state.

    A dependency for the same reason :func:`get_service` is one: two
    applications in one process — which is what the test suite builds — must
    not share a counter, or a test's assertion about how many requests were
    served depends on which tests ran before it.
    """
    metrics: Metrics = request.app.state.metrics
    return metrics


Counters = Annotated[Metrics, Depends(get_metrics)]


def _as_response(prediction: Prediction) -> PredictionResponse:
    """One :class:`~src.pipelines.serving.Prediction` as its response body."""
    return PredictionResponse.model_validate(
        {
            "fixture": prediction.fixture,
            "probabilities": prediction.probabilities,
            "model": prediction.model,
            "model_version": prediction.model_version,
            "in_sample": prediction.in_sample,
            "predicted_at": prediction.predicted_at,
        }
    )


# ---- operations --------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["operations"],
    summary="Liveness and readiness",
)
def health(service: Service) -> HealthResponse:
    """Always 200 while the process is alive; ``status`` says whether it can answer.

    Two questions in one document. A 503 here would make "the container is
    wedged" and "the model has not been built yet" indistinguishable to
    whatever is watching, and a person needs to do different things about them.
    """
    return service.health()


@router.get(
    "/version",
    response_model=VersionResponse,
    tags=["operations"],
    summary="What is running, and what it was fitted on",
)
def version(service: Service) -> VersionResponse:
    """Enough provenance to reproduce a number this service returned."""
    return service.version()


@router.get(
    "/model-card/limitations",
    response_model=LimitationsResponse,
    tags=["operations"],
    summary="What this model must not be used for",
)
def limitations(service: Service) -> LimitationsResponse:
    """The model card's own section, served rather than linked.

    Milestone 11's scope asked for the limitations to be reachable from a
    response instead of buried in a repository, and every prediction carries
    the path to this endpoint. The text is the constant the card renders from,
    so the two cannot drift.
    """
    return service.limitations()


@router.get(
    "/metrics",
    tags=["operations"],
    summary="Prometheus exposition",
    response_class=PlainTextResponse,
    responses={200: {"content": {CONTENT_TYPE: {"schema": {"type": "string"}}}}},
)
def metrics(service: Service, counters: Counters) -> PlainTextResponse:
    """What this process has served, in the format a scraper already understands.

    Not a JSON document and deliberately not a pydantic model: the exposition
    format is the interface, and describing it as a schema would invite a
    second consumer that parses the JSON — at which point the thing everything
    else scrapes has two shapes.

    The gauges are read off the service at scrape time rather than tracked, so
    ``service_ready`` here and ``status`` on ``/health`` cannot disagree: they
    are two renderings of one object rather than two records of it.
    """
    return PlainTextResponse(render(counters, service), media_type=CONTENT_TYPE)


# ---- fixtures ----------------------------------------------------------------


@router.get(
    "/fixtures",
    response_model=FixtureListResponse,
    tags=["fixtures"],
    summary="Find matches that can be priced",
    responses={503: {"description": "No feature table is loaded.", "model": ErrorResponse}},
)
def fixtures(
    service: Service,
    competition_id: Annotated[str | None, Query(max_length=50)] = None,
    team: Annotated[str | None, Query(max_length=120)] = None,
    since: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    until: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_FIXTURE_RESULTS)] = 20,
) -> FixtureListResponse:
    """Fixtures in the feature table, most recent first.

    Discovery exists because without it nothing else here is usable: the
    provider publishes results rather than a fixture list, so the matches this
    service can price are exactly the ones the batch build wrote, and a caller
    has no other way to learn which those are.
    """
    found = service.fixtures(
        competition_id=competition_id, team=team, since=since, until=until, limit=limit
    )
    rows = [
        FixtureSummary(
            match_id=str(row["match_id"]),
            competition_id=str(row["competition_id"]),
            date=row["date"].date(),
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
        )
        for _, row in found.iterrows()
    ]
    return FixtureListResponse(fixtures=rows, count=len(rows))


# ---- prediction --------------------------------------------------------------


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Price one fixture",
    responses={
        404: {
            "description": "No fixture in the table matches the request.",
            "model": ErrorResponse,
        },
        503: {"description": "No model or no feature table is loaded.", "model": ErrorResponse},
    },
)
def predict(request: FixtureRequest, service: Service) -> PredictionResponse:
    """Three calibrated probabilities for one match.

    The prediction is written to the log before it is returned, and a log that
    refuses does not fail the request — see
    :meth:`api.service.PredictionService.record`.
    """
    predictions, _ = service.predict([request.as_lookup()])
    service.record(predictions)
    return _as_response(predictions[0])


@router.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["prediction"],
    summary="Price several fixtures in one pass",
    responses={
        404: {"description": "None of the fixtures could be resolved.", "model": ErrorResponse},
        422: {"description": "More fixtures than `API_MAX_BATCH` allows.", "model": ErrorResponse},
        503: {"description": "No model or no feature table is loaded.", "model": ErrorResponse},
    },
)
def predict_batch(request: BatchRequest, service: Service) -> BatchResponse:
    """Every fixture that resolves, priced; every one that does not, named.

    Partial success, because a batch of fifty with one misspelled club should
    come back as forty-nine forecasts rather than an error. The unresolved
    requests are echoed verbatim, so a caller can see which of theirs it was
    without matching on position.
    """
    if len(request.fixtures) > service.max_batch:
        raise BatchTooLargeError(
            f"{len(request.fixtures)} fixtures requested; the limit is {service.max_batch}"
        )
    predictions, missing = service.predict([one.as_lookup() for one in request.fixtures])
    recorded = service.record(predictions)
    return BatchResponse(
        predictions=[_as_response(one) for one in predictions],
        unresolved=[request.fixtures[position] for position in missing],
        recorded=recorded,
    )
