"""The application: what is loaded at startup, and what a failure looks like.

**A factory, not a module-level app.** ``create_app`` takes its settings, so a
test builds one against a temporary directory and the process that serves it
builds one against the real configuration. A module-level ``app = FastAPI()``
would read the environment at import time, which makes every test that wants
different configuration a test that has to reload a module.

**Loading happens once, in the lifespan.** The artefact is unpickled and the
feature table is indexed on the way up and released on the way down. Neither is
retried per request: the model is a file and the table is Parquet, and a
service that reloaded them on demand would answer the first request a second
slower for the rest of time.

**Missing inputs degrade; they do not crash.** A clean checkout has no artefact
and no tables. The process starts anyway, ``/health`` reports which component
is missing and why, and ``/predict`` answers 503 with the command that fixes
it. Refusing to start would make the first thing a new reader tries fail with a
stack trace rather than a sentence.

**Every error has one owner.** The four exceptions the service raises are
mapped to status codes here, once, so no handler carries a status code and no
two handlers can disagree about what a missing model means.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from api.routes import router
from api.service import (
    BatchTooLargeError,
    FixtureNotFoundError,
    PredictionService,
    ServiceUnavailableError,
)
from src import __version__
from src.pipelines.serving import ServingError, load_index, load_servable
from src.pipelines.tables import resolve_tables
from src.storage.base import StorageError
from src.storage.predictions import open_prediction_log
from src.utils.config import Settings, load_settings
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

TITLE = "Match Outcome Predictor"

DESCRIPTION = """
Calibrated home / draw / away probabilities for 39 professional football
competitions.

**Read `/model-card/limitations` before using a number from here.** The
bookmaker's closing line beats this model in every competition it was scored
on, and every response says whether the fixture was inside the served model's
own training window — an in-sample probability is not the out-of-sample one
`docs/MODEL_CARD.md` reports.

Fixtures come from the batch feature build. The data provider publishes results
rather than a fixture list, so the matches this service can price are exactly
the ones in the feature table; `/fixtures` is how you find them.
"""

REQUEST_ID_HEADER = "x-request-id"


def build_service(settings: Settings) -> PredictionService:
    """Load the artefact, index the tables, open the log.

    Each of the three is caught separately. A process with a model and no
    tables and a process with tables and no model are different problems with
    different fixes, and a single ``try`` around all three would report
    whichever failed first as the whole story.
    """
    service = PredictionService(max_batch=settings.api.max_batch)

    try:
        service.model = load_servable(settings.paths.model_dir)
    except ServingError as error:
        service.model_error = str(error)
        logger.warning("no servable model: %s", error)

    try:
        service.index = load_index(resolve_tables(settings.paths))
        if service.index is None:
            service.index_error = "the match, ratings or feature table is missing"
    except StorageError as error:  # pragma: no cover - a corrupt Parquet file
        service.index_error = str(error)
    if service.index_error is not None:
        logger.warning("no fixture index: %s", service.index_error)

    # Caught for the same reason the other two are, and it is the one that
    # matters most: the log is optional by design, but `open_prediction_log`
    # creates the schema, so an unreachable database raised out of the lifespan
    # and the process that was meant to degrade crash-looped instead. A
    # deployment where the API and PostgreSQL restart together — a node reboot,
    # a failover — is exactly when a working forecaster is worth having.
    try:
        service.log = open_prediction_log(settings.api.prediction_log_dsn)
    except StorageError as error:
        service.log_error = str(error)
        logger.error("prediction log unavailable, serving without it: %s", error)

    logger.info("service ready=%s, log=%s", service.ready, service.log_kind)
    return service


def create_app(settings: Settings | None = None) -> FastAPI:
    """The application, wired to ``settings`` or to the real configuration."""
    resolved = settings if settings is not None else load_settings()
    configure_logging(level=resolved.logging.level, fmt=resolved.logging.format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved
        app.state.service = build_service(resolved)
        try:
            yield
        finally:
            # `suppress`, because a shutdown that raises leaves the process
            # exiting on a traceback about a database connection instead of on
            # whatever actually stopped it.
            with suppress(Exception):
                app.state.service.close()

    app = FastAPI(
        title=TITLE,
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        license_info={"name": "MIT", "identifier": "MIT"},
        contact={
            "name": "Match Outcome Predictor",
            "url": "https://github.com/Vanshcloud/Match-Outcome-Predictor",
        },
        openapi_tags=[
            {"name": "prediction", "description": "Turn a fixture into three probabilities."},
            {"name": "fixtures", "description": "Find the matches that can be priced."},
            {"name": "operations", "description": "Health, provenance and limitations."},
        ],
    )
    _install_middleware(app)
    _install_handlers(app)
    app.include_router(router)
    return app


# ---- middleware --------------------------------------------------------------


def _install_middleware(app: FastAPI) -> None:
    """One access log line per request, with how long it took.

    Structured as fields rather than prose so it greps: a service whose log
    lines are sentences is one where "which endpoint is slow" costs an
    afternoon. The request id is echoed back, so a client can quote the line it
    is asking about.
    """

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        request_id = request.headers.get(REQUEST_ID_HEADER)
        logger.info(
            "method=%s path=%s status=%d duration_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id or "-",
        )
        if request_id:
            response.headers[REQUEST_ID_HEADER] = request_id
        return response


# ---- errors ------------------------------------------------------------------


def _install_handlers(app: FastAPI) -> None:
    """Map the service's four exceptions onto status codes, once.

    ``ServingError`` is a 500 on purpose: it means the artefact and the feature
    table disagree about what a match looks like, which is a deployment fault
    rather than anything the caller did, and reporting it as a client error
    would send someone to fix the wrong thing.
    """

    def _json(status: int, request: Request, error: Exception) -> JSONResponse:
        """One body shape, and one log line per failure, whatever raised it.

        Logged at the level the status deserves: a 404 is a caller naming a
        match that is not there and is not news, a 503 is worth noticing, and a
        500 is worth waking up for.
        """
        level = logger.info if status < 500 else logger.error
        level("%d on %s: %s", status, request.url.path, error)
        return JSONResponse(status_code=status, content={"detail": str(error)})

    @app.exception_handler(ServiceUnavailableError)
    async def _unavailable(request: Request, error: ServiceUnavailableError) -> JSONResponse:
        return _json(503, request, error)

    @app.exception_handler(FixtureNotFoundError)
    async def _not_found(request: Request, error: FixtureNotFoundError) -> JSONResponse:
        return _json(404, request, error)

    @app.exception_handler(BatchTooLargeError)
    async def _too_large(request: Request, error: BatchTooLargeError) -> JSONResponse:
        return _json(422, request, error)

    @app.exception_handler(ServingError)
    async def _serving_failed(request: Request, error: ServingError) -> JSONResponse:
        return _json(500, request, error)


app = create_app()
"""The application uvicorn serves.

Built at import so ``uvicorn api.main:app`` works with no factory flag, which
is what the Dockerfile, the compose file and the README all invoke.
"""
