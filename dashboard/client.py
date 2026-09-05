"""Asking the service for a probability, over HTTP.

The dashboard does not load the artefact. If it did, "what does the model say"
would have two implementations — the served one and this one — and the day they
disagreed the disagreement would be invisible, because nothing compares them.
So this module speaks to the running service and has no opinion about models.

Built on :class:`~src.utils.http.HttpClient` rather than on ``requests``
directly, for the reason CI enforces everywhere else: outbound HTTP in this
project goes through one module that knows about timeouts, retries and rate
limits, and a second place that calls ``requests.post`` is a second place to
get those wrong.

**Every failure is one exception type.** A dashboard panel needs to render
"the service is not answering, here is why" — it does not need to distinguish a
connection error from a 503, and a panel that had to catch three exception
trees would catch two of them.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import requests

from src.utils.http import HttpClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

PREDICT_PATH = "/predict"
FIXTURES_PATH = "/fixtures"
HEALTH_PATH = "/health"
VERSION_PATH = "/version"


class ServiceError(RuntimeError):
    """The prediction service could not be reached, or refused the request.

    One type for every failure mode: unreachable, timed out, 4xx, 5xx, or a
    body that is not the JSON document the schema promises. The panel that
    catches this renders the message; it has nothing different to do for a
    connection reset than for a 503.
    """


@dataclass(frozen=True, slots=True)
class PredictionClient:
    """A thin, typed reader for the six endpoints the service exposes.

    Frozen and cheap to construct. The underlying session is created per call
    rather than held, because Streamlit reruns the script on every interaction
    and a long-lived pooled connection across those reruns is a resource whose
    lifetime nobody is tracking.
    """

    base_url: str
    timeout_seconds: float = 10.0
    http: HttpClient | None = None
    """Injected for tests, and an ``HttpClient`` rather than a session.

    ``HttpClient.__init__`` mounts its own retry adapter on both schemes, so a
    stub adapter mounted on a session handed *in* is silently replaced — a trap
    ``tests/unit/test_http.py`` already documents, and one this class would
    spring on every call because it builds its client lazily. Taking the
    finished client moves the seam past the mounting.
    """

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    @contextmanager
    def _client(self) -> Iterator[HttpClient]:
        """The injected client, or a fresh one closed on the way out.

        An injected client is *not* closed here: it belongs to whoever passed
        it, and closing another object's connection pool after one call is how
        a second call gets a confusing error. A client built here is owned here
        and is closed — Streamlit reruns this script on every interaction, and
        a pool leaked per rerun is a pool nobody is tracking.
        """
        if self.http is not None:
            yield self.http
            return
        with HttpClient(timeout_seconds=self.timeout_seconds) as made:
            yield made

    def _json(self, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError as error:
            raise ServiceError(f"the service returned a body that is not JSON: {error}") from error

    def health(self) -> dict[str, Any]:
        """``/health``. The one call a panel makes before it offers anything."""
        return self._get(HEALTH_PATH)

    def version(self) -> dict[str, Any]:
        """``/version``. Model provenance, so the page can say what answered."""
        return self._get(VERSION_PATH)

    def fixtures(self, **filters: object) -> list[dict[str, Any]]:
        """``/fixtures``. The only way to learn which matches can be priced.

        Filters with a value of ``None`` are dropped rather than sent: the
        service treats an absent parameter as "no filter" and an empty string
        as a filter matching nothing, and a blank text box means the former.
        """
        params = {name: value for name, value in filters.items() if value not in (None, "")}
        payload = self._get(FIXTURES_PATH, params=params)
        found = payload.get("fixtures", [])
        return list(found) if isinstance(found, list) else []

    def predict(self, fixture: dict[str, str]) -> dict[str, Any]:
        """``POST /predict``. One fixture, three calibrated probabilities."""
        with self._client() as client:
            try:
                response = client.post(self._url(PREDICT_PATH), json=fixture)
            except requests.RequestException as error:
                raise ServiceError(_describe(error)) from error
        return dict(self._json(response))

    def _get(self, path: str, **kwargs: object) -> dict[str, Any]:
        with self._client() as client:
            try:
                response = client.get(self._url(path), **kwargs)
            except requests.RequestException as error:
                raise ServiceError(_describe(error)) from error
        return dict(self._json(response))


def _describe(error: requests.RequestException) -> str:
    """A sentence a person reading a dashboard can act on.

    The service's own ``detail`` when there is a response to read — a 404 says
    which fixture was not found, and that is the useful half — and the
    transport error otherwise, which is what a service that is simply not
    running produces.
    """
    response = error.response
    if response is not None:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        return f"the service answered {response.status_code}: {detail or response.reason}"
    return f"the service could not be reached: {error}"
