"""The dashboard's reader for the service. No test reaches the network.

Every one mounts a stub adapter on an injected session, exactly as
``tests/unit/test_http.py`` does — the dashboard talks to the API over HTTP,
and a unit suite that needed the API running would be an integration suite
wearing the wrong marker.

The behaviour worth pinning is the failure shape. A panel renders "the service
is not answering, and here is why"; it has nothing different to do for a
connection reset than for a 503, so every failure is one exception carrying one
sentence.
"""

from __future__ import annotations

import io
import json

import pytest
import requests
from requests.adapters import BaseAdapter
from requests.structures import CaseInsensitiveDict

from dashboard.client import MAX_RETRIES, PredictionClient, ServiceError
from src.utils.http import HttpClient

BASE = "http://service.test"


class _StubAdapter(BaseAdapter):
    """Answers every request with one canned response."""

    def __init__(self, *, status: int = 200, body: object = None, raw: bytes | None = None) -> None:
        super().__init__()
        self.status = status
        self.payload = raw if raw is not None else json.dumps(body if body is not None else {})
        self.requests: list[requests.PreparedRequest] = []

    def send(self, request, **_kwargs):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        response = requests.Response()
        response.status_code = self.status
        response.reason = "OK" if self.status < 400 else "Bad"
        response.url = str(request.url)
        response.request = request
        response.headers = CaseInsensitiveDict({"Content-Type": "application/json"})
        body = self.payload if isinstance(self.payload, bytes) else str(self.payload).encode()
        response.raw = io.BytesIO(body)
        return response

    def close(self) -> None:
        return None


class _Refusing(BaseAdapter):
    """Fails the way a service that is not running fails."""

    def send(self, _request, **_kwargs):  # type: ignore[no-untyped-def]
        raise requests.ConnectionError("connection refused")

    def close(self) -> None:
        return None


def client_with(adapter: BaseAdapter) -> PredictionClient:
    """A client whose every request goes to ``adapter``.

    Through the injected session, which is the seam
    :class:`~src.utils.http.HttpClient` and the prediction log both use — and
    the reason no test in this file opens a socket.
    """
    session = requests.Session()
    # AFTER construction, and the order is load-bearing for the reason
    # tests/unit/test_http.py records: HttpClient mounts its own retry adapter
    # on both schemes, so a stub mounted first is silently replaced.
    # `max_retries=0` because 503 is a retry status and this suite asserts on
    # the error, not on the backoff schedule.
    http = HttpClient(session=session, max_retries=0, timeout_seconds=1.0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return PredictionClient(base_url=BASE, timeout_seconds=1.0, http=http)


# ---- the happy paths ---------------------------------------------------------


def test_health_is_returned_as_the_document_the_service_sent() -> None:
    adapter = _StubAdapter(body={"status": "ok", "components": []})
    assert client_with(adapter).health()["status"] == "ok"
    assert adapter.requests[0].url == f"{BASE}/health"


def test_version_is_returned_as_sent() -> None:
    adapter = _StubAdapter(body={"version": "0.12.0", "model": "ensemble-calibrated"})
    assert client_with(adapter).version()["model"] == "ensemble-calibrated"


def test_fixtures_unwraps_the_list_the_endpoint_nests() -> None:
    adapter = _StubAdapter(body={"fixtures": [{"match_id": "a"}, {"match_id": "b"}], "count": 2})
    found = client_with(adapter).fixtures(competition_id="ENG_1")
    assert [row["match_id"] for row in found] == ["a", "b"]


def test_a_body_without_a_fixture_list_reads_as_no_fixtures() -> None:
    """Defensive, and cheap: a panel iterating `None` is a traceback on a page."""
    adapter = _StubAdapter(body={"fixtures": "not a list"})
    assert client_with(adapter).fixtures() == []


def test_blank_filters_are_not_sent_at_all() -> None:
    """An empty text box means "no filter", and the service reads an empty
    string as a filter that matches nothing."""
    adapter = _StubAdapter(body={"fixtures": []})
    client_with(adapter).fixtures(competition_id="", team=None, limit=5)
    url = str(adapter.requests[0].url)
    assert "competition_id" not in url and "team" not in url
    assert "limit=5" in url


def test_predict_posts_the_fixture_and_returns_the_answer() -> None:
    adapter = _StubAdapter(body={"probabilities": {"home": 0.4, "draw": 0.3, "away": 0.3}})
    answer = client_with(adapter).predict({"match_id": "abc"})
    assert answer["probabilities"]["home"] == 0.4
    request = adapter.requests[0]
    assert request.method == "POST"
    assert json.loads(request.body or b"") == {"match_id": "abc"}


# ---- the failures, which are the point ---------------------------------------


def test_a_service_that_is_not_running_is_one_error_with_a_sentence() -> None:
    with pytest.raises(
        ServiceError, match=r"^the service could not be reached \(ConnectionError\)$"
    ):
        client_with(_Refusing()).health()


def test_a_refused_prediction_is_the_same_error() -> None:
    with pytest.raises(ServiceError, match="could not be reached"):
        client_with(_Refusing()).predict({"match_id": "abc"})


def test_an_error_status_carries_the_services_own_detail() -> None:
    """The 404 body says which fixture was not found, and that is the useful
    half — a panel that printed "404" would send the reader back to the API."""
    adapter = _StubAdapter(status=404, body={"detail": "no fixture matches this request"})
    with pytest.raises(ServiceError, match="404: no fixture matches this request"):
        client_with(adapter).predict({"match_id": "nope"})


def test_an_error_status_with_an_unreadable_body_falls_back_to_the_reason() -> None:
    adapter = _StubAdapter(status=503, raw=b"<html>gateway</html>")
    with pytest.raises(ServiceError, match="503: Bad"):
        client_with(adapter).health()


def test_a_two_hundred_that_is_not_json_is_reported_as_such() -> None:
    """A proxy returning an HTML holding page with a 200 is the real case."""
    adapter = _StubAdapter(status=200, raw=b"<html>hello</html>")
    with pytest.raises(ServiceError, match="not JSON"):
        client_with(adapter).health()


def test_the_base_url_keeps_one_slash_however_it_was_given() -> None:
    assert PredictionClient(base_url=f"{BASE}/")._url("/health") == f"{BASE}/health"
    assert PredictionClient(base_url=BASE)._url("/health") == f"{BASE}/health"


def test_without_an_injected_client_one_is_built_and_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The path the app takes. Streamlit reruns the script on every
    interaction, so a client built here is closed here — a pool leaked per
    rerun is a pool nobody is tracking.
    """
    built: list[dict[str, object]] = []
    closed: list[bool] = []

    class Recording(HttpClient):
        def __init__(self, **kwargs: object) -> None:
            built.append(kwargs)
            super().__init__(**kwargs)  # type: ignore[arg-type]

        def close(self) -> None:
            closed.append(True)
            super().close()

    monkeypatch.setattr("dashboard.client.HttpClient", Recording)
    client = PredictionClient(base_url=BASE, timeout_seconds=3.0)
    with client._client() as made:
        assert isinstance(made, HttpClient)

    # Few retries: HttpClient's download defaults made each call to a service that
    # is not running take fifteen seconds, inside a page that makes several.
    assert built == [{"timeout_seconds": 3.0, "max_retries": MAX_RETRIES}]
    assert MAX_RETRIES <= 1
    assert closed == [True]


def test_an_injected_client_is_used_and_not_closed() -> None:
    """It belongs to whoever passed it. Closing another object's connection
    pool after one call is how the second call gets a confusing error."""
    http = HttpClient(max_retries=0)
    client = PredictionClient(base_url=BASE, http=http)
    with client._client() as made:
        assert made is http
    assert http.session.adapters  # still usable
