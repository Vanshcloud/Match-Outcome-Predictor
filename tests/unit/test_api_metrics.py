"""The counters, the exposition, and the cache underneath them.

Nothing here fits a model. The exposition is a rendering of two dataclasses and
the cache is a dictionary with a bound, so both are covered against objects
built by hand — which is what lets this module assert on eviction order and on
a batch larger than the cache, neither of which a request can be relied upon to
produce. The same properties are checked against a real artefact in
``test_api_endpoints.py``, where there is a model to be skipped.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import NO_STORE, create_app
from api.metrics import CONTENT_TYPE, UNMATCHED, Metrics, render
from api.service import PredictionCache, PredictionService
from src.pipelines.serving import Prediction
from tests.unit.test_api_app import settings_for


def prediction(match_id: str, home: float = 0.5) -> Prediction:
    """One prediction, enough of it to be cached and read back."""
    return Prediction(
        fixture={
            "match_id": match_id,
            "competition_id": "ENG_1",
            "home_team": "Arsenal",
            "away_team": "Everton",
            "date": "2024-01-01",
        },
        probabilities={"home": home, "draw": 0.3, "away": 0.2},
        model="ensemble-calibrated",
        model_version="0.1.0",
        in_sample=False,
        predicted_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A degraded service — no model, no table — with the lifespan run."""
    with TestClient(create_app(settings_for(tmp_path))) as entered:
        yield entered


def families(exposition: str) -> dict[str, str]:
    """Each sample line of ``exposition``, keyed by everything before the value."""
    return {
        line.rsplit(" ", 1)[0]: line.rsplit(" ", 1)[1]
        for line in exposition.splitlines()
        if not line.startswith("#")
    }


# ---- the cache ---------------------------------------------------------------


def test_a_miss_is_not_counted_as_a_hit() -> None:
    cache = PredictionCache()
    assert cache.get("nothing") is None
    assert cache.hits == 0


def test_a_stored_prediction_comes_back_and_counts() -> None:
    cache = PredictionCache()
    cache.put("m1", prediction("m1"))
    assert cache.get("m1") is not None
    assert cache.hits == 1
    assert len(cache) == 1


def test_the_least_recently_used_is_the_one_evicted() -> None:
    """Least *recently used*, not least recently written — which is the whole
    reason this is an ``OrderedDict`` and not a queue."""
    cache = PredictionCache(maxsize=2)
    cache.put("m1", prediction("m1"))
    cache.put("m2", prediction("m2"))
    cache.get("m1")  # m1 is now the recent one
    cache.put("m3", prediction("m3"))

    assert cache.get("m2") is None
    assert cache.get("m1") is not None
    assert cache.get("m3") is not None


def test_writing_the_same_fixture_twice_is_one_entry() -> None:
    cache = PredictionCache(maxsize=2)
    cache.put("m1", prediction("m1", home=0.5))
    cache.put("m1", prediction("m1", home=0.6))
    assert len(cache) == 1
    stored = cache.get("m1")
    assert stored is not None
    assert stored.probabilities["home"] == 0.6


def test_size_zero_switches_the_cache_off_rather_than_thrashing() -> None:
    """A deployment measuring model latency wants every request to reach the
    estimators, and this is where that is honoured."""
    cache = PredictionCache(maxsize=0)
    cache.put("m1", prediction("m1"))
    assert len(cache) == 0
    assert cache.get("m1") is None


# ---- the counters ------------------------------------------------------------


def test_observing_a_request_counts_it_once_per_family() -> None:
    metrics = Metrics()
    metrics.observe(method="GET", route="/health", status=200, seconds=0.25)
    metrics.observe(method="GET", route="/health", status=200, seconds=0.75)
    metrics.observe(method="GET", route="/health", status=503, seconds=0.5)

    assert metrics.requests["GET", "/health", 200] == 2
    assert metrics.requests["GET", "/health", 503] == 1
    assert metrics.handled["GET", "/health"] == 3
    assert metrics.duration_seconds["GET", "/health"] == pytest.approx(1.5)


def test_the_count_beside_a_duration_is_every_status_of_that_route() -> None:
    """The sum and the count have to come from one pass, or the mean they
    describe can be wrong by one request."""
    metrics = Metrics()
    metrics.observe(method="POST", route="/predict", status=200, seconds=0.1)
    metrics.observe(method="POST", route="/predict", status=404, seconds=0.1)

    samples = families(render(metrics, PredictionService()))
    assert samples['http_request_duration_seconds_count{method="POST",route="/predict"}'] == "2"


# ---- the exposition ----------------------------------------------------------


def test_the_gauges_read_the_service_rather_than_a_record_of_it() -> None:
    """``service_ready`` and ``/health``'s ``status`` are two renderings of one
    object, so they cannot drift."""
    samples = families(render(Metrics(), PredictionService()))
    assert samples["service_ready"] == "0"
    assert samples['service_component_ready{component="model"}'] == "0"
    assert samples["fixtures_indexed"] == "0"


def test_the_counters_a_process_has_not_used_are_published_as_zero() -> None:
    """A counter that only appears once it is non-zero is a counter no alert
    can be written against before the first incident."""
    samples = families(render(Metrics(), PredictionService()))
    assert samples["predictions_total"] == "0"
    assert samples["predictions_logged_total"] == "0"
    assert samples["prediction_cache_hits_total"] == "0"
    assert samples["prediction_cache_entries"] == "0"


def test_service_counters_are_read_off_the_service() -> None:
    service = PredictionService(served=7, logged=5)
    service.cache.put("m1", prediction("m1"))
    service.cache.get("m1")

    samples = families(render(Metrics(), service))
    assert samples["predictions_total"] == "7"
    assert samples["predictions_logged_total"] == "5"
    assert samples["prediction_cache_hits_total"] == "1"
    assert samples["prediction_cache_entries"] == "1"


def test_every_family_declares_its_help_and_type_before_its_samples() -> None:
    """The format requires it, and a scraper that meets a sample first drops
    the family rather than complaining about it."""
    metrics = Metrics()
    metrics.observe(method="GET", route="/health", status=200, seconds=0.1)
    lines = render(metrics, PredictionService()).splitlines()

    declared: set[str] = set()
    for line in lines:
        if line.startswith("# TYPE "):
            declared.add(line.split()[2])
        elif not line.startswith("#"):
            name = line.split("{")[0].split(" ")[0]
            assert name.removesuffix("_sum").removesuffix("_count") in declared


def test_a_label_that_could_break_the_format_is_escaped() -> None:
    """Route templates cannot contain these, which is exactly why the escaping
    has to be tested here rather than left to a request to produce."""
    metrics = Metrics()
    metrics.observe(method="GET", route='/a"b\\c\nd', status=200, seconds=0.1)
    rendered = render(metrics, PredictionService())
    assert '/a\\"b\\\\c\\nd' in rendered
    assert rendered.count("\n") == len(rendered.splitlines())


def test_a_brace_in_a_route_template_is_left_alone() -> None:
    """A brace is legal inside a quoted label value. Escaping it would corrupt
    the label rather than protect it — and every parameterised route has two."""
    metrics = Metrics()
    metrics.observe(method="GET", route="/model-card/{section}", status=200, seconds=0.1)
    assert 'route="/model-card/{section}"' in render(metrics, PredictionService())


# ---- the endpoint ------------------------------------------------------------


def test_metrics_are_served_in_the_format_a_scraper_negotiates_on(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE
    assert "# TYPE http_requests_total counter" in response.text


def test_a_request_is_counted_under_the_route_it_matched(client: TestClient) -> None:
    client.get("/health")
    client.get("/health")
    samples = families(client.get("/metrics").text)
    assert samples['http_requests_total{method="GET",route="/health",status="200"}'] == "2"


def test_a_query_string_does_not_become_its_own_series(client: TestClient) -> None:
    """One series for ``/fixtures``, whatever was asked of it. A raw path would
    make every distinct query string a time series of its own."""
    client.get("/fixtures?team=Arsenal")
    client.get("/fixtures?team=Everton")
    samples = families(client.get("/metrics").text)
    assert samples['http_requests_total{method="GET",route="/fixtures",status="503"}'] == "2"
    assert not any("team=" in key for key in samples)


def test_a_request_that_matched_no_route_is_one_series(client: TestClient) -> None:
    """Otherwise a scanner walking a wordlist writes the wordlist into this
    process's memory."""
    client.get("/wp-login.php")
    client.get("/.env")
    samples = families(client.get("/metrics").text)
    assert samples[f'http_requests_total{{method="GET",route="{UNMATCHED}",status="404"}}'] == "2"


def test_the_metrics_endpoint_is_in_the_openapi_document(client: TestClient) -> None:
    """It is part of the service's interface. An endpoint a scraper is
    configured against and the document does not mention is one a reader
    concludes was removed."""
    assert "/metrics" in client.get("/openapi.json").json()["paths"]


# ---- cache directives --------------------------------------------------------


@pytest.mark.parametrize("path", ["/health", "/metrics"])
def test_health_and_metrics_are_never_stored(client: TestClient, path: str) -> None:
    """A cached health check is a proxy answering for a process it has not
    spoken to, and a cached scrape is a counter that appears to stop. Both are
    failures that look like health."""
    assert client.get(path).headers["cache-control"] == NO_STORE


def test_a_prediction_is_never_stored(client: TestClient) -> None:
    response = client.post("/predict", json={"match_id": "anything"})
    assert response.headers["cache-control"] == NO_STORE


def test_a_failure_is_never_cached_even_on_a_cacheable_route(client: TestClient) -> None:
    """``/fixtures`` answers 503 until the tables exist, and that is the one
    thing about it which is not stable for the life of the process. A proxy
    holding it for a minute would report the service down after it came up."""
    response = client.get("/fixtures")
    assert response.status_code == 503
    assert response.headers["cache-control"] == NO_STORE
