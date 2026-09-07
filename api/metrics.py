"""What this process has done, in the format a scraper already understands.

Prometheus text exposition, rendered on demand from counters this module keeps
and gauges read off the service at scrape time.

**No client library.** ``prometheus-client`` would render this, and it would
also bring a process-global registry, a multiprocess mode, a WSGI app and a
default set of platform collectors — none of which this service has a use for.
The format it implements is three rules: a ``# HELP`` line, a ``# TYPE`` line,
and samples. That is cheaper to write than the dependency is to justify, and
:mod:`api` gains no third-party import on the request path.

**Route templates, never request paths.** ``/fixtures?team=Arsenal`` and
``/fixtures?team=Everton`` are one time series, because the label is the route
the request matched rather than the URL it arrived at. A raw path would make
every distinct query string its own series, which is how a metrics endpoint
becomes the largest thing a service serves and how a scrape starts costing more
than the traffic it describes. A request that matched no route is labelled
:data:`UNMATCHED` rather than by its path, for exactly the same reason — a
scanner walking a wordlist would otherwise write the wordlist into memory.

**Counters reset when the process does, and that is the contract.** A scraper
handles a counter going back to zero; it cannot handle one that silently
rewinds mid-life. Nothing here is persisted and nothing is decremented.

**Gauges are read, not tracked.** ``service_ready`` and the per-component
readiness come from :class:`api.service.PredictionService` at the moment of the
scrape, so they cannot drift from what ``/health`` says — the two answer from
the same object rather than from two records of it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from api.service import PredictionService

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
"""The exposition format's own media type, version and all.

Prometheus negotiates on this string. Serving ``text/plain`` alone works with
most scrapers and is the sort of thing that stops working on an upgrade.
"""

UNMATCHED = "<unmatched>"
"""The route label for a request that matched no route.

Pointed brackets because no real route path can contain them, so this can never
collide with one.
"""

Route = tuple[str, str]
"""A method and a route template — the pair every HTTP series is keyed by."""


@dataclass
class Metrics:
    """Everything one process has served, keyed by route rather than by URL.

    Plain dictionaries and no lock. The event loop runs one request's middleware
    body at a time between awaits, and the two mutations below happen with no
    await between them, so a counter cannot be observed half-updated. A lock
    here would be a lock on every request for a race that the loop already
    rules out.
    """

    requests: Counter[tuple[str, str, int]] = field(default_factory=Counter)
    """Requests by method, route template and status code."""

    handled: Counter[Route] = field(default_factory=Counter)
    """Requests by method and route, whatever the status.

    Kept beside :attr:`requests` rather than derived from it by summing over
    statuses, because it is the denominator of the duration below and the two
    must be observed together — a sum with a count computed from a different
    pass is a mean that can be wrong by one request.
    """

    duration_seconds: dict[Route, float] = field(default_factory=dict)
    """Total time spent in each route, in seconds.

    A sum and a count rather than a histogram. Buckets are what a histogram is
    for and choosing them is a claim about the latency this service has; this
    one answers from a dictionary and a fitted model, so the honest thing to
    publish today is the mean and let a percentile wait until there is a
    distribution worth bucketing.
    """

    def observe(self, *, method: str, route: str, status: int, seconds: float) -> None:
        """Record one finished request."""
        key: Route = (method, route)
        self.requests[method, route, status] += 1
        self.handled[key] += 1
        self.duration_seconds[key] = self.duration_seconds.get(key, 0.0) + seconds


# ---- exposition --------------------------------------------------------------


def _escape(value: str) -> str:
    """A label value, escaped as the exposition format requires.

    Three characters and no more: backslash, double quote and newline. Route
    templates contain braces — ``/items/{id}`` — and a brace is legal inside a
    quoted label value, so escaping it would corrupt the label rather than
    protect it.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _sample(name: str, labels: tuple[tuple[str, str], ...], value: float) -> str:
    """One sample line, with its labels if it has any."""
    rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
    if not labels:
        return f"{name} {rendered}"
    inside = ",".join(f'{key}="{_escape(text)}"' for key, text in labels)
    return f"{name}{{{inside}}} {rendered}"


def render(metrics: Metrics, service: PredictionService) -> str:
    """The whole exposition, as one document.

    Built as a list of lines rather than by concatenation so the ordering is
    visible: every family is contiguous and its ``# HELP`` and ``# TYPE`` come
    first, which is what the format requires and what makes the output readable
    by a person with curl before it is read by a scraper.
    """
    lines: list[str] = [
        "# HELP http_requests_total Requests handled since this process started.",
        "# TYPE http_requests_total counter",
    ]
    for (method, route, status), count in sorted(metrics.requests.items()):
        lines.append(
            _sample(
                "http_requests_total",
                (("method", method), ("route", route), ("status", str(status))),
                count,
            )
        )

    lines += [
        "# HELP http_request_duration_seconds Time spent handling requests.",
        "# TYPE http_request_duration_seconds summary",
    ]
    for key in sorted(metrics.duration_seconds):
        labels = (("method", key[0]), ("route", key[1]))
        lines.append(
            _sample("http_request_duration_seconds_sum", labels, metrics.duration_seconds[key])
        )
        lines.append(_sample("http_request_duration_seconds_count", labels, metrics.handled[key]))

    lines += [
        "# HELP service_ready 1 when this process can price a fixture, 0 otherwise.",
        "# TYPE service_ready gauge",
        _sample("service_ready", (), int(service.ready)),
        "# HELP service_component_ready Each dependency the service loads, 1 when it came up.",
        "# TYPE service_component_ready gauge",
    ]
    for component in service.components():
        lines.append(
            _sample(
                "service_component_ready", (("component", component.name),), int(component.ready)
            )
        )

    lines += [
        "# HELP fixtures_indexed Rows in the feature table this process can price.",
        "# TYPE fixtures_indexed gauge",
        _sample("fixtures_indexed", (), len(service.index) if service.index is not None else 0),
        "# HELP predictions_total Fixtures priced, cached and fresh alike.",
        "# TYPE predictions_total counter",
        _sample("predictions_total", (), service.served),
        "# HELP predictions_logged_total Predictions written to the prediction log.",
        "# TYPE predictions_logged_total counter",
        _sample("predictions_logged_total", (), service.logged),
        "# HELP prediction_cache_hits_total Fixtures answered without calling the model.",
        "# TYPE prediction_cache_hits_total counter",
        _sample("prediction_cache_hits_total", (), service.cache.hits),
        "# HELP prediction_cache_entries Fixtures currently held in the prediction cache.",
        "# TYPE prediction_cache_entries gauge",
        _sample("prediction_cache_entries", (), len(service.cache)),
    ]
    return "\n".join(lines) + "\n"


__all__ = ["CONTENT_TYPE", "UNMATCHED", "Metrics", "render"]
