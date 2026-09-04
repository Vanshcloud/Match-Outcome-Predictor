"""The inference service: the shipped model behind HTTP.

A separate top-level package rather than ``src/api``, on purpose. ``src`` is
the library — pipelines, models, probes, everything `make reproduce` runs — and
it has no idea a web server exists. This package is one of its *callers*, like
``scripts/``, and the dependency arrow points one way: ``api`` imports ``src``
and nothing in ``src`` imports ``api``. CI enforces that, along with the rule
that keeps this package from reaching around the pipeline layer into the
feature and rating builders — the one shortcut that would put a second,
unprobed implementation of the feature layer on the request path.

Four things live here and nothing else: the request and response schemas
(:mod:`api.schemas`), the object that answers a request (:mod:`api.service`),
the routes (:mod:`api.routes`), and the application that wires them together
(:mod:`api.main`). No arithmetic — every probability this package returns comes
out of :mod:`src.models.artifact` unchanged.
"""

from src import __version__

__all__ = ["__version__"]
