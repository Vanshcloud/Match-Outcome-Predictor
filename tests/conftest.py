"""Shared fixtures.

Deliberately thin. No match data is committed to this repository — the provider
publishes no redistribution licence — so there are no data fixtures to share.
Synthetic canonical tables are built by :mod:`tests.factories` instead, which
are constructors rather than fixtures because several tests need one at module
scope where a fixture cannot reach.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.utils.logging import reset_logging


@pytest.fixture(autouse=True)
def _isolate_logging() -> Iterator[None]:
    """Reset logging state around every test, then put pytest's back.

    ``configure_logging`` is process-global and idempotent by design. Without
    the reset, whichever test configured it first would decide the level for
    every test after it and the suite's result would depend on collection
    order.

    The restore half is the part that is easy to omit and expensive to debug:
    ``reset_logging`` removes *every* root handler, pytest's own log-capture
    handler included. Left removed, the ``caplog`` fixture silently sees
    nothing in every later test — a fixture that fails by returning empty
    rather than by raising.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level

    reset_logging()
    yield

    reset_logging()
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """A minimal but complete config file, written to a temp directory.

    Complete rather than partial: ``Settings`` requires all four sections, and
    a fixture that omits one would make every test assert the same
    ValidationError instead of the behaviour it is about.
    """
    path = tmp_path / "config.yaml"
    path.write_text(
        "paths:\n"
        "  data_dir: data\n"
        "  model_dir: models\n"
        "http:\n"
        "  timeout_seconds: 30\n"
        "api:\n"
        "  host: 127.0.0.1\n"
        "  port: 8000\n"
        "logging:\n"
        "  level: INFO\n",
        encoding="utf-8",
    )
    return path
