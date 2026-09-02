"""Logging is process-global state, which makes its failure modes order-dependent.

The bug worth guarding is duplicate handlers: every configure call adds one,
nothing removes it, and the symptom is every log line printed two or three
times — noticed late, and blamed on everything except configuration.
"""

from __future__ import annotations

import logging

from src.utils.logging import configure_logging, get_logger, reset_logging


def test_configure_installs_exactly_one_handler() -> None:
    configure_logging(force=True)
    assert len(logging.getLogger().handlers) == 1


def test_repeated_configuration_is_a_no_op() -> None:
    configure_logging(force=True)
    configure_logging()
    configure_logging()
    assert len(logging.getLogger().handlers) == 1


def test_force_replaces_rather_than_appends() -> None:
    """The distinction that matters: force must reconfigure without doubling."""
    configure_logging(force=True)
    configure_logging(level="DEBUG", force=True)
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.DEBUG


def test_level_is_case_insensitive() -> None:
    configure_logging(level="warning", force=True)
    assert logging.getLogger().level == logging.WARNING


def test_logs_go_to_stderr() -> None:
    """stdout belongs to a script's actual output. Log lines mixed into it turn
    `python scripts/x.py > out.csv` into a corrupt file."""
    import sys

    configure_logging(force=True)
    handler = logging.getLogger().handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stderr


def test_get_logger_returns_the_named_logger() -> None:
    assert get_logger("src.example").name == "src.example"


def test_reset_clears_handlers_and_allows_reconfiguration() -> None:
    configure_logging(force=True)
    reset_logging()
    assert logging.getLogger().handlers == []
    # Without force: proves reset actually cleared the flag rather than only
    # the handlers, which would leave the next configure silently ignored.
    configure_logging()
    assert len(logging.getLogger().handlers) == 1


def test_emitted_records_reach_the_stream(capsys) -> None:  # type: ignore[no-untyped-def]
    """capsys, not caplog: `configure_logging(force=True)` removes every root
    handler by design, pytest's own capture handler included, so caplog is
    structurally unable to observe this. capsys reads the stderr the handler
    actually writes to."""
    configure_logging(level="INFO", force=True)
    get_logger("src.example").info("hello %s", "world")
    assert "hello world" in capsys.readouterr().err


def test_a_record_below_the_level_is_dropped(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging(level="WARNING", force=True)
    get_logger("src.example").info("should not appear")
    assert capsys.readouterr().err == ""
