"""Logging setup, applied once per process.

Libraries configure loggers; applications configure handlers. Every module in
``src`` calls :func:`get_logger` and nothing else, so importing a module never
has a side effect on logging. Entry points — scripts, the API, the dashboard —
call :func:`configure_logging` exactly once at startup.
"""

from __future__ import annotations

import logging
import sys

_configured = False

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(
    level: str = "INFO",
    fmt: str = DEFAULT_FORMAT,
    *,
    force: bool = False,
) -> None:
    """Install a single stderr handler on the root logger.

    Calling this more than once is a no-op unless ``force`` is set. Repeated
    configuration is how a process ends up printing every line two or three
    times: each call adds another handler, and nothing removes the previous one.

    Logs go to stderr, not stdout, so a script can pipe its actual output
    somewhere without the log lines contaminating it.

    Args:
        level: Minimum level name, e.g. ``"DEBUG"``. Case-insensitive.
        fmt: ``logging`` format string.
        force: Reconfigure even if already configured, replacing existing
            handlers rather than adding to them. Intended for tests.
    """
    global _configured
    if _configured and not force:
        return

    root = logging.getLogger()
    # Handlers are replaced rather than appended, so `force=True` cannot leave a
    # duplicate behind and double every subsequent line.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(level.upper())
    # urllib3 logs every request line — method, path and query — at DEBUG. A
    # webhook URL is a credential whose secret *is* its path, so a DEBUG run
    # would write it to the log. Its warnings and errors still come through.
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return the logger for ``name``, conventionally the module's ``__name__``."""
    return logging.getLogger(name)


def reset_logging() -> None:
    """Drop the configured flag and every root handler. Test helper."""
    global _configured
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    _configured = False
