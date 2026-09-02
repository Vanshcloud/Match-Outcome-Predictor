"""A polite, retrying HTTP client shared by every ingestion adapter.

One client for all outbound traffic, so timeout, retry, backoff, User-Agent and
rate-limit policy are set in one place rather than rediscovered per provider.

This module is deliberately transport-only. It knows how to fetch bytes
politely and nothing about football: caching, freshness and schema parsing
belong to the provider adapters in ``src.ingestion``, which is what lets a new
provider reuse this without inheriting another provider's caching decisions.

``requests`` only. Every provider this project reads serves static CSV, so no
browser automation and no HTML parser is installed.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import TracebackType

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src import __version__
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Built from the single source of truth rather than repeated as a literal. The
# User-Agent identifies this client to a third party on every request, and a
# hardcoded version becomes a small lie at the first release — which is exactly
# what happened at 0.2.0, caught by tests/unit/test_version.py.
#
# `from src import __version__` is the one permitted intra-package import in
# src/utils: src/__init__.py imports nothing, so it cannot create a cycle.
DEFAULT_USER_AGENT = (
    f"match-outcome-predictor/{__version__} "
    "(+https://github.com/Vanshcloud/Match-Outcome-Predictor)"
)

# Transient failures only. A 404 is deliberately absent: the file is missing,
# and asking four more times only delays the error by the backoff schedule.
# This matters here because a missing season file is a normal, expected
# condition — not every division has played every season.
RETRY_STATUSES = (429, 500, 502, 503, 504)

# Retries are confined to methods that are safe to repeat. This project only
# reads, but the restriction is stated rather than assumed, so adding a POST
# later cannot silently inherit automatic replay.
RETRY_METHODS = frozenset({"GET", "HEAD"})


class HttpClient:
    """A ``requests.Session`` with retry, backoff, timeout and rate limiting.

    Usable as a context manager so the underlying connection pool is closed
    rather than left to the garbage collector.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        max_retries: int = 5,
        backoff_factor: float = 0.5,
        user_agent: str = DEFAULT_USER_AGENT,
        min_request_interval_seconds: float = 0.0,
        session: requests.Session | None = None,
    ) -> None:
        """Configure the client.

        Args:
            timeout_seconds: Per-request timeout. Applied to every call unless
                a caller passes its own.
            max_retries: Attempts after the first for a transient failure.
            backoff_factor: urllib3 backoff base; delay grows exponentially.
            user_agent: Sent on every request. Identify the client honestly —
                an anonymous, scraper-shaped UA is how a polite consumer gets
                mistaken for an impolite one and blocked.
            min_request_interval_seconds: Minimum spacing between requests.
                This project fetches ~700 files from a single small host, so
                spacing them is a courtesy that costs nothing.
            session: Injected for tests, which use a session with a mounted
                mock adapter so no test ever reaches the network.
        """
        self.timeout_seconds = timeout_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self._last_request_at: float | None = None

        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=RETRY_STATUSES,
            allowed_methods=RETRY_METHODS,
            # False: let the response come back so `raise_for_status` produces
            # the error, giving one exception type for callers to handle
            # instead of two that mean the same thing.
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _wait_for_rate_limit(self) -> None:
        """Sleep until the configured interval since the last request elapses."""
        if self.min_request_interval_seconds <= 0 or self._last_request_at is None:
            return
        # monotonic, not time(): a clock adjustment mid-run must not turn the
        # remaining wait into a negative number or a very long sleep.
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_request_interval_seconds - elapsed
        if remaining > 0:
            logger.debug("rate limit: sleeping %.2fs", remaining)
            time.sleep(remaining)

    def get(self, url: str, **kwargs: object) -> requests.Response:
        """GET ``url``, honouring the configured timeout and rate limit.

        Raises:
            requests.HTTPError: On any 4xx or 5xx that survived the retries.
        """
        self._wait_for_rate_limit()
        kwargs.setdefault("timeout", self.timeout_seconds)
        try:
            response = self.session.get(url, **kwargs)  # type: ignore[arg-type]
        finally:
            # In the `finally` so a failed request still spaces out the next
            # one. Retrying a struggling host at full speed is how a transient
            # 503 becomes a block.
            self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response

    def download(self, url: str, destination: Path, **kwargs: object) -> Path:
        """Stream ``url`` to ``destination``, atomically.

        Written to a ``.part`` file and renamed only on success, so an
        interrupted download can never be mistaken for a complete one. That
        distinction is load-bearing here: a truncated CSV still parses, and a
        season silently missing its last eight match weeks would corrupt every
        rolling feature computed from it without raising anything.

        Returns:
            ``destination``, for chaining.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")

        kwargs.setdefault("stream", True)
        logger.info("downloading %s -> %s", url, destination.name)
        response = self.get(url, **kwargs)

        # Any failure below must remove the .part file. A short read left on
        # disk is worse than no file at all, because a freshness check only
        # asks whether a file exists and is non-empty.
        try:
            bytes_written = 0
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
                        bytes_written += len(chunk)

            # urllib3 usually raises ChunkedEncodingError when a response
            # undershoots its declared length. This is the backstop for when it
            # does not. Skipped when the server sends no Content-Length, which
            # is legal and means there is nothing to compare against.
            expected = response.headers.get("Content-Length")
            if expected is not None and bytes_written != int(expected):
                raise OSError(
                    f"truncated download for {destination.name}: "
                    f"got {bytes_written} bytes, expected {expected}"
                )
        except BaseException:
            # BaseException, not Exception: a KeyboardInterrupt mid-download
            # must not leave a partial file behind either.
            partial.unlink(missing_ok=True)
            raise

        # replace(), not rename(): atomic on POSIX and overwrites on Windows,
        # so re-downloading over an existing file works on both.
        partial.replace(destination)
        logger.info("downloaded %s (%.2f MB)", destination.name, bytes_written / 1e6)
        return destination

    def close(self) -> None:
        """Close the connection pool."""
        self.session.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
