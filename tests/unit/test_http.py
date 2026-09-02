"""The HTTP client is the only code here that talks to a third party.

No test reaches the network: every one mounts a stub adapter on an injected
session, which is the reason ``HttpClient`` takes a ``session`` argument at all.

The behaviour under test that is easiest to get wrong and hardest to notice is
the atomic download. A truncated CSV still parses — a season quietly missing
its last eight match weeks would corrupt every rolling feature computed from it
and raise nothing at all.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import requests
from requests.adapters import BaseAdapter
from requests.structures import CaseInsensitiveDict

from src.utils import http as http_module
from src.utils.http import RETRY_METHODS, RETRY_STATUSES, HttpClient

URL = "https://example.test/E0.csv"


class _StubAdapter(BaseAdapter):
    """Returns a canned response instead of making a request."""

    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        stream_factory: object = None,
    ) -> None:
        super().__init__()
        self.status = status
        self.body = body
        self.headers = headers or {}
        # Lets one test hand back a stream that fails part-way through, which
        # is the interrupted-download case.
        self.stream_factory = stream_factory
        self.calls: list[str] = []

    def send(self, request, **_kwargs):  # type: ignore[no-untyped-def]
        # The underscore marks these as deliberately unused: BaseAdapter's
        # contract requires accepting stream/timeout/verify/cert/proxies, and
        # a stub honours none of them.
        self.calls.append(str(request.url))
        response = requests.Response()
        response.status_code = self.status
        response.reason = "OK" if self.status == 200 else "Error"
        response.url = str(request.url)
        response.headers = CaseInsensitiveDict(self.headers)
        raw = self.stream_factory() if self.stream_factory else io.BytesIO(self.body)
        response.raw = raw
        return response

    def close(self) -> None:
        return None


def make_client(adapter: BaseAdapter, **kwargs: object) -> HttpClient:
    """Build a client whose transport is ``adapter``.

    The stub is mounted AFTER construction, not before, and the order is
    load-bearing: ``HttpClient.__init__`` mounts its own retry adapter on both
    schemes, so a stub mounted first is silently replaced and every test here
    reaches the real network — which presents as the suite hanging on DNS
    rather than as a failure.
    """
    session = requests.Session()
    client = HttpClient(session=session, **kwargs)  # type: ignore[arg-type]
    session.mount("https://", adapter)
    return client


# ---- requests ---------------------------------------------------------------


def test_get_returns_the_body() -> None:
    client = make_client(_StubAdapter(body=b"Div,Date\nE0,16/08/2024\n"))
    assert client.get(URL).content.startswith(b"Div,Date")


def test_an_error_status_raises() -> None:
    """404 rather than 500 on purpose: a missing season file is an expected
    condition here, and it must surface as an exception the adapter can catch
    rather than an empty body it would happily parse as zero matches."""
    client = make_client(_StubAdapter(status=404))
    with pytest.raises(requests.HTTPError):
        client.get(URL)


def test_the_user_agent_identifies_this_client() -> None:
    client = make_client(_StubAdapter(), user_agent="match-outcome-predictor/9.9")
    assert client.session.headers["User-Agent"] == "match-outcome-predictor/9.9"


def test_the_configured_timeout_is_applied() -> None:
    """A request with no timeout can hang forever, and a nightly ingest that
    hangs is indistinguishable from one that is merely slow."""
    seen: dict[str, object] = {}
    adapter = _StubAdapter()
    original = adapter.send

    def spy(request, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return original(request, **kwargs)

    adapter.send = spy  # type: ignore[method-assign]
    make_client(adapter, timeout_seconds=12.5).get(URL)
    assert seen["timeout"] == 12.5


def test_an_explicit_timeout_wins_over_the_default() -> None:
    seen: dict[str, object] = {}
    adapter = _StubAdapter()
    original = adapter.send

    def spy(request, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return original(request, **kwargs)

    adapter.send = spy  # type: ignore[method-assign]
    make_client(adapter, timeout_seconds=12.5).get(URL, timeout=1.0)
    assert seen["timeout"] == 1.0


# ---- retry policy -----------------------------------------------------------


def test_retries_cover_transient_statuses_only() -> None:
    """Asserted on the mounted Retry rather than by driving failures: the point
    is the policy, and 404 being absent from it is the load-bearing part."""
    client = HttpClient(max_retries=3)
    retry = client.session.get_adapter("https://example.test").max_retries
    assert retry.total == 3
    assert set(retry.status_forcelist) == set(RETRY_STATUSES)
    assert 404 not in retry.status_forcelist
    assert retry.allowed_methods == RETRY_METHODS
    assert retry.respect_retry_after_header
    client.close()


# ---- rate limiting ----------------------------------------------------------


def test_consecutive_requests_are_spaced(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(http_module.time, "sleep", slept.append)

    client = make_client(_StubAdapter(), min_request_interval_seconds=5.0)
    client.get(URL)
    client.get(URL)

    assert len(slept) == 1, "expected exactly one wait, before the second request"
    assert 0 < slept[0] <= 5.0


def test_the_first_request_does_not_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(http_module.time, "sleep", slept.append)
    make_client(_StubAdapter(), min_request_interval_seconds=5.0).get(URL)
    assert slept == []


def test_a_zero_interval_disables_rate_limiting(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(http_module.time, "sleep", slept.append)
    client = make_client(_StubAdapter(), min_request_interval_seconds=0.0)
    client.get(URL)
    client.get(URL)
    assert slept == []


def test_a_failed_request_still_spaces_the_next_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timestamp is recorded in a `finally`. Retrying a struggling host at
    full speed is how a transient 503 becomes a block."""
    slept: list[float] = []
    monkeypatch.setattr(http_module.time, "sleep", slept.append)

    client = make_client(_StubAdapter(status=500), min_request_interval_seconds=5.0)
    for _ in range(2):
        with pytest.raises(requests.HTTPError):
            client.get(URL)
    assert len(slept) == 1


# ---- downloads --------------------------------------------------------------


def test_download_writes_the_file(tmp_path: Path) -> None:
    body = b"Div,Date,HomeTeam\n" * 100
    client = make_client(_StubAdapter(body=body, headers={"Content-Length": str(len(body))}))
    result = client.download(URL, tmp_path / "E0.csv")
    assert result.path.read_bytes() == body
    assert result.modified is True
    assert result.bytes_written == len(body)


def test_download_creates_missing_parent_directories(tmp_path: Path) -> None:
    client = make_client(_StubAdapter(body=b"x"))
    client.download(URL, tmp_path / "raw" / "2425" / "E0.csv")
    assert (tmp_path / "raw" / "2425" / "E0.csv").is_file()


def test_download_leaves_no_part_file_behind(tmp_path: Path) -> None:
    client = make_client(_StubAdapter(body=b"x"))
    client.download(URL, tmp_path / "E0.csv")
    assert list(tmp_path.glob("*.part")) == []


def test_download_overwrites_an_existing_file(tmp_path: Path) -> None:
    """A refetch replaces last week's copy. `.replace` rather than `.rename`
    is what makes this work on Windows too."""
    destination = tmp_path / "E0.csv"
    destination.write_bytes(b"stale")
    client = make_client(_StubAdapter(body=b"fresh"))
    client.download(URL, destination)
    assert destination.read_bytes() == b"fresh"


def test_a_truncated_download_raises_and_is_deleted(tmp_path: Path) -> None:
    """The core guarantee. The server promises 500 bytes and sends 5; the
    result must be an exception and no file, never a short CSV that parses
    cleanly into a season missing most of its matches."""
    client = make_client(_StubAdapter(body=b"short", headers={"Content-Length": "500"}))
    with pytest.raises(OSError, match="truncated"):
        client.download(URL, tmp_path / "E0.csv")
    assert not (tmp_path / "E0.csv").exists()
    assert list(tmp_path.glob("*.part")) == []


def test_an_interrupted_stream_leaves_nothing_behind(tmp_path: Path) -> None:
    class _FailingStream(io.BytesIO):
        def read(self, _size: int | None = -1) -> bytes:
            raise ConnectionError("connection reset mid-stream")

    client = make_client(_StubAdapter(stream_factory=_FailingStream))
    with pytest.raises(ConnectionError):
        client.download(URL, tmp_path / "E0.csv")
    assert not (tmp_path / "E0.csv").exists()
    assert list(tmp_path.glob("*.part")) == []


def test_a_response_without_content_length_is_accepted(tmp_path: Path) -> None:
    """Omitting the header is legal. There is simply nothing to compare
    against, so the size check is skipped rather than the download rejected."""
    client = make_client(_StubAdapter(body=b"abc"))
    assert client.download(URL, tmp_path / "E0.csv").path.read_bytes() == b"abc"


def test_an_error_status_aborts_the_download(tmp_path: Path) -> None:
    client = make_client(_StubAdapter(status=404))
    with pytest.raises(requests.HTTPError):
        client.download(URL, tmp_path / "E0.csv")
    assert not (tmp_path / "E0.csv").exists()


# ---- lifecycle --------------------------------------------------------------


def test_the_context_manager_closes_the_session() -> None:
    closed: list[bool] = []
    session = requests.Session()
    session.close = lambda: closed.append(True)  # type: ignore[method-assign]

    with HttpClient(session=session) as client:
        assert isinstance(client, HttpClient)
    assert closed == [True]


# ---- conditional requests ---------------------------------------------------
#
# The provider serves ETag and Last-Modified on every file and answers both
# If-None-Match and If-Modified-Since with a 304 carrying no body. That is what
# makes an incremental re-run cost nothing and, more importantly, makes it
# correct — a cache decision based on file age is a guess.


def test_validators_are_sent_when_a_copy_exists(tmp_path: Path) -> None:
    destination = tmp_path / "E0.csv"
    destination.write_bytes(b"cached")
    seen: dict[str, str] = {}
    adapter = _StubAdapter(body=b"new")
    original = adapter.send

    def spy(request, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(request.headers)
        return original(request, **kwargs)

    adapter.send = spy  # type: ignore[method-assign]
    make_client(adapter).download(
        URL, destination, etag='"v1"', last_modified="Thu, 01 Jan 2026 00:00:00 GMT"
    )
    assert seen["If-None-Match"] == '"v1"'
    assert seen["If-Modified-Since"] == "Thu, 01 Jan 2026 00:00:00 GMT"


def test_no_validators_are_sent_without_a_local_copy(tmp_path: Path) -> None:
    """Sending them for a file that is not on disk would earn a 304 and leave
    nothing to read."""
    seen: dict[str, str] = {}
    adapter = _StubAdapter(body=b"new")
    original = adapter.send

    def spy(request, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(request.headers)
        return original(request, **kwargs)

    adapter.send = spy  # type: ignore[method-assign]
    make_client(adapter).download(URL, tmp_path / "absent.csv", etag='"v1"')
    assert "If-None-Match" not in seen


def test_an_empty_cached_file_is_not_revalidated(tmp_path: Path) -> None:
    """A zero-byte file is what an interrupted write leaves behind. Treating it
    as a valid copy would let a 304 confirm nothing."""
    destination = tmp_path / "E0.csv"
    destination.write_bytes(b"")
    seen: dict[str, str] = {}
    adapter = _StubAdapter(body=b"new")
    original = adapter.send

    def spy(request, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(request.headers)
        return original(request, **kwargs)

    adapter.send = spy  # type: ignore[method-assign]
    make_client(adapter).download(URL, destination, etag='"v1"')
    assert "If-None-Match" not in seen


def test_a_304_leaves_the_file_untouched(tmp_path: Path) -> None:
    destination = tmp_path / "E0.csv"
    destination.write_bytes(b"original")
    client = make_client(_StubAdapter(status=304, headers={"ETag": '"v1"'}))

    result = client.download(URL, destination, etag='"v1"')

    assert result.modified is False
    assert result.bytes_written == 0
    assert destination.read_bytes() == b"original"
    assert result.etag == '"v1"'


def test_a_304_without_validators_keeps_the_ones_we_sent(tmp_path: Path) -> None:
    """A 304 is permitted to omit them. Dropping them here would make the next
    run unconditional, quietly undoing the whole mechanism."""
    destination = tmp_path / "E0.csv"
    destination.write_bytes(b"original")
    client = make_client(_StubAdapter(status=304))

    result = client.download(
        URL, destination, etag='"v1"', last_modified="Thu, 01 Jan 2026 00:00:00 GMT"
    )
    assert result.etag == '"v1"'
    assert result.last_modified == "Thu, 01 Jan 2026 00:00:00 GMT"


def test_a_200_reports_the_new_validators(tmp_path: Path) -> None:
    client = make_client(
        _StubAdapter(
            body=b"new", headers={"ETag": '"v2"', "Last-Modified": "Fri, 02 Jan 2026 00:00:00 GMT"}
        )
    )
    result = client.download(URL, tmp_path / "E0.csv", etag='"v1"')
    assert result.modified is True
    assert result.etag == '"v2"'
    assert result.last_modified == "Fri, 02 Jan 2026 00:00:00 GMT"


def test_caller_supplied_headers_are_preserved(tmp_path: Path) -> None:
    destination = tmp_path / "E0.csv"
    destination.write_bytes(b"cached")
    seen: dict[str, str] = {}
    adapter = _StubAdapter(body=b"new")
    original = adapter.send

    def spy(request, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(request.headers)
        return original(request, **kwargs)

    adapter.send = spy  # type: ignore[method-assign]
    make_client(adapter).download(URL, destination, etag='"v1"', headers={"X-Trace": "abc"})
    assert seen["X-Trace"] == "abc"
    assert seen["If-None-Match"] == '"v1"'


def test_non_mapping_headers_are_refused(tmp_path: Path) -> None:
    """Narrowed rather than silenced: a caller passing headers as something
    other than a mapping should fail loudly instead of losing them."""
    destination = tmp_path / "E0.csv"
    destination.write_bytes(b"cached")
    with pytest.raises(TypeError, match="headers must be a mapping"):
        make_client(_StubAdapter()).download(URL, destination, etag='"v1"', headers=["bad"])
