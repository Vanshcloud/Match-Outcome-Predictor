"""What the last fetch learned about each provider file.

This is the state that makes an incremental re-run cheap *and* correct. For
every URL it records the HTTP validators the server returned, the checksum of
what was stored, and when it was last checked.

Two decisions worth stating.

**Validators, not file age.** The provider serves ``ETag`` and
``Last-Modified`` on every file and honours both ``If-None-Match`` and
``If-Modified-Since`` with a 304 carrying no body. So "has this changed?" has
an authoritative answer costing one round trip and zero bytes, and there is no
reason to guess from a modification time. An age heuristic is wrong in both
directions: it re-downloads unchanged files once they get old enough, and it
misses a file that changed five minutes ago. Both matter here — a settled
season is revised when the provider corrects a scoreline, and the current
season gains rows every match day.

**Misses are remembered too.** Most competition-season pairs do not exist:
leagues are founded, folded and restructured, and a full ingest asks for about
seven hundred files to find roughly six hundred and ninety. A miss costs a
request every run, and for this provider some misses cost a *whole download* —
an unavailable season answers HTTP 300 with an HTML body rather than 404, so
the page transfers before being recognised and deleted. Recording misses with a
timeout turns that recurring cost into a one-off, while still letting a league
that starts publishing be picked up later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

CACHE_VERSION = 1
CACHE_FILENAME = "fetch_cache.json"

# How long a recorded miss is trusted before the URL is tried again. Long
# enough that a full re-run costs nothing, short enough that a newly published
# division appears within a week without anyone clearing the cache by hand.
DEFAULT_MISS_TTL_DAYS = 7.0


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """What is known about one URL."""

    url: str
    fetched_at: str
    """ISO-8601 UTC. When this URL was last *checked*, which is not the same as
    when its content last changed — a 304 refreshes this and nothing else."""

    etag: str | None = None
    last_modified: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    missing: bool = False
    """The provider does not publish this file. Recorded so it is not re-probed
    on every run; re-tried once the TTL expires."""

    def age(self, now: datetime | None = None) -> timedelta:
        checked = datetime.fromisoformat(self.fetched_at)
        return (now or datetime.now(tz=UTC)) - checked

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class FetchCache:
    """A JSON sidecar mapping URL to :class:`CacheEntry`.

    Deliberately not a database. It holds a few hundred small records, is read
    once and written once per run, and being a diffable text file makes "why
    did it re-download that?" answerable by reading it.
    """

    def __init__(self, path: Path, entries: dict[str, CacheEntry] | None = None) -> None:
        self.path = path
        self._entries: dict[str, CacheEntry] = entries or {}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, url: object) -> bool:
        return url in self._entries

    @classmethod
    def load(cls, path: Path) -> FetchCache:
        """Read the cache, or start an empty one.

        A missing, unreadable or version-mismatched file yields an empty cache
        rather than an error. This is an optimisation, and an optimisation that
        can halt the pipeline is a liability: the worst a lost cache should
        ever cost is one slow run.
        """
        if not path.is_file():
            return cls(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("cache_version") != CACHE_VERSION:
                logger.info("fetch cache version changed; starting fresh")
                return cls(path)
            entries = {url: CacheEntry(**record) for url, record in raw.get("entries", {}).items()}
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            logger.warning("unreadable fetch cache at %s (%s); starting fresh", path, error)
            return cls(path)
        return cls(path, entries)

    def save(self) -> None:
        """Write the cache atomically.

        Via a temporary file and a replace, so an interrupted run leaves the
        previous cache intact rather than a half-written one that
        :meth:`load` would discard.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_version": CACHE_VERSION,
            "updated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "entries": {url: entry.to_json() for url, entry in sorted(self._entries.items())},
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def get(self, url: str) -> CacheEntry | None:
        return self._entries.get(url)

    def record_fetch(
        self,
        url: str,
        *,
        etag: str | None,
        last_modified: str | None,
        sha256: str | None,
        size: int | None,
        now: datetime | None = None,
    ) -> CacheEntry:
        """Record a successful fetch or revalidation."""
        entry = CacheEntry(
            url=url,
            fetched_at=(now or datetime.now(tz=UTC)).isoformat(timespec="seconds"),
            etag=etag,
            last_modified=last_modified,
            sha256=sha256,
            bytes=size,
            missing=False,
        )
        self._entries[url] = entry
        return entry

    def record_miss(self, url: str, now: datetime | None = None) -> CacheEntry:
        """Record that the provider does not publish this file."""
        entry = CacheEntry(
            url=url,
            fetched_at=(now or datetime.now(tz=UTC)).isoformat(timespec="seconds"),
            missing=True,
        )
        self._entries[url] = entry
        return entry

    def is_known_missing(
        self,
        url: str,
        *,
        ttl_days: float = DEFAULT_MISS_TTL_DAYS,
        now: datetime | None = None,
    ) -> bool:
        """Whether this URL was recently confirmed absent.

        A miss older than ``ttl_days`` is treated as unknown and retried, so a
        division that begins publishing is picked up without anyone having to
        know the cache exists.
        """
        entry = self._entries.get(url)
        if entry is None or not entry.missing:
            return False
        return entry.age(now) < timedelta(days=ttl_days)

    def forget(self, url: str) -> None:
        """Drop an entry, forcing the next run to fetch unconditionally."""
        self._entries.pop(url, None)
