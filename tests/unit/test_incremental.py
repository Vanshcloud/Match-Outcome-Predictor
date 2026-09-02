"""Re-running the ingest must be cheap, safe and idempotent.

The four properties under test, each of which was a real gap before this suite
existed:

1. **Only new or modified files are downloaded.** Enforced by HTTP conditional
   requests, not by file age. The provider serves ``ETag`` and
   ``Last-Modified`` on every file and answers ``If-None-Match`` with a 304
   carrying no body, so "has this changed?" has an authoritative answer. An age
   heuristic is wrong in both directions: it re-downloads unchanged files once
   they get old, and misses a file that changed minutes ago.
2. **Historical checksums are preserved.** Every cached provider file is
   recorded in a raw manifest, so which bytes produced a given table is
   checkable rather than hopeful.
3. **No duplicate matches**, however many times a season is ingested.
4. **Identical output on repeated runs.**
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.base import SeasonUnavailableError
from src.ingestion.cache import CACHE_FILENAME, FetchCache
from src.ingestion.football_data import FootballDataProvider
from src.ingestion.manifest import read_manifest, verify_manifest
from src.ingestion.registry import Competition, Feed, Registry
from src.pipelines.ingest import MATCHES_FILENAME, run_ingest
from src.utils.http import DownloadResult

TODAY = date(2026, 9, 3)

ENG = Competition(
    id="ENG_1", country="England", name="Premier League", tier=1, feed=Feed.MAIN, code="E0"
)
BRA = Competition(id="BRA_1", country="Brazil", name="Serie A", tier=1, feed=Feed.EXTRA, code="BRA")
REGISTRY = Registry(earliest_season=2024, competitions=(ENG, BRA))

HEADER = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
MATCHDAY_1 = HEADER + "E0,16/08/2024,20:00,Man United,Fulham,1,0,H\n"
MATCHDAY_2 = MATCHDAY_1 + "E0,17/08/2024,15:00,Arsenal,Wolves,2,0,H\n"


class FakeServer:
    """A provider that serves versioned bodies and honours conditional requests.

    Counts full transfers separately from revalidations, because "did we
    re-download it?" is the entire question this suite asks.
    """

    def __init__(self, bodies: dict[str, str]) -> None:
        self.bodies = bodies
        self.version: dict[str, int] = dict.fromkeys(bodies, 1)
        self.transfers: list[str] = []
        self.revalidations: list[str] = []
        self.requests: list[str] = []
        self.validators: list[tuple[str | None, str | None]] = []

    def publish(self, key: str, body: str) -> None:
        """Change a file, as the provider does when a match is played."""
        self.bodies[key] = body
        self.version[key] += 1

    def _key(self, url: str) -> str | None:
        return next((k for k in self.bodies if url.endswith(k)), None)

    def etag_for(self, key: str) -> str:
        return f'"{key}-v{self.version[key]}"'

    def download(
        self,
        url: str,
        destination: Path,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        **_kwargs: object,
    ) -> DownloadResult:
        self.requests.append(url)
        # Recorded, not ignored: the parameter must stay bound by name, or a
        # client that quietly stopped sending validators would still look
        # correct here. Matching is done on the ETag alone — a real server may
        # honour either, and one match is enough for a 304.
        self.validators.append((etag, last_modified))
        key = self._key(url)
        if key is None:
            import requests as _requests

            response = _requests.Response()
            response.status_code = 404
            raise _requests.HTTPError("404", response=response)

        current = self.etag_for(key)
        if etag == current and destination.is_file():
            self.revalidations.append(url)
            return DownloadResult(path=destination, modified=False, etag=current)

        self.transfers.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.bodies[key], encoding="utf-8")
        return DownloadResult(
            path=destination,
            modified=True,
            etag=current,
            bytes_written=len(self.bodies[key].encode()),
        )

    def close(self) -> None:
        return None


def make_provider(server: FakeServer, tmp_path: Path, **kwargs: object) -> FootballDataProvider:
    raw = tmp_path / "raw"
    return FootballDataProvider(
        REGISTRY,
        raw,
        client=server,  # type: ignore[arg-type]
        today=TODAY,
        cache=FetchCache.load(raw / CACHE_FILENAME),
        **kwargs,  # type: ignore[arg-type]
    )


# ---- 1. only new or modified files are downloaded ---------------------------


def test_an_unchanged_file_is_revalidated_but_not_transferred(tmp_path: Path) -> None:
    """The core of an incremental run. The second fetch still asks the server —
    it must, or a change would be invisible — but no bytes come back."""
    server = FakeServer({"2526/E0.csv": MATCHDAY_1})

    first = make_provider(server, tmp_path)
    first.fetch(ENG, "2025-26")
    first.save_cache()

    second = make_provider(server, tmp_path)
    second.fetch(ENG, "2025-26")

    assert server.transfers == ["https://www.football-data.co.uk/mmz4281/2526/E0.csv"]
    assert len(server.revalidations) == 1
    # The first request carried no validator (nothing was cached); the second
    # carried the ETag the first was given. Without this the test would pass
    # for a client that never sent one and a server that never checked.
    assert server.validators[0] == (None, None)
    assert server.validators[1][0] == server.etag_for("2526/E0.csv")


def test_a_modified_file_is_downloaded_again(tmp_path: Path) -> None:
    """The provider adds today's matches to the current season file. The next
    run must see them — this is the case an age-based cache gets wrong when the
    file was refreshed minutes ago."""
    server = FakeServer({"2526/E0.csv": MATCHDAY_1})

    first = make_provider(server, tmp_path)
    assert len(first.fetch(ENG, "2025-26")) == 1
    first.save_cache()

    server.publish("2526/E0.csv", MATCHDAY_2)

    second = make_provider(server, tmp_path)
    frame = second.fetch(ENG, "2025-26")

    assert len(frame) == 2, "the newly published match was not picked up"
    assert len(server.transfers) == 2
    assert set(frame["home_team"]) == {"Man United", "Arsenal"}


def test_a_settled_season_is_not_even_revalidated(tmp_path: Path) -> None:
    """A finished season gains no rows, so the ordinary run does not ask about
    it at all. This is what keeps a re-run over three decades nearly free."""
    server = FakeServer({"2425/E0.csv": MATCHDAY_1})

    first = make_provider(server, tmp_path)
    first.fetch(ENG, "2024-25")
    first.save_cache()

    server.requests.clear()
    make_provider(server, tmp_path).fetch(ENG, "2024-25")
    assert server.requests == []


def test_revalidate_rechecks_settled_seasons(tmp_path: Path) -> None:
    """Settled means "will not grow", not "will never change" — the provider
    does correct a scoreline. `--revalidate` catches that for the price of a
    304, and trusting a settled file forever made corrections permanently
    invisible."""
    server = FakeServer({"2425/E0.csv": MATCHDAY_1})

    first = make_provider(server, tmp_path)
    first.fetch(ENG, "2024-25")
    first.save_cache()

    server.publish("2425/E0.csv", MATCHDAY_2)

    corrected = make_provider(server, tmp_path, revalidate=True)
    assert len(corrected.fetch(ENG, "2024-25")) == 2


def test_the_secondary_feed_is_revalidated_every_run(tmp_path: Path) -> None:
    """One country file holds every season including the current one, so it can
    never be considered settled."""
    body = (
        "Country,League,Season,Date,Time,Home,Away,HG,AG,Res\n"
        "Brazil,Serie A,2024,14/04/2024,20:00,Palmeiras,Santos,2,1,H\n"
    )
    server = FakeServer({"new/BRA.csv": body})

    first = make_provider(server, tmp_path)
    first.fetch(BRA, "2024")
    first.save_cache()

    second = make_provider(server, tmp_path)
    second.fetch(BRA, "2024")
    assert len(server.transfers) == 1
    assert len(server.revalidations) == 1


# ---- misses are remembered --------------------------------------------------


def test_a_known_missing_season_is_not_re_requested(tmp_path: Path) -> None:
    """Most competition-season pairs do not exist, and for this provider some
    misses cost a whole HTML download because an unavailable season answers
    HTTP 300 rather than 404. Remembering them turns a recurring cost into a
    one-off."""
    server = FakeServer({"2526/E0.csv": MATCHDAY_1})

    first = make_provider(server, tmp_path)
    with pytest.raises(SeasonUnavailableError):
        first.fetch(ENG, "2024-25")
    first.save_cache()

    server.requests.clear()
    second = make_provider(server, tmp_path)
    with pytest.raises(SeasonUnavailableError, match="known missing"):
        second.fetch(ENG, "2024-25")
    assert server.requests == []


def test_a_stale_miss_is_retried(tmp_path: Path) -> None:
    """A division that begins publishing must be picked up without anyone
    having to know the cache exists."""
    server = FakeServer({})
    raw = tmp_path / "raw"
    cache = FetchCache.load(raw / CACHE_FILENAME)
    url = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
    cache.record_miss(url, now=datetime.now(tz=UTC) - timedelta(days=30))
    cache.save()

    server.bodies["2425/E0.csv"] = MATCHDAY_1
    server.version["2425/E0.csv"] = 1

    provider = FootballDataProvider(
        REGISTRY,
        raw,
        client=server,
        today=TODAY,  # type: ignore[arg-type]
        cache=FetchCache.load(raw / CACHE_FILENAME),
    )
    assert len(provider.fetch(ENG, "2024-25")) == 1


# ---- 2. historical checksums are preserved ----------------------------------


def test_the_raw_manifest_records_every_cached_file(tmp_path: Path) -> None:
    server = FakeServer({"2526/E0.csv": MATCHDAY_1})
    provider = make_provider(server, tmp_path)
    report = run_ingest(provider, tmp_path / "processed", competitions=(ENG,))

    assert report.raw_manifest is not None
    manifest = read_manifest(report.raw_manifest)
    assert manifest["kind"] == "raw-provider-files"
    assert verify_manifest(manifest, provider.raw_dir).ok


def test_the_raw_manifest_detects_a_silently_revised_file(tmp_path: Path) -> None:
    """The failure this exists to catch: the provider corrects a file, the
    table is rebuilt, and nothing records that the inputs moved."""
    server = FakeServer({"2526/E0.csv": MATCHDAY_1})
    provider = make_provider(server, tmp_path)
    report = run_ingest(provider, tmp_path / "processed", competitions=(ENG,))
    assert report.raw_manifest is not None
    manifest = read_manifest(report.raw_manifest)

    provider.local_path(ENG, "2025-26").write_text(MATCHDAY_2, encoding="utf-8")

    result = verify_manifest(manifest, provider.raw_dir)
    assert not result.ok
    assert result.changed


def test_the_cache_records_the_checksum_of_what_was_stored(tmp_path: Path) -> None:
    server = FakeServer({"2526/E0.csv": MATCHDAY_1})
    provider = make_provider(server, tmp_path)
    provider.fetch(ENG, "2025-26")

    entry = provider.cache.get("https://www.football-data.co.uk/mmz4281/2526/E0.csv")
    assert entry is not None
    assert entry.sha256 is not None
    assert entry.bytes == len(MATCHDAY_1.encode())


def test_a_304_does_not_erase_the_recorded_checksum(tmp_path: Path) -> None:
    """A revalidation transfers no bytes, so the checksum it recorded last time
    still describes the file. Dropping it would leave the manifest unable to
    say what the cached bytes were."""
    server = FakeServer({"2526/E0.csv": MATCHDAY_1})
    first = make_provider(server, tmp_path)
    first.fetch(ENG, "2025-26")
    first.save_cache()
    original = first.cache.get("https://www.football-data.co.uk/mmz4281/2526/E0.csv")

    second = make_provider(server, tmp_path)
    second.fetch(ENG, "2025-26")
    after = second.cache.get("https://www.football-data.co.uk/mmz4281/2526/E0.csv")

    assert original is not None and after is not None
    assert after.sha256 == original.sha256
    assert after.fetched_at >= original.fetched_at


# ---- 3. no duplicate matches ------------------------------------------------


def test_re_ingesting_produces_no_duplicates(tmp_path: Path) -> None:
    """match_id is a hash of the natural key, so the same fixture ingested
    twice collapses to one row rather than doubling."""
    server = FakeServer({"2526/E0.csv": MATCHDAY_1})
    processed = tmp_path / "processed"

    for _ in range(3):
        provider = make_provider(server, tmp_path)
        run_ingest(provider, processed, competitions=(ENG,))

    frame = pd.read_parquet(processed / MATCHES_FILENAME)
    assert len(frame) == 1
    assert frame["match_id"].is_unique


def test_appending_a_match_does_not_disturb_existing_ids(tmp_path: Path) -> None:
    """An incremental update adds rows; it must not renumber the ones already
    there, or every prediction and evaluation keyed to a match_id breaks."""
    server = FakeServer({"2526/E0.csv": MATCHDAY_1})
    processed = tmp_path / "processed"

    provider = make_provider(server, tmp_path)
    run_ingest(provider, processed, competitions=(ENG,))
    before = pd.read_parquet(processed / MATCHES_FILENAME)

    server.publish("2526/E0.csv", MATCHDAY_2)
    provider = make_provider(server, tmp_path)
    run_ingest(provider, processed, competitions=(ENG,))
    after = pd.read_parquet(processed / MATCHES_FILENAME)

    assert len(after) == 2
    original_id = before.iloc[0]["match_id"]
    assert original_id in set(after["match_id"])
    kept = after[after["match_id"] == original_id].iloc[0]
    assert kept["home_team"] == before.iloc[0]["home_team"]
    assert kept["home_goals"] == before.iloc[0]["home_goals"]


# ---- 4. identical output on repeated runs -----------------------------------


def test_repeated_runs_produce_byte_identical_output(tmp_path: Path) -> None:
    """Not merely equal row counts. A byte-identical Parquet is what makes the
    output checksum in the manifest meaningful, and what lets a rebuild be
    proven to have changed nothing."""
    server = FakeServer({"2526/E0.csv": MATCHDAY_2})
    processed = tmp_path / "processed"

    digests = []
    for _ in range(3):
        provider = make_provider(server, tmp_path)
        run_ingest(provider, processed, competitions=(ENG,))
        digests.append((processed / MATCHES_FILENAME).read_bytes())

    assert digests[0] == digests[1] == digests[2]


def test_output_is_independent_of_competition_order(tmp_path: Path) -> None:
    """Ingesting A then B must produce the same table as B then A. Without a
    deterministic final sort the row order would follow the fetch order, and
    the output checksum would depend on the command line."""
    body = (
        "Country,League,Season,Date,Time,Home,Away,HG,AG,Res\n"
        "Brazil,Serie A,2024,14/04/2024,20:00,Palmeiras,Santos,2,1,H\n"
    )
    server = FakeServer({"2526/E0.csv": MATCHDAY_2, "new/BRA.csv": body})

    outputs = []
    for order in ((ENG, BRA), (BRA, ENG)):
        processed = tmp_path / f"processed-{order[0].id}"
        provider = make_provider(server, tmp_path)
        run_ingest(provider, processed, competitions=order)
        outputs.append((processed / MATCHES_FILENAME).read_bytes())

    assert outputs[0] == outputs[1]


def test_the_cache_survives_a_run_that_ingested_nothing(tmp_path: Path) -> None:
    """A run that found no data still learned which files are missing. Throwing
    that away would make the next run pay for it again."""
    server = FakeServer({})
    provider = make_provider(server, tmp_path)
    run_ingest(provider, tmp_path / "processed", competitions=(ENG,))

    assert (provider.raw_dir / CACHE_FILENAME).is_file()
    reloaded = FetchCache.load(provider.raw_dir / CACHE_FILENAME)
    assert len(reloaded) > 0


def test_a_country_file_is_checked_once_per_run_not_once_per_season(tmp_path: Path) -> None:
    """A secondary-feed file holds every season at once. Ingesting Brazil's
    fifteen seasons used to ask the server about the same file sixteen times —
    once to enumerate the seasons and once per season. Correct, and fifteen
    round trips more than the question needs."""
    body = (
        "Country,League,Season,Date,Time,Home,Away,HG,AG,Res\n"
        "Brazil,Serie A,2023,14/04/2023,20:00,Palmeiras,Santos,2,1,H\n"
        "Brazil,Serie A,2024,14/04/2024,20:00,Flamengo,Santos,1,0,H\n"
        "Brazil,Serie A,2025,14/04/2025,20:00,Palmeiras,Flamengo,0,0,D\n"
    )
    server = FakeServer({"new/BRA.csv": body})
    provider = make_provider(server, tmp_path)

    seasons = provider.candidate_seasons(BRA)
    assert seasons == ("2023", "2024", "2025")
    for season in seasons:
        provider.fetch(BRA, season)

    assert len(server.requests) == 1, f"asked {len(server.requests)} times for one file"


def test_the_memo_does_not_leak_between_runs(tmp_path: Path) -> None:
    """Per-run, not persisted. A second run must still ask, or a file that
    changed between runs would be invisible."""
    body = (
        "Country,League,Season,Date,Time,Home,Away,HG,AG,Res\n"
        "Brazil,Serie A,2024,14/04/2024,20:00,Palmeiras,Santos,2,1,H\n"
    )
    server = FakeServer({"new/BRA.csv": body})

    first = make_provider(server, tmp_path)
    first.fetch(BRA, "2024")
    first.save_cache()

    server.requests.clear()
    make_provider(server, tmp_path).fetch(BRA, "2024")
    assert len(server.requests) == 1
