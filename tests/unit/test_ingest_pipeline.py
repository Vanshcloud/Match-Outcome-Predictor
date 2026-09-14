"""The ingestion pipeline coordinates; it must not compute.

The properties worth protecting are the ones later stages silently depend
on: chronological ordering, deduplication, and that a missing season is a
normal outcome rather than a failure that aborts the run.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.base import CANONICAL_SCHEMA, SeasonUnavailableError
from src.ingestion.manifest import read_manifest, verify_manifest
from src.ingestion.registry import Competition, Feed, Registry
from src.pipelines.ingest import MANIFEST_FILENAME, MATCHES_FILENAME, run_ingest
from tests.factories import canonical_frame

TODAY = date(2026, 9, 3)

ENG = Competition(
    id="ENG_1", country="England", name="Premier League", tier=1, feed=Feed.MAIN, code="E0"
)
SCO = Competition(
    id="SCO_4", country="Scotland", name="League Two", tier=4, feed=Feed.MAIN, code="SC3"
)
REGISTRY = Registry(competitions=(ENG, SCO))


def frame(rows: list[tuple[str, str, str, int, int]], competition: str = "ENG_1") -> pd.DataFrame:
    """Build a canonical frame from (season, date, home, hg, ag) tuples."""
    records = []
    for season, day, home, home_goals, away_goals in rows:
        records.append(
            {
                "match_id": f"{competition}-{season}-{day}-{home}",
                "provider": "stub",
                "competition_id": competition,
                "country": "England",
                "competition": "Premier League",
                "tier": 1,
                "season": season,
                "date": pd.Timestamp(day),
                "home_team": home,
                "away_team": "Other",
                "home_team_id": f"eng:{home.lower()}",
                "away_team_id": "eng:other",
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": (
                    "H" if home_goals > away_goals else ("A" if home_goals < away_goals else "D")
                ),
            }
        )
    return canonical_frame(records)


class StubProvider:
    """A provider with a scripted answer per season."""

    name = "stub"

    def __init__(
        self,
        seasons: dict[str, pd.DataFrame],
        competitions: tuple[Competition, ...],
        raw_dir: Path | None = None,
    ):
        self.seasons = seasons
        self._competitions = competitions
        # The pipeline validates what it wrote, and the referential checks need
        # the registry the table is supposed to agree with. A stub that omits it
        # is not standing in for the provider the pipeline actually calls.
        self.registry = Registry(competitions=competitions)
        self.calls: list[tuple[str, str]] = []
        # The pipeline checksums the raw cache and persists what the run
        # learned; a stub has to expose both or it is not testing the pipeline
        # that actually runs.
        self.raw_dir = raw_dir or Path("/nonexistent-raw")
        self.saved = 0

    def save_cache(self) -> None:
        self.saved += 1

    def competitions(self) -> tuple[Competition, ...]:
        return self._competitions

    def candidate_seasons(self, competition: Competition) -> tuple[str, ...]:
        return tuple(sorted({s for (c, s) in self.seasons if c == competition.id}))  # type: ignore[misc]

    def fetch(self, competition: Competition, season: str) -> pd.DataFrame:
        self.calls.append((competition.id, season))
        key = (competition.id, season)
        if key not in self.seasons:
            raise SeasonUnavailableError(f"{competition.id} {season}")
        return self.seasons[key]  # type: ignore[index]


def make_provider(mapping: dict[tuple[str, str], pd.DataFrame], competitions=(ENG,)):  # type: ignore[no-untyped-def]
    provider = StubProvider({}, competitions)
    provider.seasons = mapping  # type: ignore[assignment]
    return provider


# ---- happy path -------------------------------------------------------------


def test_seasons_are_combined_into_one_table(tmp_path: Path) -> None:
    provider = make_provider(
        {
            ("ENG_1", "2023-24"): frame([("2023-24", "2024-01-02", "A", 1, 0)]),
            ("ENG_1", "2024-25"): frame([("2024-25", "2024-08-16", "B", 2, 2)]),
        }
    )
    report = run_ingest(provider, tmp_path)  # type: ignore[arg-type]

    assert report.matches == 2
    assert report.seasons_ingested == 2
    assert report.competitions_ingested == 1
    assert report.per_competition == {"ENG_1": 2}


def test_the_output_is_chronological(tmp_path: Path) -> None:
    """Every temporal split and rolling feature downstream assumes this.
    Relying on an incidental ordering is how a "time-aware" split quietly
    stops being one."""
    provider = make_provider(
        {
            ("ENG_1", "2024-25"): frame([("2024-25", "2024-08-16", "B", 1, 0)]),
            ("ENG_1", "2023-24"): frame([("2023-24", "2024-01-02", "A", 1, 0)]),
        }
    )
    run_ingest(provider, tmp_path)  # type: ignore[arg-type]
    written = pd.read_parquet(tmp_path / MATCHES_FILENAME)
    assert written["date"].is_monotonic_increasing


def test_the_written_table_keeps_the_canonical_schema(tmp_path: Path) -> None:
    """Parquet round-tripping is where a nullable Int16 quietly becomes a
    float64 if the dtypes are not re-asserted."""
    provider = make_provider({("ENG_1", "2024-25"): frame([("2024-25", "2024-08-16", "A", 1, 0)])})
    run_ingest(provider, tmp_path)  # type: ignore[arg-type]
    written = pd.read_parquet(tmp_path / MATCHES_FILENAME)
    assert list(written.columns) == list(CANONICAL_SCHEMA)
    assert {c: str(d) for c, d in written.dtypes.items()} == CANONICAL_SCHEMA


# ---- duplicates and gaps ----------------------------------------------------


def test_duplicate_match_ids_are_dropped(tmp_path: Path) -> None:
    """A duplicate means the same fixture arrived twice — a competition listed
    under two registry entries, or a re-run appending instead of replacing."""
    duplicated = frame([("2024-25", "2024-08-16", "A", 1, 0)])
    provider = make_provider({("ENG_1", "2024-25"): duplicated, ("ENG_1", "2023-24"): duplicated})
    assert run_ingest(provider, tmp_path).matches == 1  # type: ignore[arg-type]


def test_a_missing_season_does_not_abort_the_run(tmp_path: Path) -> None:
    """Most (division, season) pairs do not exist: leagues are founded, folded
    and restructured. Treating that as an error would make a full ingest
    impossible."""
    provider = StubProvider(
        {("ENG_1", "2024-25"): frame([("2024-25", "2024-08-16", "A", 1, 0)])}, (ENG,)
    )
    provider.candidate_seasons = lambda _competition: ("1993-94", "2024-25")  # type: ignore[assignment,method-assign]

    report = run_ingest(provider, tmp_path)  # type: ignore[arg-type]
    assert report.matches == 1
    assert report.seasons_unavailable == 1


def test_a_competition_with_no_data_is_reported_not_fatal(tmp_path: Path) -> None:
    provider = StubProvider(
        {("ENG_1", "2024-25"): frame([("2024-25", "2024-08-16", "A", 1, 0)])}, (ENG, SCO)
    )
    provider.candidate_seasons = lambda _competition: ("2024-25",)  # type: ignore[assignment,method-assign]

    report = run_ingest(provider, tmp_path)  # type: ignore[arg-type]
    assert report.competitions_empty == ("SCO_4",)
    assert report.competitions_ingested == 1


def test_an_entirely_empty_run_returns_a_report_rather_than_raising(tmp_path: Path) -> None:
    """The caller decides whether empty is a failure. For a single-competition
    run against an offline provider it may not be."""
    provider = StubProvider({}, (ENG,))
    report = run_ingest(provider, tmp_path)  # type: ignore[arg-type]
    assert report.matches == 0
    assert report.output is None
    assert not (tmp_path / MATCHES_FILENAME).exists()


def test_a_subset_can_be_ingested(tmp_path: Path) -> None:
    provider = StubProvider(
        {("ENG_1", "2024-25"): frame([("2024-25", "2024-08-16", "A", 1, 0)])}, (ENG, SCO)
    )
    provider.candidate_seasons = lambda _competition: ("2024-25",)  # type: ignore[assignment,method-assign]
    run_ingest(provider, tmp_path, competitions=(ENG,))  # type: ignore[arg-type]
    assert [c for c, _ in provider.calls] == ["ENG_1"]


# ---- the manifest -----------------------------------------------------------


def test_a_manifest_is_written_and_verifies(tmp_path: Path) -> None:
    provider = make_provider({("ENG_1", "2024-25"): frame([("2024-25", "2024-08-16", "A", 1, 0)])})
    run_ingest(provider, tmp_path)  # type: ignore[arg-type]

    manifest = read_manifest(tmp_path / MANIFEST_FILENAME)
    assert manifest["matches"] == 1
    assert manifest["provider"] == "stub"
    assert verify_manifest(manifest, tmp_path).ok


def test_the_manifest_records_the_date_range(tmp_path: Path) -> None:
    """So "the model was trained on data up to X" is a checkable claim."""
    provider = make_provider(
        {
            ("ENG_1", "2023-24"): frame([("2023-24", "2024-01-02", "A", 1, 0)]),
            ("ENG_1", "2024-25"): frame([("2024-25", "2024-08-16", "B", 1, 0)]),
        }
    )
    run_ingest(provider, tmp_path)  # type: ignore[arg-type]
    manifest = read_manifest(tmp_path / MANIFEST_FILENAME)
    assert manifest["date_range"] == ["2024-01-02", "2024-08-16"]


def test_the_manifest_detects_a_tampered_output(tmp_path: Path) -> None:
    provider = make_provider({("ENG_1", "2024-25"): frame([("2024-25", "2024-08-16", "A", 1, 0)])})
    run_ingest(provider, tmp_path)  # type: ignore[arg-type]
    manifest = read_manifest(tmp_path / MANIFEST_FILENAME)

    (tmp_path / MATCHES_FILENAME).write_bytes(b"corrupted")
    assert verify_manifest(manifest, tmp_path).changed == (MATCHES_FILENAME,)


@pytest.mark.parametrize("attribute", ["matches", "seasons_ingested", "competitions_ingested"])
def test_the_report_summarises_itself(tmp_path: Path, attribute: str) -> None:
    provider = make_provider({("ENG_1", "2024-25"): frame([("2024-25", "2024-08-16", "A", 1, 0)])})
    report = run_ingest(provider, tmp_path)  # type: ignore[arg-type]
    assert getattr(report, attribute) == 1
    assert "1 matches" in report.summary() or "1," in report.summary()


# ---- an unreadable file must not cost the whole run -------------------------


def test_an_unreadable_season_is_skipped_not_fatal(tmp_path: Path) -> None:
    """The regression this exists for. A full ingest makes roughly seven
    hundred fetches; the first run over all forty competitions died on a single
    malformed Italian Serie B file and discarded every competition after it in
    the loop. One bad file is one season's problem.
    """
    from src.ingestion.csv_reader import ProviderFileError

    good = frame([("2024-25", "2024-08-16", "A", 1, 0)])

    class PartlyBroken(StubProvider):
        def fetch(self, _competition: Competition, season: str) -> pd.DataFrame:
            if season == "2023-24":
                raise ProviderFileError("I2.csv: undecodable")
            return good

    provider = PartlyBroken({}, (ENG,))
    provider.candidate_seasons = lambda _competition: ("2023-24", "2024-25")  # type: ignore[assignment,method-assign]

    report = run_ingest(provider, tmp_path)  # type: ignore[arg-type]

    assert report.matches == 1, "the readable season must still be ingested"
    assert report.seasons_failed == ("ENG_1 2023-24",)
    assert report.seasons_ingested == 1


def test_an_unreadable_season_is_distinct_from_an_unpublished_one(tmp_path: Path) -> None:
    """A season the provider never published is routine; a file that exists and
    cannot be parsed is a defect. Counting them together would bury the second
    in the first — most (division, season) pairs legitimately do not exist."""
    from src.ingestion.csv_reader import ProviderFileError

    class Mixed(StubProvider):
        def fetch(self, _competition: Competition, season: str) -> pd.DataFrame:
            if season == "1993-94":
                raise SeasonUnavailableError("not published")
            if season == "2023-24":
                raise ProviderFileError("corrupt")
            return frame([("2024-25", "2024-08-16", "A", 1, 0)])

    provider = Mixed({}, (ENG,))
    provider.candidate_seasons = lambda _c: ("1993-94", "2023-24", "2024-25")  # type: ignore[assignment,method-assign]

    report = run_ingest(provider, tmp_path)  # type: ignore[arg-type]

    assert report.seasons_unavailable == 1
    assert report.seasons_failed == ("ENG_1 2023-24",)
    assert "1 unreadable" in report.summary()


# ---- validation -------------------------------------------------------------


def test_the_run_validates_what_it_wrote(tmp_path: Path) -> None:
    """The run that produced a broken table should be the run that says so.
    The frame is already in memory and the checks are a second's work over it,
    so the alternative is finding out at training time."""
    provider = make_provider({("ENG_1", "2024-25"): frame([("2024-25", "2024-08-16", "A", 1, 0)])})
    report = run_ingest(provider, tmp_path)  # type: ignore[arg-type]

    assert report.validation is not None
    assert report.validation.ok
    assert report.validation.rows == 1
    assert "validation" in report.summary()


def test_validation_reports_without_gating(tmp_path: Path) -> None:
    """A table you can inspect is more useful than one the pipeline refused to
    save, so a failed check is reported and the Parquet is still written."""
    broken = frame([("2024-25", "2024-08-16", "A", 1, 0)])
    broken.loc[0, "result"] = "A"  # disagrees with 1-0
    provider = make_provider({("ENG_1", "2024-25"): broken})

    report = run_ingest(provider, tmp_path)  # type: ignore[arg-type]

    assert report.validation is not None
    assert not report.validation.ok
    assert (tmp_path / MATCHES_FILENAME).is_file()


def test_a_run_that_ingested_nothing_has_no_validation_report(tmp_path: Path) -> None:
    provider = make_provider({})
    report = run_ingest(provider, tmp_path)  # type: ignore[arg-type]
    assert report.validation is None
