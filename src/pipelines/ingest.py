"""Ingest every configured competition into one canonical match table.

A full run makes roughly 700 requests against the primary feed, most of which
are misses — the provider publishes no index, so the only way to learn that
Greece has no 1998/99 file is to ask. Two things keep that acceptable: settled
seasons are cached permanently, so the misses are paid once, and a miss is
routine rather than an error.

The output is a single Parquet file. Partitioning by competition would help a
serving layer, and is deliberately not done yet: the full 303,517 rows are 10.7 MB of Parquet,
which pandas reads in well under a second. Partition when that stops being
true, not in anticipation of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.ingestion.base import CANONICAL_SCHEMA, SeasonUnavailableError
from src.ingestion.csv_reader import ProviderFileError
from src.ingestion.football_data import FootballDataProvider
from src.ingestion.manifest import write_manifest
from src.ingestion.registry import Competition
from src.utils.logging import get_logger
from src.validation.matches import match_checks
from src.validation.report import Outcome, ValidationReport, run_checks

logger = get_logger(__name__)

MATCHES_FILENAME = "matches.parquet"
MANIFEST_FILENAME = "manifest.json"
RAW_MANIFEST_FILENAME = "manifest.json"


@dataclass
class IngestReport:
    """What one ingest run did. Returned rather than printed, so a test can
    assert on it and the CLI can render it."""

    matches: int = 0
    competitions_ingested: int = 0
    competitions_empty: tuple[str, ...] = ()
    seasons_ingested: int = 0
    seasons_unavailable: int = 0
    seasons_failed: tuple[str, ...] = ()
    """Seasons whose file was unreadable. Distinct from `seasons_unavailable`:
    that one is a season the provider never published, which is routine, while
    this is a file that exists and could not be parsed, which is a defect worth
    someone's attention."""
    per_competition: dict[str, int] = field(default_factory=dict)
    output: Path | None = None
    raw_manifest: Path | None = None
    validation: ValidationReport | None = None
    """The check suite run over the table that was just written.

    Run here rather than left to a separate command because the frame is
    already in memory and the checks are a second's work over it — so the run
    that produced a broken table is the run that says so, instead of the
    developer finding out at training time. It does not gate: the file is
    written either way, because a table you can inspect is more useful than one
    the pipeline refused to save."""

    def summary(self) -> str:
        summary = (
            f"{self.matches:,} matches across {self.competitions_ingested} competitions "
            f"and {self.seasons_ingested} seasons "
            f"({self.seasons_unavailable} season files not published)"
        )
        if self.seasons_failed:
            summary += f"; {len(self.seasons_failed)} unreadable"
        if self.validation is not None:
            summary += f"; validation {self.validation.summary()}"
        return summary


def ingest_competition(
    provider: FootballDataProvider,
    competition: Competition,
    report: IngestReport,
) -> list[pd.DataFrame]:
    """Fetch every available season of one competition."""
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for season in provider.candidate_seasons(competition):
        try:
            frames.append(provider.fetch(competition, season))
            report.seasons_ingested += 1
        except SeasonUnavailableError:
            # The expected outcome for most (division, season) pairs. Logged at
            # debug so a full run does not print six hundred lines saying that
            # a league which did not exist in 1994 did not exist in 1994.
            report.seasons_unavailable += 1
            logger.debug("unavailable: %s %s", competition.id, season)
        except ProviderFileError as error:
            # An unreadable file is one competition-season's problem, not the
            # run's. Aborting here would mean a single malformed file out of
            # roughly seven hundred discards every competition after it in the
            # loop — which is exactly what happened the first time this ran
            # over all forty, and cost the whole ingest.
            failed.append(f"{competition.id} {season}")
            logger.error("unreadable: %s %s — %s", competition.id, season, error)
    report.seasons_failed = report.seasons_failed + tuple(failed)
    return frames


def run_ingest(
    provider: FootballDataProvider,
    processed_dir: Path,
    *,
    competitions: tuple[Competition, ...] | None = None,
) -> IngestReport:
    """Ingest competitions into ``processed_dir`` and write a manifest.

    Args:
        provider: The source adapter.
        processed_dir: Destination for the canonical table and its manifest.
        competitions: Subset to ingest. Defaults to every registered one.

    Returns:
        An :class:`IngestReport`. A run that finds nothing returns a report
        with zero matches rather than raising — the caller decides whether an
        empty result is a failure, and for a single-competition run against an
        offline provider it may not be.
    """
    targets = competitions if competitions is not None else provider.competitions()
    frames: list[pd.DataFrame] = []
    report = IngestReport()
    empty: list[str] = []

    for index, competition in enumerate(targets, start=1):
        logger.info("[%d/%d] %s — %s", index, len(targets), competition.id, competition.name)
        collected = ingest_competition(provider, competition, report)
        if not collected:
            empty.append(competition.id)
            logger.warning("%s: no seasons available", competition.id)
            continue
        rows = sum(len(frame) for frame in collected)
        report.per_competition[competition.id] = rows
        report.competitions_ingested += 1
        frames.extend(collected)

    report.competitions_empty = tuple(empty)

    # Persisted before the early return: a run that ingested nothing still
    # learned which files are missing and which are unchanged, and throwing
    # that away would make the next run pay for it again.
    provider.save_cache()
    report.raw_manifest = _write_raw_manifest(provider.raw_dir)

    if not frames:
        logger.warning("nothing ingested")
        return report

    matches = _combine(frames)
    report.matches = len(matches)
    report.validation = run_checks(matches, match_checks(provider.registry))
    for failure in report.validation.of(Outcome.FAILED):
        # Warnings included: a validation failure nobody sees is a validation
        # suite nobody has. The severity decides whether it stops a caller, not
        # whether it is worth printing.
        logger.warning("validation: %s — %s", failure.name, failure.message)

    processed_dir.mkdir(parents=True, exist_ok=True)
    output = processed_dir / MATCHES_FILENAME
    matches.to_parquet(output, index=False)
    report.output = output

    write_manifest(
        processed_dir / MANIFEST_FILENAME,
        processed_dir,
        [output],
        extra={
            "provider": provider.name,
            "matches": report.matches,
            "competitions": report.competitions_ingested,
            "seasons": report.seasons_ingested,
            "per_competition": report.per_competition,
            "date_range": [
                matches["date"].min().strftime("%Y-%m-%d"),
                matches["date"].max().strftime("%Y-%m-%d"),
            ],
        },
    )
    logger.info("wrote %s — %s", output.name, report.summary())
    return report


def _write_raw_manifest(raw_dir: Path) -> Path | None:
    """Checksum every cached provider file.

    This is the historical record: it says exactly which bytes produced a given
    canonical table, so "the model was trained on this data" is checkable
    rather than hopeful, and a file the provider silently revises shows up as a
    changed checksum rather than as a quietly different model.

    Separate from the processed-table manifest because they answer different
    questions — one is provenance of the *inputs*, the other of the *output* —
    and because the raw manifest survives a rebuild of the canonical table.
    """
    if not raw_dir.is_dir():
        return None
    files = sorted(path for path in raw_dir.rglob("*.csv") if path.is_file())
    if not files:
        return None
    destination = raw_dir / RAW_MANIFEST_FILENAME
    write_manifest(destination, raw_dir, files, extra={"kind": "raw-provider-files"})
    return destination


def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate, deduplicate and order the canonical table.

    Sorted by date then competition then match_id: a deterministic order makes
    the Parquet byte-reproducible for the manifest, and chronological order is
    what every temporal split and rolling feature downstream assumes. Relying
    on an incidental ordering is how a "time-aware" split quietly stops being
    one.
    """
    combined = pd.concat(frames, ignore_index=True)

    # match_id is a hash of the natural key, so a duplicate means the same
    # fixture was ingested twice — a competition listed under two registry
    # entries, or a re-run appending instead of replacing. Dropping is correct
    # and the count is worth knowing about.
    before = len(combined)
    combined = combined.drop_duplicates(subset="match_id", keep="first")
    if len(combined) != before:
        logger.warning("dropped %d duplicate match ids", before - len(combined))

    combined = combined.sort_values(
        ["date", "competition_id", "match_id"], kind="stable"
    ).reset_index(drop=True)
    # Re-assert the schema: concat of many frames can widen a nullable dtype if
    # any one of them arrived empty.
    return combined.astype(CANONICAL_SCHEMA)
