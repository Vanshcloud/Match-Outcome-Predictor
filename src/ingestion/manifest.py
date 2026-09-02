"""Checksum manifests: what was downloaded, and whether it still matches.

This is the project's dataset versioning. DVC would also solve it, and is
deliberately not a dependency: DVC earns its keep when the data cannot be
re-fetched, and every byte here comes from a stable public URL. A manifest of
checksums plus ``scripts/fetch_data.py`` reproduces any past state in about
fifty lines, and adds nothing to the toolchain a contributor has to install.

What a manifest actually buys:

- **Reproducibility.** "The model was trained on *this* data" becomes a
  checkable claim rather than a hopeful one.
- **Corruption detection.** A truncated or half-rewritten file is caught before
  it becomes a season with eight missing match weeks.
- **Change awareness.** The provider revises files — a corrected scoreline, a
  late-added referee. A changed checksum on a season believed settled is worth
  knowing about.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

MANIFEST_VERSION = 1

# Hashed in blocks rather than with `path.read_bytes()`. The country files are
# small today, but a manifest utility that loads whole files into memory is one
# that fails on the first large one, and the streaming version is no longer.
_CHUNK_SIZE = 1 << 20


def checksum(path: Path) -> str:
    """Return the SHA-256 of ``path`` as hex."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FileEntry:
    """One recorded file."""

    path: str
    """Relative to the manifest's root, and stored with forward slashes so a
    manifest written on Windows verifies on Linux."""

    sha256: str
    bytes: int

    def to_json(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


def build_entries(root: Path, paths: Iterable[Path]) -> tuple[FileEntry, ...]:
    """Checksum ``paths``, recording each relative to ``root``."""
    entries = [
        FileEntry(
            path=path.relative_to(root).as_posix(),
            sha256=checksum(path),
            bytes=path.stat().st_size,
        )
        for path in sorted(paths)
    ]
    return tuple(entries)


def write_manifest(
    destination: Path,
    root: Path,
    paths: Iterable[Path],
    *,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write a manifest describing ``paths`` and return it.

    Args:
        destination: Where the JSON is written.
        root: Base directory paths are recorded relative to.
        paths: Files to record.
        extra: Provenance to embed — row counts, competitions, the provider.
    """
    entries = build_entries(root, paths)
    manifest: dict[str, object] = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "file_count": len(entries),
        "total_bytes": sum(entry.bytes for entry in entries),
        "files": [entry.to_json() for entry in entries],
        **(extra or {}),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys so two manifests over the same data diff cleanly; only
    # created_at should ever move.
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("manifest: %d files, %.1f MB", len(entries), manifest["total_bytes"] / 1e6)  # type: ignore[operator]
    return manifest


def read_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded: dict[str, object] = json.load(handle)
    return loaded


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """What verifying a manifest found."""

    missing: tuple[str, ...]
    changed: tuple[str, ...]
    unchanged: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.changed

    def summary(self) -> str:
        return (
            f"{len(self.unchanged)} unchanged, "
            f"{len(self.changed)} changed, {len(self.missing)} missing"
        )


def verify_manifest(manifest: dict[str, object], root: Path) -> VerificationResult:
    """Re-checksum every recorded file and report the differences.

    Reports rather than raises. A changed file is not automatically an error —
    the provider does revise history — and the caller is better placed to
    decide whether to refuse, warn, or re-ingest.
    """
    missing: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []

    files: list[dict[str, object]] = manifest.get("files", [])  # type: ignore[assignment]
    for entry in files:
        relative = str(entry["path"])
        target = root / relative
        if not target.is_file():
            missing.append(relative)
        elif checksum(target) != entry["sha256"]:
            changed.append(relative)
        else:
            unchanged.append(relative)

    return VerificationResult(tuple(missing), tuple(changed), tuple(unchanged))
