"""Reading provider CSV files that are not as clean as they look.

Every behaviour here was measured against real football-data.co.uk files, not
guessed. The four problems it solves, with the evidence:

1. **Mixed encodings.** Most seasons decode as UTF-8; the 2004/05 English
   Premier League file contains byte ``0xa0`` (a cp1252 non-breaking space) and
   raises ``UnicodeDecodeError``. Two seasons (2021/22, 2024/25) additionally
   carry a UTF-8 BOM, which without ``utf-8-sig`` becomes an invisible prefix
   on the first column name — so ``df["Div"]`` raises ``KeyError`` for a column
   plainly visible in the header.

2. **Ragged rows.** English 2003/04 has a 57-column header with 32 rows of 72
   fields and 13 of 62; 1993/94 and 1994/95 have 28-column headers with rows of
   7. Both directions occur, and rows are padded or truncated to the header.

   This originally *raised* whenever a truncated field was non-empty, on the
   theory that a surplus with real data meant a shifted row. Running it over
   all forty competitions falsified that immediately: Italian Serie B 2003/04
   has a 42-column header whose last eight names are blank, and rows of 49
   fields carrying extra unnamed statistics. Every named column in those rows
   is correctly aligned — ``Atalanta 4-1 Triestina`` reads perfectly — so the
   "misalignment" was an artefact of the header being narrower than the data,
   not of the data being wrong.

   The rule is therefore: pad and truncate, and *warn* rather than raise. Only
   named columns are ever read, and a field beyond the last named column cannot
   affect one. Structural raggedness is a poor proxy for corruption; the real
   guard is semantic and lives in the adapter, which checks every row's date,
   teams, score and stated result before accepting it. A gate that goes red on
   valid data is a gate somebody switches off.

3. **Blank rows.** The 1993/94 file has 552 rows, 90 of them entirely empty.
   Dropping them leaves 462 — exactly the fixture count for a 22-team
   double round-robin, which is how the rule was confirmed rather than assumed.

4. **HTML served as CSV.** A missing season/division returns HTTP **300** with
   an HTML body, not 404. ``raise_for_status`` does not treat 3xx as an error,
   so a naive fetch saves a "Multiple Choices" page as ``E0.csv`` and the
   parser turns it into garbage rows. :func:`looks_like_csv` is the guard.

Everything is returned as strings. Type coercion belongs to the adapter, which
knows what each column means; a reader that guesses dtypes is a reader that
turns a team called "Nan" into a missing value.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Tried in order. utf-8-sig first because it also strips a BOM when present and
# behaves identically to utf-8 when not; cp1252 is the fallback that covers the
# Windows-authored files. cp1252 decodes almost any byte sequence, so it must
# come last or it would mask a genuine encoding problem by producing mojibake.
ENCODINGS: tuple[str, ...] = ("utf-8-sig", "cp1252")

# A response body must start with one of these to be accepted as CSV. Checked
# case-insensitively against the leading bytes.
_HTML_MARKERS: tuple[bytes, ...] = (b"<!doctype", b"<html", b"<?xml")


class ProviderFileError(RuntimeError):
    """A provider file is unusable: undecodable, not CSV, or misaligned."""


def looks_like_csv(payload: bytes) -> bool:
    """Return whether ``payload`` is plausibly a CSV file rather than an error page.

    Deliberately cheap and structural. It is not a validator — the adapter
    still has to parse the thing — it exists to catch the specific failure this
    provider produces, where a missing file arrives as an HTML document with a
    non-error status code.
    """
    head = payload[:512].lstrip().lower()
    if not head:
        return False
    if head.startswith(_HTML_MARKERS):
        return False
    # A CSV header is one line with at least one separator. An HTML fragment
    # that slipped past the marker check is very unlikely to satisfy this and
    # then also parse into a frame with the expected columns.
    first_line = head.split(b"\n", 1)[0]
    return b"," in first_line


def decode(payload: bytes, *, source: str = "<bytes>") -> str:
    """Decode ``payload`` using the first encoding that accepts it.

    Args:
        payload: Raw file bytes.
        source: Name used in the error message, so a failure names the file.

    Raises:
        ProviderFileError: If no candidate encoding decodes the bytes.
    """
    for encoding in ENCODINGS:
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ProviderFileError(f"{source}: not decodable as any of {ENCODINGS}")


def read_provider_csv(path: Path) -> list[dict[str, str]]:
    """Read a provider CSV into a list of string-valued row mappings.

    Blank rows are dropped; short rows are padded and long rows truncated to
    the header width. Raggedness that discards non-empty data is warned about,
    never raised on — see the module docstring for the real file that made
    raising the wrong choice.

    Returns:
        One dict per data row, keyed by header name. Values are stripped
        strings; a missing field is ``""``, never ``None``, so callers have one
        empty representation to handle rather than two.

    Raises:
        ProviderFileError: If the file is undecodable, is not CSV, or has no
            header — the three cases where there is nothing to read at all.
    """
    payload = path.read_bytes()
    if not looks_like_csv(payload):
        raise ProviderFileError(
            f"{path.name}: not a CSV file. This provider serves an HTML error "
            f"page with a non-error status code when a season is unavailable."
        )

    rows = list(csv.reader(decode(payload, source=path.name).splitlines()))
    if not rows:  # pragma: no cover - unreachable while looks_like_csv runs first
        # Unreachable today: looks_like_csv already requires a non-empty first
        # line containing a comma, so anything reaching here has at least one
        # row. Kept so that reordering these two checks fails with a sentence
        # rather than an IndexError on rows[0].
        raise ProviderFileError(f"{path.name}: empty file")

    # Header names carry their own whitespace in some files, and a trailing
    # space on a column name is invisible in every tool that would show it.
    header = [name.strip() for name in rows[0]]
    if not any(header):
        raise ProviderFileError(f"{path.name}: no header row")

    width = len(header)
    # Fields past the last *named* column are never read, so discarding them
    # cannot lose anything. Tracked to decide whether a truncation is worth
    # mentioning at all.
    last_named = max((i for i, name in enumerate(header) if name), default=-1)

    records: list[dict[str, str]] = []
    short_of_named: list[int] = []
    padded = truncated = blank = discarded = 0

    for line_number, row in enumerate(rows[1:], start=2):
        if not any(field.strip() for field in row):
            blank += 1
            continue

        if len(row) < width:
            padded += 1
            if len(row) <= last_named:
                # The row is too short to reach every named column, so padding
                # blanks a field that was meant to carry data. Collected rather
                # than logged here: one file can contain hundreds, and a
                # per-row warning would bury everything else in a full ingest.
                short_of_named.append(line_number)
            row = row + [""] * (width - len(row))
        elif len(row) > width:
            truncated += 1
            if any(field.strip() for field in row[width:]):
                discarded += 1
            row = row[:width]

        # Unnamed columns are skipped rather than collected under a shared ""
        # key, where every one but the last would be silently discarded by the
        # dict. The 1993/94 file has a 28-field header with 7 names and 21
        # blanks, so this is the common case, not an exotic one.
        records.append(
            {name: value.strip() for name, value in zip(header, row, strict=True) if name}
        )

    if blank or padded or truncated:
        logger.debug(
            "%s: %d rows (%d blank dropped, %d padded, %d truncated)",
            path.name,
            len(records),
            blank,
            padded,
            truncated,
        )
    if short_of_named:
        logger.warning(
            "%s: %d row(s) too short to fill %d named columns "
            "(lines %s%s); their tail reads as missing",
            path.name,
            len(short_of_named),
            last_named + 1,
            ", ".join(str(n) for n in short_of_named[:3]),
            "..." if len(short_of_named) > 3 else "",
        )
    if discarded:
        # Named columns are unaffected, so this is information rather than a
        # problem — but silently dropping non-empty data should still be
        # visible to anyone reading the logs.
        logger.warning(
            "%s: %d row(s) carried data past the %d-column header; " "named columns are unaffected",
            path.name,
            discarded,
            width,
        )
    return records
