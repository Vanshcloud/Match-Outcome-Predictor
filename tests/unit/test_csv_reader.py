"""The CSV reader exists because real provider files are malformed in four
specific ways. Each test here names the file the behaviour was measured on, so
a future change can tell which real-world case it would break.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.csv_reader import (
    ProviderFileError,
    decode,
    looks_like_csv,
    read_provider_csv,
)


def write(tmp_path: Path, name: str, payload: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


# ---- HTML served as CSV -----------------------------------------------------
#
# The provider answers a missing season with HTTP 300 and a "Multiple Choices"
# page. raise_for_status does not treat 3xx as an error, so this is the only
# thing standing between an error page and the parser.


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"<!DOCTYPE HTML PUBLIC><html><head>", id="doctype"),
        pytest.param(b"<html><body>Multiple Choices</body></html>", id="html"),
        pytest.param(b"  \n  <!doctype html>", id="leading-whitespace"),
        pytest.param(b"<?xml version='1.0'?><rss/>", id="xml"),
        pytest.param(b"", id="empty"),
        pytest.param(b"no separators here at all", id="no-comma"),
    ],
)
def test_non_csv_payloads_are_rejected(payload: bytes) -> None:
    assert not looks_like_csv(payload)


def test_a_real_header_is_accepted() -> None:
    assert looks_like_csv(b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,16/08/2024,A,B,1,0,H\n")


def test_reading_an_html_page_names_the_real_problem(tmp_path: Path) -> None:
    path = write(tmp_path, "E0.csv", b"<!DOCTYPE HTML><html>300 Multiple Choices</html>")
    with pytest.raises(ProviderFileError, match="not a CSV"):
        read_provider_csv(path)


# ---- encodings --------------------------------------------------------------


def test_utf8_is_read() -> None:
    assert decode(b"Div,Date\nE0,x\n").startswith("Div")


def test_a_bom_is_stripped() -> None:
    """2021/22 and 2024/25 carry a UTF-8 BOM. Without utf-8-sig it becomes an
    invisible prefix on the first column name, and df["Div"] raises KeyError
    for a column plainly visible in the header."""
    decoded = decode(b"\xef\xbb\xbfDiv,Date\n")
    assert decoded.startswith("Div"), "BOM survived into the first column name"


def test_cp1252_is_read() -> None:
    """The 2004/05 English file contains byte 0xa0, a cp1252 non-breaking
    space, and raises UnicodeDecodeError under UTF-8."""
    assert "\xa0" in decode(b"Div,Ref\nE0,A\xa0B\n")


def test_undecodable_bytes_name_the_file() -> None:
    # A lone continuation byte is invalid UTF-8; cp1252 maps every byte except
    # a handful, so this uses one of the unmapped ones.
    with pytest.raises(ProviderFileError, match="E0.csv"):
        decode(b"\x81\x8d\x8f", source="E0.csv")


# ---- ragged rows ------------------------------------------------------------


def test_short_rows_are_padded(tmp_path: Path) -> None:
    """1993/94 and 1994/95 have 28-column headers with 7-field rows."""
    path = write(tmp_path, "a.csv", b"A,B,C\n1,2\n")
    assert read_provider_csv(path) == [{"A": "1", "B": "2", "C": ""}]


def test_long_rows_are_truncated_when_the_surplus_is_empty(tmp_path: Path) -> None:
    """2003/04 has a 57-column header with rows of 72 and 62 fields. Every
    surplus field observed was empty — stray commas, not shifted data."""
    path = write(tmp_path, "a.csv", b"A,B\n1,2,,,\n")
    assert read_provider_csv(path) == [{"A": "1", "B": "2"}]


def test_data_past_the_header_is_dropped_with_a_warning(tmp_path: Path) -> None:
    """This used to raise, and running over all forty competitions proved that
    wrong. Italian Serie B 2003/04 has a 42-column header whose last eight
    names are blank, and rows of 49 fields carrying extra unnamed statistics —
    every named column correctly aligned. Raising there discarded the rest of
    the ingest for data that was perfectly fine.

    Fields beyond the last named column are never read, so dropping them cannot
    corrupt a row. It is still worth saying out loud.
    """
    path = write(tmp_path, "a.csv", b"A,B\n1,2,SURPRISE\n")
    assert read_provider_csv(path) == [{"A": "1", "B": "2"}]


def test_named_columns_survive_a_wider_row(tmp_path: Path) -> None:
    """The Serie B shape, reduced: a header narrower than its rows, with
    unnamed trailing slots. The named fields must read exactly."""
    path = write(
        tmp_path,
        "I2.csv",
        b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,,,\n"
        b"I2,12/10/03,Atalanta,Triestina,4,1,H,,,,,3,0,,2,0\n",
    )
    row = read_provider_csv(path)[0]
    assert row["HomeTeam"] == "Atalanta"
    assert row["AwayTeam"] == "Triestina"
    assert (row["FTHG"], row["FTAG"], row["FTR"]) == ("4", "1", "H")


def test_a_row_too_short_for_its_named_columns_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Padding here blanks a field that was meant to carry data, so the row is
    kept but the loss is reported. The adapter's own validation then decides
    whether what survived is still a usable match."""
    import logging

    path = write(tmp_path, "a.csv", b"A,B,C\n1\n")
    with caplog.at_level(logging.WARNING, logger="src.ingestion.csv_reader"):
        assert read_provider_csv(path) == [{"A": "1", "B": "", "C": ""}]
    assert "too short" in caplog.text


# ---- blank rows and headers -------------------------------------------------


def test_blank_rows_are_dropped(tmp_path: Path) -> None:
    """1993/94 ships 552 rows of which 90 are empty; dropping them leaves 462,
    exactly a 22-team double round-robin."""
    path = write(tmp_path, "a.csv", b"A,B\n1,2\n,\n   ,  \n3,4\n")
    assert read_provider_csv(path) == [{"A": "1", "B": "2"}, {"A": "3", "B": "4"}]


def test_unnamed_columns_are_skipped(tmp_path: Path) -> None:
    """The 1993/94 header has 7 names and 21 blanks. Collected under a shared
    "" key, every one but the last would be silently discarded by the dict."""
    path = write(tmp_path, "a.csv", b"A,,B,\n1,x,2,y\n")
    assert read_provider_csv(path) == [{"A": "1", "B": "2"}]


def test_header_whitespace_is_stripped(tmp_path: Path) -> None:
    """A trailing space on a column name is invisible in every tool that would
    show it, and turns every lookup into a KeyError."""
    path = write(tmp_path, "a.csv", b" A , B \n1,2\n")
    assert read_provider_csv(path) == [{"A": "1", "B": "2"}]


def test_values_are_stripped(tmp_path: Path) -> None:
    path = write(tmp_path, "a.csv", b"A,B\n 1 , 2 \n")
    assert read_provider_csv(path) == [{"A": "1", "B": "2"}]


def test_a_headerless_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProviderFileError, match="no header"):
        read_provider_csv(write(tmp_path, "a.csv", b",,,\n1,2,3\n"))


def test_a_completely_empty_file_is_refused(tmp_path: Path) -> None:
    """Distinct from an HTML page: nothing at all arrived. A zero-byte file on
    disk is what an interrupted write leaves behind."""
    path = write(tmp_path, "a.csv", b"\n")
    with pytest.raises(ProviderFileError):
        read_provider_csv(path)
