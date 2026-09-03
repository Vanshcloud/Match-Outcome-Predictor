"""The reporting machinery, tested apart from the checks that use it.

The distinction worth protecting here is three-way. A suite that reports a
skipped check as passed is worse than useless — it is a green tick over a
question nobody asked — and it is the mistake this module exists to avoid.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.validation.report import (
    Check,
    CheckResult,
    Outcome,
    Severity,
    ValidationReport,
    check,
    names,
    run_checks,
)

FRAME = pd.DataFrame({"n": range(10)})


@check("always true")
def _passes(_matches: pd.DataFrame) -> str | None:
    return None


@check("always false")
def _fails(_matches: pd.DataFrame) -> str | None:
    return "it did not hold"


@check("a caveat", severity=Severity.WARNING)
def _warns(_matches: pd.DataFrame) -> str | None:
    return "worth a look"


@check("needs a population", min_rows=1_000)
def _needs_rows(_matches: pd.DataFrame) -> str | None:  # pragma: no cover - never reached
    raise AssertionError("a skipped check must not run its body")


def test_a_passing_check_reports_passed() -> None:
    result = _passes(FRAME)
    assert result.outcome is Outcome.PASSED
    assert result.message == "ok"
    assert not result.failed


def test_a_failing_check_carries_its_message() -> None:
    result = _fails(FRAME)
    assert result.outcome is Outcome.FAILED
    assert result.message == "it did not hold"
    assert result.blocking


def test_a_failing_warning_is_not_blocking() -> None:
    """The whole point of the severity field: thin data is a caveat, a score
    that disagrees with its own result is a defect, and a report that cannot
    tell them apart gets ignored."""
    result = _warns(FRAME)
    assert result.failed
    assert not result.blocking


def test_a_check_below_min_rows_is_skipped_without_running() -> None:
    """Not merely skipped — the body must not execute. A distribution check run
    over forty rows can pass by luck, and that is the false green this exists
    to prevent."""
    result = _needs_rows(FRAME)
    assert result.outcome is Outcome.SKIPPED
    assert "needs 1,000" in result.message


def test_a_skipped_check_is_not_a_passed_check() -> None:
    report = run_checks(FRAME, [_passes, _needs_rows])
    assert len(report.of(Outcome.PASSED)) == 1
    assert len(report.of(Outcome.SKIPPED)) == 1
    assert report.ok


def test_a_report_is_ok_when_only_warnings_failed() -> None:
    report = run_checks(FRAME, [_passes, _warns])
    assert report.ok
    assert report.blocking == ()


def test_a_report_is_not_ok_when_an_error_failed() -> None:
    report = run_checks(FRAME, [_passes, _fails, _warns])
    assert not report.ok
    assert [result.name for result in report.blocking] == ["always false"]


def test_every_check_runs_even_after_one_fails() -> None:
    """Stopping at the first failure makes fixing a batch an iterative guessing
    game, and the expensive part — reading the table — already happened."""
    report = run_checks(FRAME, [_fails, _passes, _warns])
    assert len(report.results) == 3


def test_the_summary_counts_every_outcome() -> None:
    report = run_checks(FRAME, [_passes, _fails, _needs_rows])
    assert report.summary() == "1 passed, 1 failed, 1 skipped over 10 matches"


def test_markdown_puts_failures_first() -> None:
    """A report whose one broken check sits nineteenth is a report whose top
    line is all anyone reads."""
    report = run_checks(FRAME, [_passes, _passes, _fails])
    body = report.to_markdown().splitlines()[2:]
    assert body[0].startswith("| FAIL |")
    assert all(line.startswith("| pass |") for line in body[1:])


def test_the_report_serialises_to_json() -> None:
    report = run_checks(FRAME, [_passes, _fails])
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["ok"] is False
    assert payload["rows"] == 10
    assert payload["results"][1] == {
        "name": "always false",
        "outcome": "failed",
        "severity": "error",
        "message": "it did not hold",
    }


def test_names_do_not_execute_the_checks() -> None:
    """Introspection has to be free of the frame. Running a check to learn its
    name means the suite cannot be listed without data that satisfies it."""
    assert names([_passes, _needs_rows]) == ("always true", "needs a population")


def test_the_decorator_produces_a_check_object() -> None:
    assert isinstance(_passes, Check)
    assert _passes.severity is Severity.ERROR
    assert _passes.min_rows == 0


def test_a_check_result_is_immutable() -> None:
    result = _passes(FRAME)
    with pytest.raises(AttributeError):
        result.name = "renamed"  # type: ignore[misc]


def test_an_empty_report_is_ok() -> None:
    report = ValidationReport(results=(), rows=0)
    assert report.ok
    assert report.summary() == "0 passed, 0 failed, 0 skipped over 0 matches"


def test_a_result_renders_its_own_fields() -> None:
    result = CheckResult(name="n", outcome=Outcome.SKIPPED, severity=Severity.WARNING, message="m")
    assert result.to_dict()["outcome"] == "skipped"
    assert not result.failed
