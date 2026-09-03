"""How a check is written, and what a run of them produces.

Each check answers one question and returns either ``None`` (it passed) or a
message saying what it found. The :func:`check` decorator turns that into a
:class:`CheckResult` carrying the name and severity, so a check body contains
only the question — twenty checks, no twenty copies of the same four lines of
bookkeeping.

**Three outcomes, not two.** A check can also be *skipped*, and conflating that
with passing would be a lie the report tells: "the home-win rate is plausible"
means nothing over forty rows, and a suite that reports it green on a
single-competition run is a suite that will report green when it matters too.
Below :paramref:`check.min_rows` the result is explicitly SKIPPED and says so.

**Severity separates two different failures.** A result that disagrees with its
own scoreline is a defect — something is wrong with the code. A competition
with only three hundred matches is a caveat — the data is thin, which is worth
printing and is not a reason to refuse. Only ERROR failures make
:attr:`ValidationReport.ok` false.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class Severity(StrEnum):
    """How much a failure of this check matters."""

    ERROR = "error"
    """The data cannot be used as it stands. Fails the run."""

    WARNING = "warning"
    """Worth a human's attention; not a reason to stop."""


class Outcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    """Not applicable to this data — almost always too few rows to be meaningful."""


@dataclass(frozen=True, slots=True)
class CheckResult:
    """What one check found."""

    name: str
    outcome: Outcome
    severity: Severity
    message: str

    @property
    def failed(self) -> bool:
        return self.outcome is Outcome.FAILED

    @property
    def blocking(self) -> bool:
        """A failure serious enough to stop the pipeline."""
        return self.failed and self.severity is Severity.ERROR

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "outcome": str(self.outcome),
            "severity": str(self.severity),
            "message": self.message,
        }


# A check body asks one question of the table and returns the answer as a
# failure message, or None if there is nothing to report.
CheckBody = Callable[[pd.DataFrame], str | None]


@dataclass(frozen=True, slots=True)
class Check:
    """One named, callable question about the table.

    An object rather than a decorated function so the suite stays
    *introspectable*: documentation and tests can list the checks, and assert
    the names are unique, without executing any of them against a frame that
    does not have the columns they expect.
    """

    name: str
    """Human-readable, and the report's stable key. Phrased as the property
    being asserted ("match ids are unique") so a passing report reads as a list
    of true statements and a failing one names the statement that stopped
    being true."""

    body: CheckBody
    severity: Severity = Severity.ERROR
    min_rows: int = 0
    """Below this many rows the check is skipped rather than run. For
    distribution checks, which are statements about a population."""

    def __call__(self, matches: pd.DataFrame) -> CheckResult:
        if len(matches) < self.min_rows:
            return CheckResult(
                name=self.name,
                outcome=Outcome.SKIPPED,
                severity=self.severity,
                message=f"{len(matches):,} rows, needs {self.min_rows:,}",
            )
        message = self.body(matches)
        return CheckResult(
            name=self.name,
            outcome=Outcome.PASSED if message is None else Outcome.FAILED,
            severity=self.severity,
            message=message if message is not None else "ok",
        )


def check(
    name: str,
    *,
    severity: Severity = Severity.ERROR,
    min_rows: int = 0,
) -> Callable[[CheckBody], Check]:
    """Decorator form of :class:`Check`, so a suite reads as a list of questions."""

    def decorate(body: CheckBody) -> Check:
        return Check(name=name, body=body, severity=severity, min_rows=min_rows)

    return decorate


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The outcome of running a suite of checks over one table."""

    results: tuple[CheckResult, ...]
    rows: int

    @property
    def ok(self) -> bool:
        """True when nothing blocking failed. Warnings do not make it false."""
        return not any(result.blocking for result in self.results)

    def of(self, outcome: Outcome) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.outcome is outcome)

    @property
    def blocking(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.blocking)

    def summary(self) -> str:
        passed = len(self.of(Outcome.PASSED))
        failed = len(self.of(Outcome.FAILED))
        skipped = len(self.of(Outcome.SKIPPED))
        return f"{passed} passed, {failed} failed, {skipped} skipped " f"over {self.rows:,} matches"

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "rows": self.rows,
            "summary": self.summary(),
            "results": [result.to_dict() for result in self.results],
        }

    def to_markdown(self) -> str:
        """Render as a table. Failures first — a report where the one broken
        check sits nineteenth is a report whose top line is all anyone reads."""
        symbol = {Outcome.PASSED: "pass", Outcome.FAILED: "FAIL", Outcome.SKIPPED: "skip"}
        ordered = sorted(self.results, key=lambda result: (result.outcome is not Outcome.FAILED,))
        lines = [
            "| | Check | Severity | Detail |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {symbol[result.outcome]} | {result.name} | {result.severity} | {result.message} |"
            for result in ordered
        ]
        return "\n".join(lines)


def run_checks(matches: pd.DataFrame, checks: Iterable[Check]) -> ValidationReport:
    """Run every check and collect the results.

    Every check runs even after one fails. A validation pass that stops at the
    first problem makes fixing a batch of them an iterative guessing game, and
    these are cheap: the expensive part is reading the table, which happened
    once before the first check.
    """
    return ValidationReport(
        results=tuple(single(matches) for single in checks),
        rows=len(matches),
    )


def names(checks: Sequence[Check]) -> tuple[str, ...]:
    """The check names in suite order, for tests and documentation."""
    return tuple(single.name for single in checks)
