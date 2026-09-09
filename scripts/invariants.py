#!/usr/bin/env python
"""Run CI's architectural invariants here, instead of learning about them there.

    python scripts/invariants.py                  # the architecture gates
    python scripts/invariants.py --job authorship # the commit-history gates
    python scripts/invariants.py --list           # name them, run nothing

The invariants are shell — eighteen greps that hold the dependency graph in the
shape the documents claim it has — and they live in `.github/workflows/ci.yml`
because that is where they are enforced. Until this script they could only be
*run* there: `make quality` covers lint, format and types, so the first news of
a violated boundary was a red job six minutes after a push.

**It reads the workflow rather than restating it.** A second copy of eighteen
greps is a second copy that drifts, and the half that drifts is always the one
nobody looked at again — which is the same argument `src/pipelines/derived.py`
makes about writing a pipeline twice. So the workflow stays the single source of
truth and this parses it. A step edited in CI is a step that changes here on the
next run, with nothing to keep in step.

**A job that has stopped existing is an error, not a pass.** A gate that
silently checks nothing is worse than no gate: it reports success in the one
situation where it should be shouting. Renaming the job in the workflow makes
this exit non-zero and say so.

Only jobs whose steps are pure shell can run here — the architecture greps and
the authorship checks. `Lint, types, tests` sets up Python and installs pinned
requirements first, which is `make quality` and `make test` rather than this.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.utils.paths import PROJECT_ROOT  # noqa: E402

WORKFLOW = Path(".github") / "workflows" / "ci.yml"
DEFAULT_JOB = "invariants"
"""The job this exists for: the greps that hold the dependency graph.

``authorship`` is the other one that runs unchanged here — it is git and grep
— and it is reachable with ``--job`` rather than by default, because it reads
the whole history and this is meant to be the check somebody runs before every
push.
"""

SHELL = ["bash", "-e"]
"""What GitHub runs a ``run:`` block with on Linux, so it is what runs here.

``-e`` matters: several of these steps are a grep followed by a test, and
without it a failing command mid-block would be shrugged off and the step would
pass on its last line.
"""


@dataclass(frozen=True, slots=True)
class Step:
    """One shell check out of the workflow."""

    name: str
    script: str


def steps(workflow: Path, job: str) -> list[Step]:
    """The runnable steps of ``job``, in the order CI runs them.

    Raises:
        ValueError: If the job is absent or has no shell steps. Both are the
            same failure — a gate that checks nothing — and both must be loud
            rather than reported as "no violations found". ``ValueError`` for
            the missing job rather than ``KeyError`` because the message is
            printed, and ``KeyError`` renders its own with quotes round it.
    """
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    jobs = document.get("jobs") or {}
    if job not in jobs:
        raise ValueError(f"no job {job!r} in {workflow}; it has {', '.join(sorted(jobs))}")
    found = [
        Step(name=str(one.get("name") or "unnamed"), script=str(one["run"]))
        for one in jobs[job].get("steps") or []
        if one.get("run")
    ]
    if not found:
        raise ValueError(f"job {job!r} has no shell steps to run")
    return found


def run(step: Step, *, root: Path) -> bool:
    """Run one step from the repository root. ``True`` when it passed.

    Output is captured and printed only on failure. Eighteen passing greps that
    each announce themselves is a wall of text somebody stops reading, and the
    thing worth reading is the one that failed.
    """
    done = subprocess.run([*SHELL, "-c", step.script], cwd=root, capture_output=True, text=True)
    if done.returncode != 0:
        print(f"  FAIL  {step.name}")
        for line in (done.stdout + done.stderr).splitlines():
            print(f"        {line}")
        return False
    print(f"  ok    {step.name}")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        metavar="ID",
        help=f"workflow job to run. Repeatable. Defaults to {DEFAULT_JOB}.",
    )
    parser.add_argument("--workflow", type=Path, default=None, help="the workflow file to read")
    parser.add_argument("--list", action="store_true", help="name the steps, run nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workflow = args.workflow or PROJECT_ROOT / WORKFLOW
    failed = 0
    for job in args.job or [DEFAULT_JOB]:
        try:
            found = steps(workflow, job)
        except (ValueError, OSError) as error:
            print(f"{error}")
            return 1
        print(f"\n{job}: {len(found)} step(s)")
        if args.list:
            for step in found:
                print(f"  {step.name}")
            continue
        failed += sum(not run(step, root=PROJECT_ROOT) for step in found)

    if args.list:
        return 0
    print("\nevery invariant holds" if not failed else f"\n{failed} invariant(s) violated")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
