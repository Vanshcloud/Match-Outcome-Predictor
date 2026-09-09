"""The local runner for CI's boundary checks.

What is worth testing is not that the eighteen greps pass — CI runs them, and
so does `make invariants` — but that this script cannot *quietly* run none of
them. A gate that reports success because it found nothing to check is worse
than no gate, and every case below is a version of that.

The real workflow is read once, to prove the job it defaults to is still there.
Everything else runs against a synthetic one, so a test does not depend on
which greps the project happens to have today.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.utils.paths import PROJECT_ROOT


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "invariants_cli", PROJECT_ROOT / "scripts" / "invariants.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()

WORKFLOW = """
name: CI
jobs:
  invariants:
    name: Architectural invariants
    steps:
      - uses: actions/checkout@v5
      - name: A boundary that holds
        run: |
          echo "fine"
      - name: A boundary that does not
        run: |
          echo "the dashboard reaches past the reporting layer"
          exit 1
  backend:
    name: Lint, types, tests
    steps:
      - uses: actions/setup-python@v6
"""


@pytest.fixture
def workflow(tmp_path: Path) -> Path:
    path = tmp_path / "ci.yml"
    path.write_text(WORKFLOW, encoding="utf-8")
    return path


def test_the_job_it_defaults_to_is_the_one_ci_actually_runs() -> None:
    """The single coupling to the real file, and the one worth pinning: rename
    that job in the workflow and this script would have nothing to run."""
    found = CLI.steps(PROJECT_ROOT / CLI.WORKFLOW, CLI.DEFAULT_JOB)
    assert len(found) > 1
    assert all(step.script.strip() for step in found)


def test_a_step_with_nothing_to_run_is_not_a_step(workflow: Path) -> None:
    """`uses:` steps check out the repository, which has already happened."""
    assert [step.name for step in CLI.steps(workflow, "invariants")] == [
        "A boundary that holds",
        "A boundary that does not",
    ]


def test_a_renamed_job_is_an_error_rather_than_a_clean_run(workflow: Path) -> None:
    """The failure this whole module is about: a gate that checks nothing must
    not report that nothing is wrong."""
    with pytest.raises(ValueError, match="no job"):
        CLI.steps(workflow, "boundaries")


def test_a_job_of_nothing_but_actions_is_an_error_too(workflow: Path) -> None:
    with pytest.raises(ValueError, match="no shell steps"):
        CLI.steps(workflow, "backend")


def test_a_violated_invariant_exits_non_zero_and_prints_what_the_grep_said(
    workflow: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The output of a passing grep is noise; the output of a failing one is
    the whole reason to run this."""
    assert CLI.main(["--workflow", str(workflow)]) == 1
    printed = capsys.readouterr().out
    assert "1 invariant(s) violated" in printed
    assert "reaches past the reporting layer" in printed
    assert "A boundary that holds" in printed


def test_every_step_runs_even_after_one_fails(
    workflow: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stopping at the first violation would mean one push per boundary."""
    CLI.main(["--workflow", str(workflow)])
    printed = capsys.readouterr().out
    assert "ok    A boundary that holds" in printed
    assert "FAIL  A boundary that does not" in printed


def test_a_missing_workflow_says_so_rather_than_raising(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert CLI.main(["--workflow", str(tmp_path / "gone.yml")]) == 1
    assert "gone.yml" in capsys.readouterr().out


def test_listing_names_the_checks_and_runs_none(
    workflow: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert CLI.main(["--workflow", str(workflow), "--list"]) == 0
    printed = capsys.readouterr().out
    assert "A boundary that does not" in printed
    assert "violated" not in printed


def test_the_shell_stops_at_the_first_failing_command_in_a_block() -> None:
    """Several of these steps are a grep followed by a test of its output.
    Without `-e` a failure mid-block would be shrugged off and the step would
    pass on its last line, which is a gate that always says yes."""
    assert CLI.SHELL == ["bash", "-e"]
    step = CLI.Step(name="two commands", script="false\necho 'kept going'\n")
    assert not CLI.run(step, root=PROJECT_ROOT)


def test_the_command_is_wired_into_the_makefile() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "scripts/invariants.py" in makefile
    assert "quality: lint format-check typecheck invariants" in makefile
