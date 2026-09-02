"""The commit-msg hook is this project's only enforcement of sole authorship.

A rule that lives in a prompt survives until the next fresh session; a hook
survives a reclone, because core.hooksPath points at a version-controlled
directory. These tests exist so the hook cannot silently rot — if someone edits
its patterns, this suite says which behaviour changed.

Both halves matter. The rejection cases stop attribution landing; the
acceptance cases stop the pattern getting so broad that it blocks honest prose,
which is how a guard ends up disabled by whoever it inconveniences first.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from src.utils.paths import PROJECT_ROOT

HOOK = PROJECT_ROOT / "scripts" / "hooks" / "commit-msg"

# `git var GIT_AUTHOR_IDENT` honours these, so the hook's author check is
# deterministic here rather than dependent on the machine's git config.
VANSH_ENV = {
    "GIT_AUTHOR_NAME": "Vanshcloud",
    "GIT_AUTHOR_EMAIL": "vanshwar@gmail.com",
    "GIT_COMMITTER_NAME": "Vanshcloud",
    "GIT_COMMITTER_EMAIL": "vanshwar@gmail.com",
    "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
    "HOME": str(Path.home()),
}


def run_hook(message: str, env: dict[str, str] | None = None) -> int:
    """Run the commit-msg hook against ``message`` and return its exit code."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(message)
        path = handle.name
    try:
        result = subprocess.run(
            ["bash", str(HOOK), path],
            capture_output=True,
            env=env or VANSH_ENV,
            check=False,
        )
        return result.returncode
    finally:
        Path(path).unlink()


def test_the_hook_exists() -> None:
    assert HOOK.is_file(), f"missing hook: {HOOK}"


def test_the_hook_is_executable() -> None:
    """git will not run a hook without the bit set, and it fails silently —
    the commit simply succeeds unguarded."""
    assert HOOK.stat().st_mode & 0o111, "hook is not executable"


REJECTED = [
    pytest.param(
        "feat: a change\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n", id="coauthor"
    ),
    pytest.param("feat: a change\n\nco-authored-by: someone <x@y.z>\n", id="coauthor-lowercase"),
    pytest.param(
        "feat: a change\n\n\U0001f916 Generated with [Claude Code](https://claude.ai/code)\n",
        id="generated-with",
    ),
    pytest.param(
        "feat: a change\n\nClaude-Session: https://claude.ai/code/session_x\n", id="session-trailer"
    ),
    pytest.param("feat: a change\n\nSigned-off-by: Claude <x@y.z>\n", id="signed-off-by-claude"),
]


@pytest.mark.parametrize("message", REJECTED)
def test_attribution_is_rejected(message: str) -> None:
    assert run_hook(message) != 0, "hook accepted an attribution trailer"


ACCEPTED = [
    pytest.param("feat(ingestion): add the football-data.co.uk adapter\n", id="plain"),
    pytest.param(
        "chore: add .claude/ to gitignore\n\nThe directory is local tooling state.\n",
        id="legitimate-claude-mention",
    ),
    pytest.param(
        "docs: explain how the split was generated with a fixed seed\n",
        id="generated-in-prose",
    ),
]


@pytest.mark.parametrize("message", ACCEPTED)
def test_legitimate_messages_are_accepted(message: str) -> None:
    assert run_hook(message) == 0, "hook rejected a legitimate message"


def test_a_non_vansh_author_is_rejected() -> None:
    """Absence of a co-author trailer is not the same as the right author."""
    env = dict(VANSH_ENV, GIT_AUTHOR_NAME="Someone Else", GIT_AUTHOR_EMAIL="someone@example.com")
    assert run_hook("feat: a change\n", env=env) != 0


def test_the_hook_is_the_one_git_will_actually_run() -> None:
    """A perfect hook in a directory git does not consult protects nothing.
    `make hooks` sets this; the test states what the setting must be."""
    result = subprocess.run(
        ["git", "config", "core.hooksPath"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "scripts/hooks", "run `make hooks`"
