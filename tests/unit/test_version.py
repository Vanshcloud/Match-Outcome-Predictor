"""One version number, one place.

The sibling project shipped 0.1.0 in five files while its release tag said
v1.0.0, because nothing compared them. This is that comparison.
"""

from __future__ import annotations

import re
import tomllib

import src
from src.utils.paths import PROJECT_ROOT

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def test_version_is_semver() -> None:
    assert SEMVER.match(src.__version__), f"not semver: {src.__version__!r}"


def test_pyproject_reads_the_version_rather_than_repeating_it() -> None:
    """A literal `version = "..."` in pyproject.toml is a second copy to drift.

    setuptools' dynamic-version mechanism is what keeps there being only one,
    so the assertion is that the mechanism is configured — not that two numbers
    happen to match today.
    """
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert "version" in pyproject["project"]["dynamic"]
    assert "version" not in pyproject["project"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "src.__version__"}


def test_user_agent_default_matches_the_project_version() -> None:
    """The User-Agent identifies this client to a third-party host. A stale
    version there is a small lie told on every request."""
    from src.utils.http import HttpClient

    major_minor = ".".join(src.__version__.split(".")[:2])
    client = HttpClient()
    assert major_minor in client.session.headers["User-Agent"]
    client.close()
