"""Where a reader's favourites are kept between visits.

The whole of the persistence: one JSON file, keyed by whatever
:mod:`dashboard.domain.identity` says the reader is. Favourites kept in
``st.session_state`` would last only as long as a browser tab.

**A file rather than the database that already exists.** ``docker-compose.yml``
runs PostgreSQL, but it is the *service's*: it holds served predictions so the
model's calibration can be measured against what happened. Putting a reader's
league list in it would mean the dashboard image growing ``psycopg`` —
explicitly listed as an omission in ``requirements-dashboard.txt`` — a
connection pool nobody is tracking across Streamlit reruns, and a schema
migration for a preference. This file is a few hundred bytes read with the
standard library.

ponytail: one JSON file, rewritten whole, last writer wins. Two people editing
different profiles in the same second lose one edit. The upgrade is the
Postgres that is already in the compose file, and the seam for it is this
module — nothing above it knows what a profile is stored in.

**A failed write is a sentence, not an exception.** The compose file mounts
``data/`` read-only, a container can run with no writable volume at all, and a
dashboard that crashed on a click because it could not save a preference would
be worse than one that says it could not. :data:`last_error` carries the
reason and the sidebar renders it.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.utils.config import load_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

STORE_ENV = "DASHBOARD_PROFILE_STORE"
"""Where the file lives, overriding the default under the data directory.

An environment variable rather than a ``configs/config.yaml`` entry, for the
reason ``DASHBOARD_FIXTURE_PROVIDER`` is one: the settings model is loaded by
the pipelines, the API and the dashboard alike, and a key only one of the three
reads is one the other two carry for nothing.
"""

LEAGUES = "leagues"
TEAMS = "teams"

last_error: str | None = None
"""Why the last save did not happen, in the words the sidebar prints.

Module state, and deliberately: the alternative is every accessor in
:mod:`dashboard.domain.favourites` returning a value *and* an error, on a path
where the caller is a view that mostly wants the value.
"""


def path() -> Path:
    """The store file. ``$DASHBOARD_PROFILE_STORE``, or under the data directory.

    Resolved per call rather than cached. It is two environment reads and a
    YAML parse that :func:`~src.utils.config.load_settings` already does once
    per script run, and a cached path is one a test has to reset and a
    container has to restart to change.
    """
    stated = os.environ.get(STORE_ENV)
    if stated:
        return Path(stated)
    return load_settings().paths.data_dir / "dashboard" / "profiles.json"


def read() -> dict[str, dict[str, list[str]]]:
    """Every profile the file holds, or nothing if it holds nothing readable.

    A missing file is the ordinary first-run state. A *corrupt* one is not, and
    it still returns empty rather than raising: a reader whose favourites file
    was truncated by a full disk should get a dashboard with no favourites, not
    a stack trace on every page.
    """
    target = path()
    try:
        with target.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as failure:
        logger.warning("could not read %s: %s", target, failure)
        return {}
    return (
        {str(name): _clean(saved) for name, saved in loaded.items() if isinstance(saved, dict)}
        if isinstance(loaded, dict)
        else {}
    )


def saved(identity: str) -> dict[str, list[str]]:
    """One reader's lists, empty for a reader the file has never seen."""
    return read().get(identity, {LEAGUES: [], TEAMS: []})


def save(identity: str, *, leagues: list[str], teams: list[str]) -> bool:
    """Write one reader's lists. ``False`` with :data:`last_error` set on failure.

    Read-modify-write of the whole file, which is the smallest correct thing
    for a document this size: the alternative is a second file per profile and
    a directory to enumerate.
    """
    global last_error
    everyone = read()
    everyone[identity] = {LEAGUES: list(leagues), TEAMS: list(teams)}
    target = path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_atomically(target, everyone)
    except OSError as failure:
        last_error = f"favourites could not be saved to {target}: {failure}"
        logger.warning("%s", last_error)
        return False
    last_error = None
    return True


def identities() -> list[str]:
    """Every reader the file knows about, in the order a picker lists them."""
    return sorted(read())


def _write_atomically(target: Path, everyone: dict[str, dict[str, list[str]]]) -> None:
    """Write beside the target and rename over it.

    A partial JSON document is unreadable, and the failure that produces one —
    a process killed mid-write, a disk filling — is exactly when a reader can
    least afford to lose the file. ``os.replace`` is atomic within a
    filesystem, and the temporary file is made in the target's own directory so
    that it is one. ``mkstemp`` rather than ``NamedTemporaryFile``: the handle
    has to outlive the ``with`` that writes it, and a file object kept open
    past its context manager is the thing that reads as a bug six months on.
    """
    descriptor, made = tempfile.mkstemp(dir=target.parent, prefix=target.name, suffix=".part")
    partial = Path(made)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(everyone, handle, indent=2, sort_keys=True, ensure_ascii=False)
        partial.replace(target)
    except OSError:
        partial.unlink(missing_ok=True)
        raise


def _clean(saved_lists: dict[str, Any]) -> dict[str, list[str]]:
    """One profile's document, with anything that is not a list of names dropped.

    The file is editable by hand — that is a feature of a JSON store, and it
    means the parser meets whatever a person typed into it.
    """
    return {key: _names(saved_lists.get(key)) for key in (LEAGUES, TEAMS)}


def _names(value: Any) -> list[str]:
    return [str(one) for one in value] if isinstance(value, list) else []
