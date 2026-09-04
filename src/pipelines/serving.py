"""Fit the shipped model once, write it down, and read it back to serve.

Milestone 11 is the first thing here that needs a model to outlive the process
that fitted it. Everything before it fits inside a backtest fold and is thrown
away, which is what makes the folds honest; a service cannot work that way, so
the fit happens once — in :mod:`scripts.build_model` — and the result is
persisted with a manifest beside it.

**This is the layer that knows where things live.** :mod:`src.models.artifact`
turns frames into a fitted object and back into probabilities and touches no
path; the file format, the directory and the provenance are here, matching the
split every other pipeline in this package already follows.

**The manifest is not decoration.** A pickled estimator is only loadable by
compatible versions of the library that produced it, and the failure when it is
not ranges from an exception to a silently different tree. So the versions that
did the fitting are recorded, the loader compares them, and a mismatch is
reported rather than discovered later as a forecast nobody can reproduce. The
checksum comes from the same :mod:`src.ingestion.manifest` every derived table
already uses.

**Fixtures are looked up, never recomputed.** A design row is thirty columns
built by the audited pipelines and proved causal by the leakage suite;
rebuilding one at request time would be a second implementation of the feature
layer, living outside every probe that guards the first. So the service reads
the row the batch build wrote. What that costs is stated plainly rather than
worked around: this API prices fixtures that are in the feature table, and the
provider publishes results rather than a fixture list, so an unplayed match is
not something this project has the inputs to price at all.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src import __version__
from src.ingestion.manifest import checksum, read_manifest, write_manifest
from src.models.artifact import ArtifactError, ServableModel, fit_servable
from src.models.dataset import DESIGN_COLUMNS
from src.models.ensemble import MEMBERS
from src.pipelines.tables import KEY_COLUMN, TablePaths, load_modelling_frame
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_FILENAME = "servable.joblib"
MANIFEST_FILENAME = "servable.manifest.json"

ARTIFACT_KIND = "servable-model"

FIXTURE_COLUMNS: tuple[str, ...] = (
    KEY_COLUMN,
    "competition_id",
    "competition",
    "country",
    "season",
    "date",
    "home_team",
    "away_team",
)
"""What a fixture is identified and described by in a response.

The canonical columns a caller can recognise a match from, and nothing else —
no scoreline, and no odds. The scoreline because a service that returned the
result beside its forecast would be answering a different question; the odds
because they are the benchmark this project measures itself against and
handing them back with a prediction invites the comparison to be made wrongly.
"""


class ServingError(RuntimeError):
    """A servable model could not be built, written or read."""


def _library_versions() -> dict[str, str]:
    """The versions that matter to a pickle, read from the installed packages.

    Only the three that actually appear inside a fitted blend. A full
    ``pip freeze`` in the manifest would go stale on every unrelated upgrade
    and teach a reader to ignore the mismatch warning that matters.
    """
    from importlib.metadata import version

    return {name: version(name) for name in ("scikit-learn", "xgboost", "numpy")}


# ---- writing ----------------------------------------------------------------


def build_servable(
    paths: TablePaths,
    *,
    members: Sequence[str] = MEMBERS,
    columns: Sequence[str] = DESIGN_COLUMNS,
    competitions: Sequence[str] | None = None,
) -> ServableModel | None:
    """Fit the shipped model on everything the three tables hold.

    Returns:
        The fitted model, or ``None`` when a table is missing or the filter
        selected nothing — the clean-checkout case, which
        :func:`~src.pipelines.tables.load_modelling_frame` already reports as a
        logged error rather than an exception.
    """
    frame = load_modelling_frame(paths, competitions=competitions)
    if frame is None:
        return None
    return fit_servable(frame, members=members, columns=columns)


def save_servable(model: ServableModel, model_dir: Path) -> Path:
    """Write ``model`` to ``model_dir`` with a manifest beside it."""
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / MODEL_FILENAME
    joblib.dump(model, destination)

    write_manifest(
        model_dir / MANIFEST_FILENAME,
        model_dir,
        [destination],
        extra={
            "kind": ARTIFACT_KIND,
            "model": model.name,
            "members": list(model.member_names),
            "columns": list(model.columns),
            "temperature": model.temperature,
            "trained_matches": model.trained_matches,
            "trained_from": model.trained_from.date().isoformat(),
            "trained_through": model.trained_through.date().isoformat(),
            "project_version": __version__,
            "libraries": _library_versions(),
        },
    )
    logger.info("wrote %s — %s", destination.name, model.name)
    return destination


# ---- reading ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A servable model and the manifest that came with it."""

    model: ServableModel
    manifest: Mapping[str, Any]
    path: Path

    @property
    def library_mismatches(self) -> tuple[str, ...]:
        """Libraries whose installed version differs from the one that fitted this.

        Reported, not enforced. A patch release of numpy will not change a
        forecast and refusing to start would be the wrong call; a major one
        might, and a service that never mentioned it would be the wrong call
        too. :mod:`api` puts the list in ``/version``.
        """
        recorded = self.manifest.get("libraries")
        if not isinstance(recorded, dict):
            return ()
        installed = _library_versions()
        return tuple(
            f"{name}: fitted with {was}, running {installed.get(name, 'absent')}"
            for name, was in sorted(recorded.items())
            if installed.get(name) != was
        )


def load_servable(model_dir: Path) -> LoadedModel:
    """Read the persisted model back, checking the bytes first.

    Raises:
        ServingError: If the artefact or its manifest is absent, if the file
            does not match the checksum recorded for it, or if unpickling
            fails. All four are the same problem from the caller's side —
            there is no model to serve — and each names which one it was.
    """
    path = model_dir / MODEL_FILENAME
    manifest_path = model_dir / MANIFEST_FILENAME
    if not path.is_file():
        raise ServingError(f"no model at {path}; run `make model` to fit one")
    if not manifest_path.is_file():
        raise ServingError(f"no manifest at {manifest_path}; the model cannot be identified")

    try:
        manifest = read_manifest(manifest_path)
    except (OSError, json.JSONDecodeError) as error:
        raise ServingError(f"unreadable manifest at {manifest_path}: {error}") from error

    _verify_checksum(path, manifest)

    try:
        loaded = joblib.load(path)
    except Exception as error:  # unpickling can raise anything at all
        raise ServingError(f"could not unpickle {path}: {error}") from error
    if not isinstance(loaded, ServableModel):
        raise ServingError(f"{path} holds {type(loaded).__name__}, not a servable model")

    logger.info("loaded %s trained through %s", loaded.name, loaded.trained_through.date())
    return LoadedModel(model=loaded, manifest=manifest, path=path)


def _verify_checksum(path: Path, manifest: Mapping[str, Any]) -> None:
    """Fail unless the file on disk is the one the manifest describes.

    A half-written artefact — an interrupted `make model`, a truncated copy
    into an image — unpickles to an exception at best and to something
    plausible at worst. Checking is one hash of a file that is read once per
    process start.
    """
    files = manifest.get("files")
    recorded = next(
        (
            entry.get("sha256")
            for entry in (files if isinstance(files, list) else [])
            if isinstance(entry, dict) and entry.get("path") == path.name
        ),
        None,
    )
    if recorded is None:
        raise ServingError(f"the manifest records no checksum for {path.name}")
    if checksum(path) != recorded:
        raise ServingError(f"{path.name} does not match its manifest checksum; refit or re-copy")


# ---- the table the service answers from --------------------------------------


def _normalise(value: object) -> str:
    """A team or competition name reduced to what a caller can be expected to type."""
    return " ".join(str(value).split()).casefold()


@dataclass(frozen=True, slots=True)
class FixtureIndex:
    """The modelling frame, with the two ways a caller names a match resolved.

    Built once at startup. Both lookups are dictionaries rather than a
    ``DataFrame.query`` per request: the frame is three hundred thousand rows,
    and a scan per request would make the service's latency a function of how
    much football has been played.
    """

    frame: pd.DataFrame
    by_id: Mapping[str, int]
    by_key: Mapping[tuple[str, str, str, str], int]

    def __len__(self) -> int:
        return len(self.frame)

    def resolve(
        self,
        *,
        match_id: str | None = None,
        competition_id: str | None = None,
        home_team: str | None = None,
        away_team: str | None = None,
        date: str | None = None,
    ) -> pd.DataFrame | None:
        """The one row for a fixture, or ``None`` if the table has no such match.

        ``match_id`` wins when given. Otherwise the natural key — competition,
        both clubs and the date — which is what someone reading a fixture list
        actually has. Team names are matched case- and whitespace-insensitively
        because a provider's spacing is not something a caller should have to
        reproduce.
        """
        position: int | None = None
        if match_id is not None:
            position = self.by_id.get(match_id)
        elif None not in (competition_id, home_team, away_team, date):
            key = (
                str(competition_id),
                _normalise(home_team),
                _normalise(away_team),
                str(date),
            )
            position = self.by_key.get(key)
        if position is None:
            return None
        return self.frame.iloc[[position]]

    def search(
        self,
        *,
        competition_id: str | None = None,
        team: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        """Fixtures matching every filter given, most recent first.

        Discovery, because without it a caller cannot name a match: the
        provider publishes no fixture list, so the only fixtures that exist are
        the ones in the table, and a service that could only answer about
        matches you already knew the id of would be answering nobody.
        """
        rows = self.frame
        if competition_id is not None:
            rows = rows[rows["competition_id"] == competition_id]
        if team is not None:
            wanted = _normalise(team)
            rows = rows[
                rows["home_team"].map(_normalise).eq(wanted)
                | rows["away_team"].map(_normalise).eq(wanted)
            ]
        if since is not None:
            rows = rows[rows["date"] >= pd.Timestamp(since)]
        if until is not None:
            rows = rows[rows["date"] <= pd.Timestamp(until)]
        return rows.sort_values("date", ascending=False).head(limit)


def build_index(frame: pd.DataFrame) -> FixtureIndex:
    """Index a modelling frame by match id and by natural key.

    A duplicate natural key keeps its **first** row and is logged. The
    validation suite already fails a table with two fixtures on one date
    between one pair of clubs, so this is the belt to that braces — and
    silently serving whichever row pandas returned last is the one outcome
    worth ruling out.
    """
    identifiers = frame[KEY_COLUMN].astype(str).tolist()
    by_id = {identifier: position for position, identifier in enumerate(identifiers)}

    by_key: dict[tuple[str, str, str, str], int] = {}
    collisions = 0
    dates = pd.to_datetime(frame["date"]).dt.date.astype(str).tolist()
    competitions = frame["competition_id"].astype(str).tolist()
    home = [_normalise(name) for name in frame["home_team"]]
    away = [_normalise(name) for name in frame["away_team"]]
    for position, key in enumerate(zip(competitions, home, away, dates, strict=True)):
        if key in by_key:
            collisions += 1
            continue
        by_key[key] = position
    if collisions:
        logger.warning("%d fixture(s) share a natural key; the earliest row wins", collisions)

    return FixtureIndex(frame=frame, by_id=by_id, by_key=by_key)


def load_index(paths: TablePaths) -> FixtureIndex | None:
    """The joined modelling frame, indexed. ``None`` when a table is absent."""
    frame = load_modelling_frame(paths)
    if frame is None:
        return None
    index = build_index(frame)
    logger.info("indexed %d fixture(s)", len(index))
    return index


# ---- what a served prediction is --------------------------------------------


@dataclass(frozen=True, slots=True)
class Prediction:
    """One fixture, three probabilities, and the caveats that belong with them."""

    fixture: Mapping[str, Any]
    probabilities: Mapping[str, float]
    model: str
    model_version: str
    in_sample: bool
    predicted_at: datetime

    def as_record(self) -> dict[str, Any]:
        """The row a prediction log stores: the inputs, the output, the model."""
        return {
            "match_id": str(self.fixture[KEY_COLUMN]),
            "competition_id": str(self.fixture["competition_id"]),
            "home_team": str(self.fixture["home_team"]),
            "away_team": str(self.fixture["away_team"]),
            "match_date": str(self.fixture["date"]),
            "model": self.model,
            "model_version": self.model_version,
            "prob_home": self.probabilities["home"],
            "prob_draw": self.probabilities["draw"],
            "prob_away": self.probabilities["away"],
            "in_sample": self.in_sample,
            "predicted_at": self.predicted_at,
        }


def describe_fixture(row: pd.Series) -> dict[str, Any]:
    """The identifying columns of one match, as plain JSON-able values."""
    described: dict[str, Any] = {}
    for column in FIXTURE_COLUMNS:
        value = row.get(column)
        if column == "date":
            described[column] = pd.Timestamp(str(value)).date().isoformat()
        elif pd.isna(value):
            described[column] = None
        else:
            described[column] = str(value)
    return described


def predict_fixtures(
    model: ServableModel, rows: pd.DataFrame, *, model_version: str = __version__
) -> list[Prediction]:
    """Price every row of ``rows``, one :class:`Prediction` each.

    One call into the model for the whole frame rather than one per row: the
    estimators vectorise, and a loop would make a batch request cost what its
    rows cost separately.

    Raises:
        ServingError: If the rows cannot be priced — a missing design column,
            or an empty frame. Both come out of the model layer as its own
            errors, and are re-raised here under the name the API catches.
    """
    try:
        stated = model.predict(rows)
    except (ArtifactError, ValueError) as error:
        raise ServingError(str(error)) from error

    now = datetime.now(tz=UTC)
    return [
        Prediction(
            fixture=describe_fixture(row),
            probabilities=_as_probabilities(stated[position]),
            model=model.name,
            model_version=model_version,
            in_sample=model.is_in_sample(pd.Timestamp(row["date"])),
            predicted_at=now,
        )
        for position, (_, row) in enumerate(rows.iterrows())
    ]


def _as_probabilities(row: np.ndarray) -> dict[str, float]:
    """A CLASSES-ordered array as the three named probabilities.

    Named rather than positional in the response: an array of three floats is
    the single easiest thing in this project to read in the wrong order, and it
    stays wrong while summing to one.
    """
    return {
        "home": float(row[0]),
        "draw": float(row[1]),
        "away": float(row[2]),
    }
