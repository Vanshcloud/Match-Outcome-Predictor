"""Run logging that cannot fail a training run.

The behaviour worth testing is not that MLflow works — it does — but that this
wrapper swallows everything. The real output of a training run is the score
table, written before anything here is called, and a command that failed
because a log directory was read-only would be failing for a reason that has
nothing to do with training.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.models.tracking import ARTIFACTS_DIRNAME, EXPERIMENT, STORE_FILENAME, log_run, store_uri


def test_a_run_is_written_to_a_local_sqlite_file(tmp_path: Path) -> None:
    """SQLite, not a server and not MLflow's directory store: a server is
    infrastructure, and the directory store is in maintenance mode and refuses
    to be written to at all."""
    run = log_run("lightgbm", {"n_estimators": 400}, {"log_loss": 1.0}, model_dir=tmp_path)
    assert run is not None
    assert (tmp_path / STORE_FILENAME).is_file()


def test_the_run_carries_its_settings_and_its_scores(tmp_path: Path) -> None:
    import mlflow

    log_run("xgboost", {"max_depth": 5}, {"log_loss": 1.0213, "rps": 0.21}, model_dir=tmp_path)
    mlflow.set_tracking_uri(store_uri(tmp_path))
    found = mlflow.search_runs(experiment_names=[EXPERIMENT])
    assert found["params.max_depth"].iloc[0] == "5"
    assert found["metrics.log_loss"].iloc[0] == pytest.approx(1.0213)


def test_an_infinite_metric_is_dropped_rather_than_stored(tmp_path: Path) -> None:
    """MLflow stores infinity as a string no comparison in its UI can order,
    and the home-always baseline produces one every run."""
    import mlflow

    log_run("home_always", {}, {"log_loss": math.inf, "rps": 0.43}, model_dir=tmp_path)
    mlflow.set_tracking_uri(store_uri(tmp_path))
    found = mlflow.search_runs(experiment_names=[EXPERIMENT])
    assert "metrics.log_loss" not in found.columns
    assert found["metrics.rps"].iloc[0] == pytest.approx(0.43)


def test_a_tracking_failure_is_logged_and_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property this module exists for. Losing the run log is an
    inconvenience; losing the run because the log could not be written is a
    training command that fails for an unrelated reason."""
    import mlflow

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("the store is read-only")

    monkeypatch.setattr(mlflow, "set_experiment", refuse)
    assert log_run("lightgbm", {}, {"log_loss": 1.0}, model_dir=tmp_path) is None


def test_the_store_uri_names_a_sqlite_file_in_the_model_directory(tmp_path: Path) -> None:
    assert store_uri(tmp_path).startswith("sqlite:///")
    assert store_uri(tmp_path).endswith(STORE_FILENAME)


def test_nothing_is_written_outside_the_model_directory(tmp_path: Path) -> None:
    """Left unset, MLflow roots artefacts at `./mlruns` relative to the working
    directory — which the clean-checkout gate would then fail on."""
    log_run("lightgbm", {}, {"log_loss": 1.0}, model_dir=tmp_path / "models")
    experiment_root = tmp_path / "models" / ARTIFACTS_DIRNAME
    assert not (Path.cwd() / "mlruns").exists()
    assert experiment_root.parent.is_dir()
