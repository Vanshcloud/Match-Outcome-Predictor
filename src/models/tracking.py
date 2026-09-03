"""Run logging, to a local MLflow store, that cannot fail a training run.

There are runs now, so there is something to track: six model families, a
search that produced their settings, and an ablation that will produce more.
What is logged is small — the settings a run used and the scores it got — and
it goes into a **SQLite file** under ``models/``, because a tracking server is
infrastructure and this is one person's laptop. Not MLflow's directory-based
file store, which its 3.x releases refuse outright: that backend is in
maintenance mode and raises unless an environment variable opts out of the
warning. SQLite is the supported local backend and needs nothing running.

Two decisions worth stating.

**A tracking failure is logged and swallowed.** The real output of a training
run is the score table and its manifest, written by the pipeline before
anything here is called. Losing the run log is an inconvenience; losing the run
because the log could not be written is a training command that fails for a
reason that has nothing to do with training.

**Nothing here is imported by the model code.** The zoo does not know it is
being tracked, so a model can be fitted and scored with MLflow absent
entirely — which is how the test suite runs it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

EXPERIMENT = "match-outcome-predictor"
STORE_FILENAME = "mlflow.db"
ARTIFACTS_DIRNAME = "mlartifacts"


def store_uri(model_dir: Path) -> str:
    """Where runs are recorded: a SQLite file inside the model directory."""
    return f"sqlite:///{(model_dir / STORE_FILENAME).resolve()}"


def artifact_uri(model_dir: Path) -> str:
    """Where MLflow would put artefacts, set explicitly.

    Nothing here logs one, and that is exactly why it has to be named: left
    unset, MLflow defaults the experiment's artefact root to ``./mlruns``
    *relative to the working directory*, so running the trainer from the repo
    root would create a directory the clean-checkout gate then fails on.
    """
    return (model_dir / ARTIFACTS_DIRNAME).resolve().as_uri()


def log_run(
    name: str,
    parameters: Mapping[str, Any],
    metrics: Mapping[str, float],
    *,
    model_dir: Path,
    experiment: str = EXPERIMENT,
) -> str | None:
    """Record one model's settings and scores. Returns the run id, or None.

    Args:
        name: The run's name, which is the model's name.
        parameters: What the model was configured with.
        metrics: Its scores. Non-finite values are dropped — MLflow stores
            infinity as a string that no comparison in its UI can order, and
            the home-always baseline produces one every run.
        model_dir: Where the file store lives.
        experiment: The experiment to file the run under.
    """
    finite = {key: float(value) for key, value in metrics.items() if _is_finite(value)}
    try:
        import mlflow

        model_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(store_uri(model_dir))
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=artifact_uri(model_dir))
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(dict(parameters))
            mlflow.log_metrics(finite)
            return str(run.info.run_id)
    except Exception as error:  # noqa: BLE001 - see the module docstring
        logger.warning("run not tracked (%s): %s", type(error).__name__, error)
        return None


def _is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")
