"""Typed configuration: YAML defaults, overlaid by environment variables.

Precedence, lowest to highest: ``configs/config.yaml`` -> ``.env`` -> real
environment variables. That direction is what lets a container override a
committed default without editing the image. Secrets live only in the
environment and never in the YAML, which is committed.

The result is frozen and validated. A config object that can be mutated after
load is a config object that will be mutated somewhere unhelpful, and then the
value a module reads depends on import order.

Pydantic v2 only. The v1 ``@validator`` decorator, ``.dict()`` and the nested
``class Config`` still run and merely warn, so the test suite runs with
``-W error::DeprecationWarning`` to keep them from creeping back in.

``pydantic-settings`` would also solve this. It is deliberately not a
dependency: the overlay this project needs is an explicit map of five
variables, which is less code than configuring the library to do the same
thing — and the explicit map is the point, see :data:`ENV_OVERRIDES`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.http import DEFAULT_USER_AGENT
from src.utils.paths import PROJECT_ROOT, resolve

DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"

# Environment variable -> dotted path into the config tree.
#
# Explicit rather than inferred from the model. An inferred mapping silently
# accepts a typo as "not a config variable"; here a variable that is not in
# this dict is not configurable by environment at all, and that is visible in
# one place a reader can check.
ENV_OVERRIDES: dict[str, str] = {
    "DATA_DIR": "paths.data_dir",
    "MODEL_DIR": "paths.model_dir",
    "API_PORT": "api.port",
    "API_MAX_BATCH": "api.max_batch",
    "PREDICTION_LOG_DSN": "api.prediction_log_dsn",
    "LOG_LEVEL": "logging.level",
}

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class _Strict(BaseModel):
    """Base for every config section.

    ``extra="forbid"`` turns a mistyped YAML key into an error instead of a
    value that is silently ignored — the failure mode where a setting appears
    to be configured and is not. ``frozen=True`` makes the loaded config
    immutable for the life of the process.

    ``validate_default=True`` is the non-obvious one. Pydantic skips validators
    on a field that falls back to its default, on the reasoning that a default
    written by the author is already correct. That reasoning does not hold for
    any default a validator is supposed to *transform*: ``data_dir`` defaults
    to the relative ``Path("data")`` and is only made absolute by
    ``_resolve_against_root``. Without this flag, a config that names the
    directory gets an absolute path and one that omits it gets a relative one —
    so every path in the project would quietly depend on the working directory
    the process started in, which is the exact failure src/utils/paths.py
    exists to prevent. Set on the base so no future section can reintroduce it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class PathsConfig(_Strict):
    """Where data and models live. Relative entries resolve against the root.

    Only the two roots are configurable. The subdirectories are derived
    properties rather than settings, because their relationship is a fact about
    the pipeline's layout, not a preference — letting someone point
    ``processed_dir`` outside ``data_dir`` creates a tree no other tool in the
    repo knows how to find.
    """

    data_dir: Path = Path("data")
    model_dir: Path = Path("models")

    @field_validator("data_dir", "model_dir")
    @classmethod
    def _resolve_against_root(cls, value: Path) -> Path:
        return resolve(value)

    @property
    def raw_dir(self) -> Path:
        """Provider files exactly as downloaded. Never edited in place."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Canonical match records: one schema, every provider, Parquet."""
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        """The feature store. Parquet, partitioned, rebuildable from processed."""
        return self.data_dir / "features"

    @property
    def reports_dir(self) -> Path:
        """Measurements, not data: backtest scores and the tables built from them.

        Separate from ``features_dir`` because the two have different
        lifetimes. A feature table is rebuilt whenever the feature set changes;
        a backtest result is evidence about a particular set of forecasters and
        is worth keeping after they change.
        """
        return self.data_dir / "reports"


class HttpConfig(_Strict):
    """Outbound request policy, shared by every ingestion adapter."""

    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=5, ge=0)
    backoff_factor: float = Field(default=0.5, ge=0)
    user_agent: str = DEFAULT_USER_AGENT
    """Defaults to the version-derived string in src.utils.http. Overridable,
    but deliberately not repeated in configs/config.yaml: a literal there is a
    second copy of the version that nothing keeps in step."""
    min_request_interval_seconds: float = Field(default=0.0, ge=0)


class ApiConfig(_Strict):
    """How the FastAPI service binds, batches and logs what it served."""

    port: int = Field(default=8000, gt=0, lt=65536)
    """The port the service is served on.

    There is deliberately no ``host``. It existed, was mapped to ``API_HOST``,
    was listed in ``.env.example`` and was read by nothing: the Makefile binds
    ``127.0.0.1`` and the image binds ``0.0.0.0``, both as literals, because a
    container's exposure is controlled by its published port and a bind address
    a health probe can be configured out of is a bind address that will be.
    A setting that appears to be configured and is not is precisely the failure
    ``extra="forbid"`` exists to prevent, so it is gone rather than documented.
    """

    max_batch: int = Field(default=50, gt=0, le=1000)
    """How many fixtures one request may ask for.

    A bound rather than a preference: every fixture in a batch is a row of the
    same design matrix, so the cost is linear and unbounded, and an unbounded
    request is the cheapest denial of service there is. Fifty is a round of
    fixtures across the competitions this project covers.
    """

    prediction_log_dsn: str | None = None
    """PostgreSQL connection string for the served-prediction log.

    ``None`` disables the log and the service runs without a database — the
    clean-checkout and CI case. **Environment only**: it carries a password,
    and ``configs/config.yaml`` is committed. There is deliberately no entry
    for it in that file, so there is no line for someone to fill in by
    accident.
    """


class LoggingConfig(_Strict):
    """Logging verbosity and line format."""

    level: str = "INFO"
    format: str = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

    @field_validator("level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        """Reject an unknown level rather than let ``setLevel`` raise later.

        ``logging.setLevel("INFOO")`` fails deep inside a script's startup with
        a message that does not mention configuration. Failing here names the
        setting and lists the alternatives.
        """
        upper = value.upper()
        if upper not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"unknown log level {value!r}; expected one of {sorted(_VALID_LOG_LEVELS)}"
            )
        return upper


class Settings(_Strict):
    """The fully resolved configuration for one process.

    Sections are added by the milestone that first reads them — ingestion,
    storage and split settings arrive with Milestones 2, 3 and 7. A section
    declared before anything reads it is a setting that looks configurable and
    is not.

    Every section is optional, because every field within one has a default.
    Requiring the header would mean `logging: {}` and omitting `logging:`
    behaved differently, which is a distinction with no meaning behind it —
    and it would make the minimal valid config four empty mappings. An unknown
    top-level section is still an error: `extra="forbid"` is inherited.
    """

    paths: PathsConfig = Field(default_factory=PathsConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _assign(tree: dict[str, Any], dotted: str, value: str) -> None:
    """Set ``dotted`` (e.g. ``"api.port"``) to ``value`` inside ``tree``."""
    head, _, tail = dotted.partition(".")
    if not tail:
        tree[head] = value
        return
    branch = tree.setdefault(head, {})
    if not isinstance(branch, dict):
        raise TypeError(f"cannot descend into {head!r}: not a mapping")
    _assign(branch, tail, value)


def apply_env_overrides(tree: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    """Overlay environment variables onto a parsed config tree.

    Values arrive as strings; Pydantic coerces them to the declared field type
    during validation, so ``API_PORT=9000`` becomes an int and a non-numeric
    value raises rather than silently producing a string port.

    An empty string is treated as unset. A container that declares
    ``API_PORT=""`` to mean "use the default" would otherwise fail validation.

    Args:
        tree: The parsed YAML, mutated in place and also returned.
        env: Environment mapping. Defaults to the real environment; passing one
            explicitly keeps tests independent of the developer's machine.
    """
    source = os.environ if env is None else env
    for variable, dotted in ENV_OVERRIDES.items():
        raw = source.get(variable)
        if raw is not None and raw != "":
            _assign(tree, dotted, raw)
    return tree


def load_settings(
    config_path: Path | None = None,
    *,
    env: dict[str, str] | None = None,
    use_dotenv: bool = True,
) -> Settings:
    """Load, overlay and validate configuration.

    Args:
        config_path: YAML defaults. Falls back to ``configs/config.yaml``.
        env: Environment mapping to overlay. Defaults to the real environment.
        use_dotenv: Load a ``.env`` file first. Disabled in tests so a
            developer's local ``.env`` cannot change the result of a test run.

    Returns:
        A validated, immutable :class:`Settings`.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        pydantic.ValidationError: On an unknown key or an invalid value.
    """
    if use_dotenv and env is None:
        # override=False: a real environment variable outranks the file, which
        # is what a deployment expects when it injects configuration.
        load_dotenv(PROJECT_ROOT / ".env", override=False)

    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        # safe_load, not load: the config file is committed, but a config
        # loader that can construct arbitrary Python objects is a habit that
        # becomes a vulnerability the first time one is read from elsewhere.
        tree = yaml.safe_load(handle) or {}

    return Settings.model_validate(apply_env_overrides(tree, env))
