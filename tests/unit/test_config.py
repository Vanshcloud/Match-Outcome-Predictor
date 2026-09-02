"""Configuration is the layer where a mistake looks like a working setting.

These tests cover the three failure modes that matter: a typo that is silently
ignored, an override that goes the wrong way, and a value that is invalid but
only fails much later somewhere unrelated.

Every test passes `env` explicitly and `use_dotenv=False`, so nothing here
depends on the developer's machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.utils.config import (
    ENV_OVERRIDES,
    Settings,
    apply_env_overrides,
    load_settings,
)
from src.utils.paths import PROJECT_ROOT


def load(path: Path, env: dict[str, str] | None = None) -> Settings:
    return load_settings(path, env=env or {}, use_dotenv=False)


# ---- defaults ---------------------------------------------------------------


def test_committed_config_loads(config_file: Path) -> None:
    settings = load(config_file)
    assert settings.api.port == 8000
    assert settings.logging.level == "INFO"


def test_the_real_config_file_is_valid() -> None:
    """configs/config.yaml is committed and shipped. If it does not validate,
    every entry point fails at startup — so the file itself is under test, not
    just the loader."""
    settings = load_settings(env={}, use_dotenv=False)
    assert settings.paths.data_dir.is_absolute()


def test_omitted_sections_fall_back_to_field_defaults(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("paths: {}\nhttp: {}\napi: {}\nlogging: {}\n", encoding="utf-8")
    settings = load(path)
    assert settings.http.max_retries == 5
    assert settings.paths.data_dir == PROJECT_ROOT / "data"


# ---- paths ------------------------------------------------------------------


def test_relative_paths_resolve_against_the_project_root(config_file: Path) -> None:
    settings = load(config_file)
    assert settings.paths.data_dir == PROJECT_ROOT / "data"


def test_absolute_paths_survive(tmp_path: Path) -> None:
    """A container mounts a volume and points DATA_DIR at it."""
    path = tmp_path / "c.yaml"
    path.write_text(
        f"paths:\n  data_dir: {tmp_path}\nhttp: {{}}\napi: {{}}\nlogging: {{}}\n",
        encoding="utf-8",
    )
    assert load(path).paths.data_dir == tmp_path


def test_subdirectories_are_derived_from_data_dir(config_file: Path) -> None:
    """They are properties, not settings, so they cannot be pointed at a tree
    the rest of the repo does not know how to find."""
    paths = load(config_file).paths
    assert paths.raw_dir == paths.data_dir / "raw"
    assert paths.processed_dir == paths.data_dir / "processed"
    assert paths.features_dir == paths.data_dir / "features"


# ---- environment overlay ----------------------------------------------------


def test_environment_outranks_the_file(config_file: Path) -> None:
    """The direction that lets a deployment override a committed default."""
    settings = load(config_file, {"API_PORT": "9000", "LOG_LEVEL": "DEBUG"})
    assert settings.api.port == 9000
    assert settings.logging.level == "DEBUG"


def test_override_values_are_coerced_to_the_declared_type(config_file: Path) -> None:
    """Environment values are strings. A string port would be accepted by
    anything that only formats it, and rejected by anything that binds it."""
    assert isinstance(load(config_file, {"API_PORT": "9000"}).api.port, int)


def test_a_non_numeric_port_is_rejected_at_load(config_file: Path) -> None:
    with pytest.raises(ValidationError):
        load(config_file, {"API_PORT": "not-a-port"})


def test_an_empty_override_is_treated_as_unset(config_file: Path) -> None:
    """A container that declares API_PORT="" means "use the default", and must
    not fail validation for saying so."""
    assert load(config_file, {"API_PORT": ""}).api.port == 8000


def test_an_unmapped_variable_is_ignored(config_file: Path) -> None:
    """Only ENV_OVERRIDES is configurable. Anything else is somebody else's
    environment variable, and reading it would be a surprise."""
    assert load(config_file, {"PORT": "9999"}).api.port == 8000


def test_every_override_targets_a_real_field(config_file: Path) -> None:
    """Guards the one thing an explicit map gets wrong: a dotted path that no
    longer matches the model after a rename. Without this the variable would
    silently stop working, since extra="forbid" is checked on the section, not
    on a path that never reaches one."""
    settings = load(config_file)
    for dotted in ENV_OVERRIDES.values():
        section_name, _, field = dotted.partition(".")
        section = getattr(settings, section_name)
        assert field in type(section).model_fields, f"{dotted} targets no field"


def test_overlay_creates_missing_intermediate_sections() -> None:
    tree: dict[str, object] = {}
    apply_env_overrides(tree, {"API_PORT": "9000"})
    assert tree == {"api": {"port": "9000"}}


def test_overlay_refuses_to_descend_into_a_scalar() -> None:
    """Malformed YAML — `api: 5` — must name the problem rather than raise an
    AttributeError three frames deeper."""
    with pytest.raises(TypeError, match="cannot descend"):
        apply_env_overrides({"api": 5}, {"API_PORT": "9000"})


# ---- validation -------------------------------------------------------------


def test_an_unknown_key_is_an_error_not_a_shrug(tmp_path: Path) -> None:
    """The failure mode this guards: a setting that appears configured, is
    misspelled, and is silently ignored forever."""
    path = tmp_path / "c.yaml"
    path.write_text(
        "paths: {}\nhttp: {}\napi:\n  prot: 8000\nlogging: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="prot"):
        load(path)


def test_an_unknown_log_level_is_rejected_by_name(config_file: Path) -> None:
    with pytest.raises(ValidationError, match="unknown log level"):
        load(config_file, {"LOG_LEVEL": "VERBOSE"})


@pytest.mark.parametrize(
    ("variable", "value"),
    [("API_PORT", "0"), ("API_PORT", "70000")],
)
def test_out_of_range_ports_are_rejected(config_file: Path, variable: str, value: str) -> None:
    with pytest.raises(ValidationError):
        load(config_file, {variable: value})


def test_settings_are_frozen(config_file: Path) -> None:
    """A mutable config is one that gets mutated somewhere unhelpful, after
    which the value a module reads depends on import order."""
    settings = load(config_file)
    with pytest.raises(ValidationError):
        settings.api.port = 1234  # type: ignore[misc]


def test_a_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "absent.yaml")


def test_an_empty_config_file_uses_defaults(tmp_path: Path) -> None:
    """safe_load returns None for an empty file; the `or {}` handles it."""
    path = tmp_path / "c.yaml"
    path.write_text("", encoding="utf-8")
    assert load(path).api.port == 8000


def test_defaulted_paths_are_still_resolved(tmp_path: Path) -> None:
    """Regression. Pydantic skips validators on a defaulted field unless
    `validate_default=True` is set, so `data_dir` came back as the relative
    `Path("data")` when the section was omitted and as an absolute path when it
    was named. Everything downstream would then depend on the working
    directory — silently, and only for configs that leave the section out.
    """
    path = tmp_path / "c.yaml"
    path.write_text("", encoding="utf-8")
    paths = load(path).paths
    assert paths.data_dir.is_absolute()
    assert paths.model_dir.is_absolute()
    assert paths.raw_dir == PROJECT_ROOT / "data" / "raw"


# ---- .env loading -----------------------------------------------------------


def test_dotenv_is_read_when_no_env_is_injected(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one path the rest of this module deliberately avoids.

    Every other test injects `env` so the developer's own .env cannot change a
    result. This one asserts the production path still consults the file — and
    with `override=False`, so a real environment variable outranks it. That is
    the direction a deployment needs: inject a variable, and the committed file
    does not win.
    """
    calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        "src.utils.config.load_dotenv",
        lambda path, override: calls.append((path, override)),
    )
    load_settings(config_file, use_dotenv=True)

    assert len(calls) == 1
    path, override = calls[0]
    assert path == PROJECT_ROOT / ".env"
    assert override is False


def test_dotenv_is_skipped_when_env_is_injected(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise every test in the suite would depend on whether the developer
    happens to have a .env file, which is the definition of a flaky test."""
    calls: list[object] = []
    monkeypatch.setattr(
        "src.utils.config.load_dotenv",
        lambda path, override: calls.append((path, override)),
    )
    load_settings(config_file, env={}, use_dotenv=True)
    assert calls == []
