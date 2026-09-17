"""The compose file publishes nothing beyond the host it runs on."""

from __future__ import annotations

import yaml

from src.utils.paths import PROJECT_ROOT


def test_every_published_port_is_bound_to_localhost() -> None:
    """The API and the dashboard have no authentication and the database has a
    literal password, so an unqualified ``"8501:8501"`` would put all three on
    every network the host joins."""
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text())
    published = [
        str(port) for service in compose["services"].values() for port in service.get("ports", [])
    ]
    assert published
    assert all(port.startswith("127.0.0.1:") for port in published), published
