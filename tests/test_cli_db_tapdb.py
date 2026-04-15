"""Unit tests for ursa db TapDB orchestration."""

from __future__ import annotations

from types import SimpleNamespace
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from daylib_ursa.cli import app
from daylib_ursa.cli import db as db_cli


def _tapdb_dependency_spec() -> str:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    for dependency in pyproject["project"]["dependencies"]:
        if dependency.startswith("daylily-tapdb"):
            return dependency
    raise AssertionError("daylily-tapdb dependency missing from pyproject.toml")


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        database_target="local",
        aws_profile="lsmc",
        tapdb_database_name="ursa",
        tapdb_client_id="local",
        tapdb_config_path="/tmp/ursa-tapdb.yaml",
        get_effective_region=lambda: "us-west-2",
    )


def test_build_local_uses_tapdb_bootstrap_then_overlay(monkeypatch):
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(db_cli, "ensure_tapdb_version", lambda: _tapdb_dependency_spec())
    monkeypatch.setattr(db_cli, "get_settings", _settings)
    monkeypatch.setattr(
        db_cli,
        "run_tapdb_cli",
        lambda **kwargs: (
            events.append(("tapdb", kwargs["args"]))
            or SimpleNamespace(stdout="", stderr="", returncode=0)
        ),
    )
    monkeypatch.setattr(
        db_cli,
        "export_database_url_for_target",
        lambda **_kwargs: (
            events.append(("db_url", "resolved"))
            or "postgresql+psycopg2://ursa@localhost:5432/ursa_dev"
        ),
    )
    monkeypatch.setattr(
        db_cli,
        "_apply_ursa_overlay",
        lambda *, start_step, total_steps: events.append(("overlay", (start_step, total_steps))),
    )

    db_cli.build(
        target="local",
        cluster="",
        region="us-west-2",
        profile="lsmc",
        namespace="ursa",
    )

    assert events == [
        ("tapdb", ["bootstrap", "local", "--no-gui"]),
        ("db_url", "resolved"),
        ("overlay", (3, 3)),
    ]


def test_build_aurora_requires_cluster(monkeypatch):
    monkeypatch.setattr(db_cli, "ensure_tapdb_version", lambda: _tapdb_dependency_spec())
    monkeypatch.setattr(db_cli, "get_settings", _settings)

    with pytest.raises(db_cli.typer.Exit) as exc_info:
        db_cli.build(
            target="aurora",
            cluster="",
            region="us-west-2",
            profile="lsmc",
            namespace="ursa",
        )

    assert exc_info.value.exit_code == 1


def test_reset_uses_tapdb_delete_then_bootstrap_then_overlay(monkeypatch):
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(db_cli, "get_settings", _settings)
    monkeypatch.setattr(
        db_cli,
        "run_tapdb_cli",
        lambda **kwargs: (
            events.append(("tapdb", kwargs["args"]))
            or SimpleNamespace(stdout="", stderr="", returncode=0)
        ),
    )
    monkeypatch.setattr(
        db_cli,
        "export_database_url_for_target",
        lambda **_kwargs: (
            events.append(("db_url", "resolved"))
            or "postgresql+psycopg2://ursa@localhost:5432/ursa_dev"
        ),
    )
    monkeypatch.setattr(
        db_cli,
        "_apply_ursa_overlay",
        lambda *, start_step, total_steps: events.append(("overlay", (start_step, total_steps))),
    )

    db_cli.reset(
        force=True,
        target="local",
        cluster="",
        region="us-west-2",
        profile="lsmc",
        namespace="ursa",
    )

    assert events == [
        ("tapdb", ["db", "delete", "dev", "--force"]),
        ("tapdb", ["bootstrap", "local", "--no-gui"]),
        ("db_url", "resolved"),
        ("overlay", (4, 4)),
    ]


def test_seed_prepares_database_url_before_overlay(monkeypatch):
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(db_cli, "get_settings", _settings)
    monkeypatch.setattr(
        db_cli,
        "export_database_url_for_target",
        lambda **_kwargs: (
            events.append(("db_url", "resolved"))
            or "postgresql+psycopg2://ursa@localhost:5432/ursa_dev"
        ),
    )
    monkeypatch.setattr(
        db_cli,
        "_apply_ursa_overlay",
        lambda *, start_step, total_steps: events.append(("overlay", (start_step, total_steps))),
    )

    db_cli.seed(
        target="local",
        region="us-west-2",
        profile="lsmc",
        namespace="ursa",
    )

    assert events == [
        ("db_url", "resolved"),
        ("overlay", (1, 1)),
    ]


def test_nuke_routes_destructive_action_through_tapdb(monkeypatch):
    tapdb_calls: list[list[str]] = []

    monkeypatch.setattr(db_cli, "get_settings", _settings)
    monkeypatch.setattr(
        db_cli,
        "run_tapdb_cli",
        lambda **kwargs: (
            tapdb_calls.append(kwargs["args"])
            or SimpleNamespace(stdout="", stderr="", returncode=0)
        ),
    )

    db_cli.nuke(
        force=True,
        target="local",
        region="us-west-2",
        profile="lsmc",
        namespace="ursa",
    )

    assert tapdb_calls == [["db", "delete", "dev", "--force"]]


def test_build_rejects_invalid_target(monkeypatch):
    monkeypatch.setattr(db_cli, "ensure_tapdb_version", lambda: _tapdb_dependency_spec())
    monkeypatch.setattr(db_cli, "get_settings", _settings)

    with pytest.raises(db_cli.typer.Exit) as exc_info:
        db_cli.build(
            target="bogus",
            cluster="",
            region="us-west-2",
            profile="lsmc",
            namespace="ursa",
        )

    assert exc_info.value.exit_code == 1


def test_ursa_cli_db_help_exposes_nuke():
    runner = CliRunner()
    result = runner.invoke(app, ["db", "--help"])

    assert result.exit_code == 0
    assert "build" in result.output
    assert "seed" in result.output
    assert "reset" in result.output
    assert "nuke" in result.output
