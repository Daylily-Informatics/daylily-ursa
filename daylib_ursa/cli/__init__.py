"""Ursa CLI built on cli-core-yo."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from cli_core_yo.app import create_app, run
from cli_core_yo.spec import CliSpec, ConfigSpec, EnvSpec, PluginSpec, XdgSpec

from daylib_ursa.integrations.tapdb_runtime import (
    DEFAULT_AWS_PROFILE,
    DEFAULT_AWS_REGION,
    DEFAULT_TAPDB_DATABASE_NAME,
    TapDBRuntimeError,
    ensure_tapdb_version,
)


def _validate_ursa_config(content: str) -> list[str]:
    try:
        config = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        return [f"YAML parse error: {exc}"]
    if config is None:
        return []
    if not isinstance(config, dict):
        return ["Root YAML object must be a mapping"]
    return []


def _ursa_info_hook() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    project_root = Path(__file__).resolve().parents[2]
    rows.append(("Project Root", str(project_root)))

    try:
        from daylib_ursa.config import get_settings

        settings = get_settings()
    except Exception:
        settings = None

    rows.append(
        (
            "AWS Profile",
            os.environ.get("AWS_PROFILE")
            or getattr(settings, "aws_profile", None)
            or DEFAULT_AWS_PROFILE,
        )
    )
    rows.append(
        (
            "AWS Region",
            os.environ.get("AWS_REGION")
            or getattr(settings, "cognito_region", None)
            or DEFAULT_AWS_REGION,
        )
    )
    rows.append(
        (
            "TapDB Env",
            os.environ.get("TAPDB_ENV") or getattr(settings, "tapdb_env", None) or "dev",
        )
    )
    rows.append(
        (
            "TapDB Client",
            os.environ.get("TAPDB_CLIENT_ID")
            or getattr(settings, "tapdb_client_id", None)
            or "local",
        )
    )
    rows.append(
        (
            "TapDB Namespace",
            os.environ.get("TAPDB_DATABASE_NAME")
            or getattr(settings, "tapdb_database_name", None)
            or DEFAULT_TAPDB_DATABASE_NAME,
        )
    )
    if settings is not None:
        rows.append(("Bloom URL", settings.bloom_base_url))
        rows.append(("Atlas URL", settings.atlas_base_url))

    try:
        from cli_core_yo.runtime import get_context

        state_dir = get_context().xdg_paths.state
    except Exception:
        state_dir = Path.home() / ".config" / "ursa"

    pid_file = state_dir / "server.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            rows.append(("Dev Server", f"Running (PID {pid})"))
        except (ValueError, ProcessLookupError, PermissionError):
            rows.append(("Dev Server", "Stopped"))
    else:
        rows.append(("Dev Server", "Stopped"))

    return rows


def _ensure_tapdb_dependency() -> None:
    try:
        ensure_tapdb_version()
    except TapDBRuntimeError as exc:
        raise SystemExit(
            "Ursa CLI startup failed. "
            "Install daylily-tapdb==3.0.6 or use the supported local editable override. "
            f"Details: {exc}"
        ) from exc


_ensure_tapdb_dependency()

_CONFIG_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "config" / "ursa-config.example.yaml"
).read_bytes()


spec = CliSpec(
    prog_name="ursa",
    app_display_name="Ursa",
    dist_name="daylily-ursa",
    root_help="Ursa development CLI for beta analysis APIs and integrations.",
    xdg=XdgSpec(
        app_dir_name="ursa",
    ),
    config=ConfigSpec(
        primary_filename="ursa-config.yaml",
        template_bytes=_CONFIG_TEMPLATE,
        validator=_validate_ursa_config,
    ),
    env=EnvSpec(
        active_env_var="URSA_ACTIVE",
        project_root_env_var="URSA_PROJECT_ROOT",
        activate_script_name="activate",
        deactivate_script_name="ursa_deactivate",
    ),
    plugins=PluginSpec(
        explicit=[
            "daylib_ursa.cli.server.register",
            "daylib_ursa.cli.env.register",
            "daylib_ursa.cli.test.register",
            "daylib_ursa.cli.quality.register",
            "daylib_ursa.cli.integrations.register",
            "daylib_ursa.cli.monitor.register",
        ],
    ),
    info_hooks=[_ursa_info_hook],
)

app = create_app(spec)


def main() -> None:
    raise SystemExit(run(spec))
