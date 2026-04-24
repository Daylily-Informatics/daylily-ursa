"""Ursa CLI built on cli-core-yo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from cli_core_yo.app import create_app, run
from cli_core_yo.spec import (
    BackendDetectSpec,
    BackendValidationSpec,
    CliSpec,
    ConfigSpec,
    EnvSpec,
    ExecutionBackendSpec,
    PluginSpec,
    PolicySpec,
    PrereqSpec,
    RuntimeSpec,
    XdgSpec,
)

from daylib_ursa.cli._registry_v2 import URSA_RUNTIME_TAG
from daylib_ursa.integrations.tapdb_runtime import (
    DEFAULT_AWS_PROFILE,
    DEFAULT_AWS_REGION,
    DEFAULT_TAPDB_DATABASE_NAME,
)
from daylib_ursa.config import build_default_config_template
from daylib_ursa.ursa_config import _resolve_deployment_code


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
            getattr(settings, "tapdb_env", None) or "dev",
        )
    )
    rows.append(
        (
            "TapDB Client",
            getattr(settings, "tapdb_client_id", None) or "local",
        )
    )
    rows.append(
        (
            "TapDB Namespace",
            getattr(settings, "tapdb_database_name", None) or DEFAULT_TAPDB_DATABASE_NAME,
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


def _build_spec() -> CliSpec:
    return CliSpec(
        prog_name="ursa",
        app_display_name="Ursa",
        dist_name="daylily-ursa",
        root_help=(
            "Ursa development CLI for beta analysis APIs and integrations.\n\n"
            "Examples:\n"
            "  ursa config init\n"
            "  ursa db build --target local\n"
            "  ursa server start --port 8913\n"
            "  ursa monitor start --config config/workset-monitor-config.yaml --foreground"
        ),
        xdg=XdgSpec(
            app_dir_name=f"ursa-{_resolve_deployment_code()}",
        ),
        policy=PolicySpec(),
        config=ConfigSpec(
            xdg_relative_path=f"ursa-config-{_resolve_deployment_code()}.yaml",
            template_bytes=build_default_config_template(),
            validator=_validate_ursa_config,
        ),
        env=EnvSpec(
            active_env_var="URSA_ACTIVE",
            project_root_env_var="URSA_PROJECT_ROOT",
            activate_script_name="activate <deploy-name>",
            deactivate_script_name="ursa_deactivate",
            preferred_backend="ursa-conda",
        ),
        runtime=RuntimeSpec(
            supported_backends=[
                ExecutionBackendSpec(
                    name="ursa-conda",
                    kind="conda",
                    entry_guidance="source ./activate <deploy-name>",
                    detect=BackendDetectSpec(env_vars=("CONDA_PREFIX",)),
                    validation=BackendValidationSpec(env_vars=("CONDA_PREFIX",)),
                )
            ],
            default_backend="ursa-conda",
            guard_mode="enforced",
            prereqs=[
                PrereqSpec(
                    key="ursa-conda-active-env",
                    kind="env_var",
                    value="CONDA_DEFAULT_ENV",
                    help="Activate Ursa with source ./activate <deploy-name>.",
                    applies_to_backends={"ursa-conda"},
                    tags={URSA_RUNTIME_TAG},
                    success_message="Deployment-scoped conda environment is active.",
                    failure_message=(
                        "Ursa CLI requires an active deployment-scoped conda environment. "
                        "Run `source ./activate <deploy-name>`."
                    ),
                ),
                PrereqSpec(
                    key="ursa-conda-env-name",
                    kind="command_probe",
                    value=(
                        sys.executable,
                        "-c",
                        "import os, sys; env = os.environ.get('CONDA_DEFAULT_ENV', '').strip(); "
                        "sys.exit(0 if env and '-' in env else 1)",
                    ),
                    help="Use a deployment-scoped conda environment such as URSA-local2.",
                    applies_to_backends={"ursa-conda"},
                    tags={URSA_RUNTIME_TAG},
                    success_message="Deployment-scoped conda environment name is valid.",
                    failure_message=(
                        "Ursa CLI requires a deployment-scoped conda environment name with '-'. "
                        "Run `source ./activate <deploy-name>`."
                    ),
                ),
                PrereqSpec(
                    key="ursa-daylily-tapdb",
                    kind="python_import",
                    value="daylily_tapdb",
                    help="Install daylily-tapdb into the active Ursa environment.",
                    applies_to_backends={"ursa-conda"},
                    tags={URSA_RUNTIME_TAG},
                    success_message="Dependency available: daylily-tapdb",
                    failure_message=(
                        "Missing dependency: daylily-tapdb. Re-run `source ./activate <deploy-name>`."
                    ),
                ),
                PrereqSpec(
                    key="ursa-daylily-auth-cognito",
                    kind="python_import",
                    value="daylily_auth_cognito",
                    help="Install daylily-auth-cognito into the active Ursa environment.",
                    applies_to_backends={"ursa-conda"},
                    tags={URSA_RUNTIME_TAG},
                    success_message="Dependency available: daylily-auth-cognito",
                    failure_message=(
                        "Missing dependency: daylily-auth-cognito. "
                        "Re-run `source ./activate <deploy-name>`."
                    ),
                ),
            ],
        ),
        plugins=PluginSpec(
            explicit=[
                "daylib_ursa.cli.db.register",
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


spec = _build_spec()

app = create_app(spec)


def main() -> None:
    raise SystemExit(run(spec))
