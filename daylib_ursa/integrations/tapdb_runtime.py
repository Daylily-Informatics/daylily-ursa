"""TapDB runtime integration for Ursa."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from daylily_tapdb import InstanceFactory, TAPDBConnection, TemplateManager

DEFAULT_AWS_PROFILE = "lsmc"
DEFAULT_AWS_REGION = "us-west-2"
DEFAULT_TAPDB_CLIENT_ID = "local"
DEFAULT_TAPDB_DATABASE_NAME = "ursa"
DEFAULT_TAPDB_DOMAIN_CODE = "Z"
DEFAULT_TAPDB_OWNER_REPO = "ursa"
DEFAULT_TAPDB_LOCAL_DB_PORT = "5588"
DEFAULT_TAPDB_LOCAL_UI_PORT = "8918"

_TARGET_TO_TAPDB_ENV = {
    "local": "dev",
    "aurora": "prod",
    "prod": "prod",
}
_LOCAL_ENGINE_TYPES = {"local", "postgres", "postgresql", "system-service", "pg"}
_AURORA_ENGINE_TYPES = {"aurora", "aurora-postgres", "rds", "rds-aurora"}


class TapDBRuntimeError(RuntimeError):
    """Raised when TapDB runtime setup or invocation fails."""


def _sanitize_deployment_code(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9-]+", "-", (value or "").strip())
    cleaned = cleaned.strip("-")
    return cleaned or "local"


@dataclass(frozen=True)
class TapdbClientBundle:
    connection: TAPDBConnection
    template_manager: TemplateManager
    instance_factory: InstanceFactory


def ensure_tapdb_version() -> str:
    try:
        return importlib.metadata.version("daylily-tapdb")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TapDBRuntimeError("daylily-tapdb is required but not installed.") from exc


def tapdb_env_for_target(
    target: str,
    *,
    config_path: str = "",
    client_id: str = DEFAULT_TAPDB_CLIENT_ID,
    namespace: str = DEFAULT_TAPDB_DATABASE_NAME,
) -> str:
    normalized = (target or "").strip().lower()
    if normalized not in _TARGET_TO_TAPDB_ENV:
        raise TapDBRuntimeError(f"Unsupported database target '{target}'. Use local or aurora.")

    detected = _detect_tapdb_env_for_target(
        normalized,
        config_path=config_path,
        client_id=client_id,
        namespace=namespace,
    )
    if detected:
        return detected
    return _TARGET_TO_TAPDB_ENV[normalized]


def _detect_tapdb_env_for_target(
    target: str,
    *,
    config_path: str,
    client_id: str,
    namespace: str,
) -> str | None:
    try:
        from daylily_tapdb.cli.db_config import get_db_config_for_env
    except Exception:
        return None

    discovered: dict[str, dict[str, str]] = {}
    for env_name in ("dev", "prod"):
        try:
            cfg = (
                get_db_config_for_env(
                    env_name,
                    config_path=config_path or None,
                    client_id=client_id,
                    database_name=namespace,
                )
                or {}
            )
        except Exception:
            continue
        if cfg:
            discovered[env_name] = cfg
    if not discovered:
        return None

    if target == "local":
        for env_name, cfg in discovered.items():
            engine_type = (cfg.get("engine_type") or "").strip().lower()
            if engine_type in _LOCAL_ENGINE_TYPES and _has_required_euid_prefixes(cfg):
                return env_name
        for env_name, cfg in discovered.items():
            host = (cfg.get("host") or "").strip().lower()
            if host in {"localhost", "127.0.0.1", "::1"} and _has_required_euid_prefixes(cfg):
                return env_name
        return None

    if target in {"aurora", "prod"}:
        for env_name, cfg in discovered.items():
            engine_type = (cfg.get("engine_type") or "").strip().lower()
            if engine_type in _AURORA_ENGINE_TYPES:
                return env_name
        for env_name, cfg in discovered.items():
            host = (cfg.get("host") or "").strip().lower()
            if host.endswith(".rds.amazonaws.com"):
                return env_name
        return None
    return None


def _has_required_euid_prefixes(cfg: Mapping[str, str]) -> bool:
    audit_log_prefix = (cfg.get("audit_log_euid_prefix") or "").strip()
    return bool(audit_log_prefix)


def _get_tapdb_db_config_for_env(
    tapdb_env: str,
    *,
    config_path: str,
    client_id: str,
    database_name: str,
) -> dict[str, str]:
    from daylily_tapdb.cli.db_config import get_db_config_for_env

    cfg = get_db_config_for_env(
        tapdb_env,
        config_path=config_path or None,
        client_id=client_id,
        database_name=database_name,
    )
    if not cfg:
        raise TapDBRuntimeError(f"No TapDB database config resolved for TAPDB_ENV={tapdb_env}.")
    return cfg


def _build_sqlalchemy_url(cfg: Mapping[str, str]) -> str:
    user = quote((cfg.get("user") or "").strip(), safe="")
    password = quote((cfg.get("password") or "").strip(), safe="")
    host = (cfg.get("host") or "localhost").strip()
    port = (cfg.get("port") or "5432").strip()
    database = (cfg.get("database") or "").strip()
    if not user:
        user = "postgres"
    if not database:
        raise TapDBRuntimeError("TapDB DB config is missing database name.")
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"postgresql+psycopg2://{auth}{host}:{port}/{database}"


def _resolved_default_identity() -> tuple[str, str, str, str, str, str]:
    try:
        from daylib_ursa.config import get_settings

        settings = get_settings()
        client_id = str(
            os.environ.get("TAPDB_CLIENT_ID") or getattr(settings, "tapdb_client_id", "") or ""
        ).strip()
        namespace = str(
            os.environ.get("TAPDB_DATABASE_NAME")
            or getattr(settings, "tapdb_database_name", "")
            or ""
        ).strip()
        tapdb_env = (
            str(os.environ.get("TAPDB_ENV") or getattr(settings, "tapdb_env", "") or "")
            .strip()
            .lower()
        )
        config_path = str(getattr(settings, "tapdb_config_path", "") or "").strip()
        domain_registry_path = str(
            getattr(settings, "tapdb_domain_registry_path", "") or ""
        ).strip()
        prefix_registry_path = str(
            getattr(settings, "tapdb_prefix_ownership_registry_path", "") or ""
        ).strip()
    except Exception:
        client_id = ""
        namespace = ""
        tapdb_env = ""
        config_path = ""
        domain_registry_path = ""
        prefix_registry_path = ""

    return (
        client_id or DEFAULT_TAPDB_CLIENT_ID,
        namespace or DEFAULT_TAPDB_DATABASE_NAME,
        tapdb_env or "",
        config_path or "",
        domain_registry_path or "",
        prefix_registry_path or "",
    )


def _resolve_runtime_env(
    *,
    target: str,
    client_id: str = DEFAULT_TAPDB_CLIENT_ID,
    profile: str = DEFAULT_AWS_PROFILE,
    region: str = DEFAULT_AWS_REGION,
    namespace: str = DEFAULT_TAPDB_DATABASE_NAME,
    tapdb_env: str | None = None,
    config_path: str = "",
) -> dict[str, str]:
    (
        default_client_id,
        default_namespace,
        default_tapdb_env,
        default_config_path,
        default_domain_registry_path,
        default_prefix_registry_path,
    ) = (
        _resolved_default_identity()
    )
    resolved_client_id = (client_id or default_client_id).strip() or default_client_id
    resolved_namespace = (namespace or default_namespace).strip() or default_namespace
    resolved_cfg_path = str(config_path or default_config_path).strip()
    if not resolved_cfg_path:
        resolved_cfg_path = _resolve_tapdb_config_path(
            namespace=resolved_namespace,
            client_id=resolved_client_id,
            config_path=default_config_path,
        )
    resolved_env = (
        (
            tapdb_env
            or default_tapdb_env
            or tapdb_env_for_target(
                target,
                config_path=resolved_cfg_path or "",
                client_id=resolved_client_id,
                namespace=resolved_namespace,
            )
        )
        .strip()
        .lower()
    )
    return {
        "aws_profile": (profile or DEFAULT_AWS_PROFILE).strip() or DEFAULT_AWS_PROFILE,
        "aws_region": (region or DEFAULT_AWS_REGION).strip() or DEFAULT_AWS_REGION,
        "client_id": resolved_client_id,
        "database_name": resolved_namespace,
        "tapdb_env": resolved_env,
        "config_path": resolved_cfg_path or "",
        "domain_code": DEFAULT_TAPDB_DOMAIN_CODE,
        "owner_repo_name": DEFAULT_TAPDB_OWNER_REPO,
        "domain_registry_path": default_domain_registry_path,
        "prefix_registry_path": default_prefix_registry_path,
    }


def _resolve_tapdb_config_path(
    *,
    namespace: str,
    client_id: str,
    config_path: str = "",
) -> str | None:
    explicit = str(config_path or "").strip()
    if explicit:
        return explicit
    return None


def _require_config_path(runtime_env: Mapping[str, str]) -> str:
    config_path = str(runtime_env.get("config_path") or "").strip()
    if not config_path:
        raise TapDBRuntimeError(
            "TapDB config path is required. Resolve it via Ursa settings and pass it explicitly "
            "to TapDB with --config."
        )
    if not Path(config_path).expanduser().is_absolute():
        raise TapDBRuntimeError(
            f"TapDB config path must be absolute: {config_path!r}. Resolve it via Ursa settings "
            "and pass the absolute path explicitly to TapDB with --config."
        )
    return config_path


def _require_absolute_registry_path(value: str, *, option_name: str) -> str:
    path = str(value or "").strip()
    if not path:
        raise TapDBRuntimeError(
            f"TapDB {option_name} is required. Load it from Ursa settings and pass it explicitly "
            f"to TapDB with {option_name}."
        )
    if not Path(path).expanduser().is_absolute():
        raise TapDBRuntimeError(
            f"TapDB {option_name} must be absolute: {path!r}. Resolve it via Ursa settings and "
            f"pass the absolute path explicitly to TapDB with {option_name}."
        )
    return path


def ensure_local_tapdb_namespace_config(
    *,
    client_id: str = DEFAULT_TAPDB_CLIENT_ID,
    profile: str = DEFAULT_AWS_PROFILE,
    region: str = DEFAULT_AWS_REGION,
    namespace: str = DEFAULT_TAPDB_DATABASE_NAME,
    config_path: str = "",
) -> subprocess.CompletedProcess[str]:
    runtime_env = _resolve_runtime_env(
        target="local",
        client_id=client_id,
        profile=profile,
        region=region,
        namespace=namespace,
        tapdb_env="dev",
        config_path=config_path,
    )
    resolved_config_path = Path(_require_config_path(runtime_env)).expanduser()
    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)

    domain_registry_path = _require_absolute_registry_path(
        runtime_env["domain_registry_path"],
        option_name="domain-registry-path",
    )
    prefix_registry_path = _require_absolute_registry_path(
        runtime_env["prefix_registry_path"],
        option_name="prefix-ownership-registry-path",
    )

    child_env = os.environ.copy()
    child_env["AWS_PROFILE"] = runtime_env["aws_profile"]
    child_env["AWS_REGION"] = runtime_env["aws_region"]
    child_env["AWS_DEFAULT_REGION"] = runtime_env["aws_region"]
    child_env["MERIDIAN_DOMAIN_CODE"] = runtime_env["domain_code"]
    child_env["TAPDB_OWNER_REPO"] = runtime_env["owner_repo_name"]
    child_env.setdefault("PYTHONSAFEPATH", "1")

    init_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "daylily_tapdb.cli",
            "--config",
            str(resolved_config_path),
            "db-config",
            "init",
            "--client-id",
            runtime_env["client_id"],
            "--database-name",
            runtime_env["database_name"],
            "--owner-repo-name",
            runtime_env["owner_repo_name"],
            "--domain-code",
            f"{runtime_env['tapdb_env']}={runtime_env['domain_code']}",
            "--domain-registry-path",
            domain_registry_path,
            "--prefix-ownership-registry-path",
            prefix_registry_path,
            "--env",
            runtime_env["tapdb_env"],
            "--db-port",
            f"{runtime_env['tapdb_env']}={DEFAULT_TAPDB_LOCAL_DB_PORT}",
            "--ui-port",
            f"{runtime_env['tapdb_env']}={DEFAULT_TAPDB_LOCAL_UI_PORT}",
        ],
        env=child_env,
        capture_output=True,
        text=True,
    )
    if init_result.returncode != 0:
        stderr = (init_result.stderr or "").strip()
        stdout = (init_result.stdout or "").strip()
        details = stderr or stdout or "tapdb config init failed without output."
        raise TapDBRuntimeError(f"tapdb db-config init failed: {details}")

    update_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "daylily_tapdb.cli",
            "--config",
            str(resolved_config_path),
            "--client-id",
            runtime_env["client_id"],
            "--database-name",
            runtime_env["database_name"],
            "db-config",
            "update",
            "--env",
            runtime_env["tapdb_env"],
            "--owner-repo-name",
            runtime_env["owner_repo_name"],
            "--domain-code",
            runtime_env["domain_code"],
            "--domain-registry-path",
            domain_registry_path,
            "--prefix-ownership-registry-path",
            prefix_registry_path,
            "--engine-type",
            "local",
            "--host",
            "localhost",
            "--port",
            DEFAULT_TAPDB_LOCAL_DB_PORT,
            "--ui-port",
            DEFAULT_TAPDB_LOCAL_UI_PORT,
            "--database",
            runtime_env["database_name"],
        ],
        env=child_env,
        capture_output=True,
        text=True,
    )
    if update_result.returncode != 0:
        stderr = (update_result.stderr or "").strip()
        stdout = (update_result.stdout or "").strip()
        details = stderr or stdout or "tapdb db-config update failed without output."
        raise TapDBRuntimeError(f"tapdb db-config update failed: {details}")
    return update_result


def export_database_url_for_target(
    *,
    target: str,
    client_id: str = DEFAULT_TAPDB_CLIENT_ID,
    profile: str = DEFAULT_AWS_PROFILE,
    region: str = DEFAULT_AWS_REGION,
    namespace: str = DEFAULT_TAPDB_DATABASE_NAME,
    tapdb_env: str | None = None,
    config_path: str = "",
) -> str:
    ensure_tapdb_version()
    runtime_env = _resolve_runtime_env(
        target=target,
        client_id=client_id,
        profile=profile,
        region=region,
        namespace=namespace,
        tapdb_env=tapdb_env,
        config_path=config_path,
    )
    resolved_config_path = _require_config_path(runtime_env)
    cfg = _get_tapdb_db_config_for_env(
        runtime_env["tapdb_env"],
        config_path=resolved_config_path,
        client_id=runtime_env["client_id"],
        database_name=runtime_env["database_name"],
    )
    return _build_sqlalchemy_url(cfg)


def get_tapdb_bundle(
    *,
    target: str = "local",
    client_id: str = DEFAULT_TAPDB_CLIENT_ID,
    profile: str = DEFAULT_AWS_PROFILE,
    region: str = DEFAULT_AWS_REGION,
    namespace: str = DEFAULT_TAPDB_DATABASE_NAME,
    tapdb_env: str | None = None,
    config_path: str = "",
    app_username: str | None = None,
) -> TapdbClientBundle:
    ensure_tapdb_version()
    runtime_env = _resolve_runtime_env(
        target=target,
        client_id=client_id,
        profile=profile,
        region=region,
        namespace=namespace,
        tapdb_env=tapdb_env,
        config_path=config_path,
    )
    resolved_config_path = _require_config_path(runtime_env)
    cfg = _get_tapdb_db_config_for_env(
        runtime_env["tapdb_env"],
        config_path=resolved_config_path,
        client_id=runtime_env["client_id"],
        database_name=runtime_env["database_name"],
    )
    connection = TAPDBConnection(
        app_username=str(app_username or runtime_env["client_id"]).strip()
        or runtime_env["client_id"],
        db_hostname=f"{cfg.get('host', 'localhost')}:{cfg.get('port', '5432')}",
        db_user=cfg.get("user"),
        db_pass=cfg.get("password", ""),
        db_name=cfg.get("database") or runtime_env["database_name"],
        engine_type=cfg.get("engine_type"),
        region=runtime_env["aws_region"],
        secret_arn=cfg.get("secret_arn"),
        iam_auth=str(cfg.get("iam_auth", "true")).strip().lower() not in {"0", "false", "no"},
        domain_code=runtime_env["domain_code"],
        owner_repo_name=runtime_env["owner_repo_name"],
    )
    template_manager = TemplateManager(Path(resolved_config_path) if resolved_config_path else None)
    instance_factory = InstanceFactory(
        template_manager,
        domain_code=runtime_env["domain_code"],
    )
    return TapdbClientBundle(
        connection=connection,
        template_manager=template_manager,
        instance_factory=instance_factory,
    )


def run_tapdb_cli(
    args: Sequence[str],
    *,
    target: str,
    client_id: str = DEFAULT_TAPDB_CLIENT_ID,
    profile: str = DEFAULT_AWS_PROFILE,
    region: str = DEFAULT_AWS_REGION,
    namespace: str = DEFAULT_TAPDB_DATABASE_NAME,
    tapdb_env: str | None = None,
    config_path: str = "",
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    ensure_tapdb_version()
    runtime_env = _resolve_runtime_env(
        target=target,
        client_id=client_id,
        profile=profile,
        region=region,
        namespace=namespace,
        tapdb_env=tapdb_env,
        config_path=config_path,
    )
    cmd = [
        sys.executable,
        "-m",
        "daylily_tapdb.cli",
        "--config",
        _require_config_path(runtime_env),
        "--env",
        runtime_env["tapdb_env"],
    ]
    cmd.extend(args)

    child_env = os.environ.copy()
    child_env["AWS_PROFILE"] = runtime_env["aws_profile"]
    child_env["AWS_REGION"] = runtime_env["aws_region"]
    child_env["AWS_DEFAULT_REGION"] = runtime_env["aws_region"]
    child_env["MERIDIAN_DOMAIN_CODE"] = runtime_env["domain_code"]
    child_env["TAPDB_OWNER_REPO"] = runtime_env["owner_repo_name"]
    child_env.setdefault("PYTHONSAFEPATH", "1")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=child_env,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or "tapdb command failed without output."
        raise TapDBRuntimeError(f"tapdb {' '.join(args)} failed: {details}")
    return result


def run_schema_drift_check(
    *,
    target: str,
    client_id: str = DEFAULT_TAPDB_CLIENT_ID,
    profile: str = DEFAULT_AWS_PROFILE,
    region: str = DEFAULT_AWS_REGION,
    namespace: str = DEFAULT_TAPDB_DATABASE_NAME,
    tapdb_env: str | None = None,
    config_path: str = "",
    cwd: Path | None = None,
) -> dict[str, object]:
    """Run TapDB schema drift check in report-only mode and normalize the result."""

    env_name = (
        (
            tapdb_env
            or tapdb_env_for_target(
                target,
                config_path=config_path,
                client_id=client_id,
                namespace=namespace,
            )
        )
        .strip()
        .lower()
    )
    tool_version = ensure_tapdb_version()
    result = run_tapdb_cli(
        ["db", "schema", "drift-check", env_name, "--json", "--no-strict"],
        target=target,
        client_id=client_id,
        profile=profile,
        region=region,
        namespace=namespace,
        tapdb_env=env_name,
        config_path=config_path,
        cwd=cwd,
        check=False,
    )

    payload: dict[str, object] = {}
    raw_stdout = (result.stdout or "").strip()
    if raw_stdout:
        try:
            parsed = json.loads(raw_stdout)
        except json.JSONDecodeError:
            parsed = {"raw_stdout": raw_stdout}
        if isinstance(parsed, dict):
            payload = parsed

    status = "check_failed"
    if result.returncode == 0:
        status = "clean"
    elif result.returncode == 1:
        status = "drift"

    counts = payload.get("counts")
    summary = "schema drift report unavailable"
    if isinstance(counts, dict):
        expected = counts.get("expected")
        live = counts.get("live")
        summary = f"expected={expected} live={live}"
    elif status == "clean":
        summary = "no schema drift reported"
    elif status == "drift":
        summary = "schema drift detected"

    normalized: dict[str, object] = {
        "status": status,
        "checked_at": datetime.now(UTC).isoformat(),
        "environment": env_name,
        "tool_version": tool_version,
        "summary": summary,
        "report": payload,
        "strict": False,
    }
    stderr = (result.stderr or "").strip()
    if stderr and status == "check_failed":
        normalized["stderr"] = stderr
    return normalized
