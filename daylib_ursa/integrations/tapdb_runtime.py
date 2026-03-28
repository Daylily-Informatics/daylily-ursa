"""TapDB runtime integration for Ursa."""

from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from daylily_tapdb import InstanceFactory, TAPDBConnection, TemplateManager

TAPDB_REQUIRED_VERSION = "3.0.6"
DEFAULT_AWS_PROFILE = "lsmc"
DEFAULT_AWS_REGION = "us-west-2"
DEFAULT_TAPDB_CLIENT_ID = "local"
DEFAULT_TAPDB_DATABASE_NAME = "ursa"

_TARGET_TO_TAPDB_ENV = {
    "local": "dev",
    "aurora": "prod",
    "prod": "prod",
}
_LOCAL_ENGINE_TYPES = {"local", "postgres", "postgresql", "system-service", "pg"}
_AURORA_ENGINE_TYPES = {"aurora", "aurora-postgres", "rds", "rds-aurora"}


class TapDBRuntimeError(RuntimeError):
    """Raised when TapDB runtime setup or invocation fails."""


@dataclass(frozen=True)
class TapdbClientBundle:
    connection: TAPDBConnection
    template_manager: TemplateManager
    instance_factory: InstanceFactory


def _parse_semver_tuple(version: str) -> tuple[int, int, int] | None:
    match = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", version or "")
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _is_local_tapdb_override_install() -> bool:
    try:
        import daylily_tapdb  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - best effort only
        return False

    module_file = getattr(daylily_tapdb, "__file__", "")
    if not module_file:
        return False
    module_path = Path(module_file).resolve()

    base = Path.cwd().resolve()
    candidates = _local_tapdb_repo_candidates(base)
    return any(
        candidate.exists() and module_path.is_relative_to(candidate) for candidate in candidates
    )


def _local_tapdb_repo_candidates(base: Path) -> list[Path]:
    """Return preferred local TapDB repo candidates for the current workspace."""
    repo_root = Path(__file__).resolve().parents[2]
    bases = [base, repo_root]
    seen: set[Path] = set()
    candidates: list[Path] = []
    for candidate_base in bases:
        for candidate in (
            (candidate_base / "../../daylily/daylily-tapdb").resolve(),
            (candidate_base / "../../daylily/lims_repos/daylily-tapdb").resolve(),
            (candidate_base / "../daylily-tapdb").resolve(),
        ):
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
    git_repos = [candidate for candidate in candidates if (candidate / ".git").exists()]
    other_repos = [
        candidate
        for candidate in candidates
        if candidate.exists() and candidate not in git_repos
    ]
    return [*git_repos, *other_repos]


def ensure_tapdb_version(required_version: str = TAPDB_REQUIRED_VERSION) -> str:
    try:
        installed_version = importlib.metadata.version("daylily-tapdb")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TapDBRuntimeError(
            f"daylily-tapdb=={required_version} is required but not installed."
        ) from exc

    if installed_version != required_version:
        module_version = ""
        try:
            from daylily_tapdb import __version__ as module_version
        except Exception:  # pragma: no cover
            module_version = ""

        required_tuple = _parse_semver_tuple(required_version)
        installed_tuple = _parse_semver_tuple(installed_version)
        allow_local_override = (
            required_tuple is not None
            and installed_tuple is not None
            and installed_tuple >= required_tuple
            and _is_local_tapdb_override_install()
        )
        if module_version != required_version and not allow_local_override:
            raise TapDBRuntimeError(
                "daylily-tapdb version mismatch. "
                f"Required baseline: {required_version}, installed: {installed_version}."
            )
    return installed_version


def tapdb_env_for_target(target: str) -> str:
    normalized = (target or "").strip().lower()
    if normalized not in _TARGET_TO_TAPDB_ENV:
        raise TapDBRuntimeError(f"Unsupported database target '{target}'. Use local or aurora.")

    explicit_override = ""
    if normalized == "local":
        explicit_override = (os.environ.get("URSA_TAPDB_ENV_LOCAL") or "").strip().lower()
    elif normalized in {"aurora", "prod"}:
        explicit_override = (os.environ.get("URSA_TAPDB_ENV_AURORA") or "").strip().lower()
    if explicit_override:
        return explicit_override

    detected = _detect_tapdb_env_for_target(normalized)
    if detected:
        return detected
    return _TARGET_TO_TAPDB_ENV[normalized]


def _detect_tapdb_env_for_target(target: str) -> str | None:
    try:
        from daylily_tapdb.cli.db_config import get_db_config_for_env
    except Exception:
        return None

    discovered: dict[str, dict[str, str]] = {}
    for env_name in ("dev", "prod"):
        try:
            cfg = get_db_config_for_env(env_name) or {}
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


def _get_tapdb_db_config_for_env(tapdb_env: str) -> dict[str, str]:
    from daylily_tapdb.cli.db_config import get_db_config_for_env

    cfg = get_db_config_for_env(tapdb_env)
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


def _resolved_default_identity() -> tuple[str, str, str]:
    try:
        from daylib_ursa.config import get_settings

        settings = get_settings()
        client_id = str(getattr(settings, "tapdb_client_id", "") or "").strip()
        namespace = str(getattr(settings, "tapdb_database_name", "") or "").strip()
        tapdb_env = str(getattr(settings, "tapdb_env", "") or "").strip().lower()
    except Exception:
        client_id = ""
        namespace = ""
        tapdb_env = ""

    return (
        client_id or DEFAULT_TAPDB_CLIENT_ID,
        namespace or DEFAULT_TAPDB_DATABASE_NAME,
        tapdb_env or "",
    )


def _resolve_runtime_env(
    *,
    target: str,
    client_id: str = DEFAULT_TAPDB_CLIENT_ID,
    profile: str = DEFAULT_AWS_PROFILE,
    region: str = DEFAULT_AWS_REGION,
    namespace: str = DEFAULT_TAPDB_DATABASE_NAME,
    tapdb_env: str | None = None,
) -> dict[str, str]:
    default_client_id, default_namespace, default_tapdb_env = _resolved_default_identity()
    resolved_env = (
        tapdb_env
        or default_tapdb_env
        or tapdb_env_for_target(target)
    ).strip().lower()
    env = os.environ.copy()
    env["AWS_PROFILE"] = (profile or DEFAULT_AWS_PROFILE).strip() or DEFAULT_AWS_PROFILE
    env["AWS_REGION"] = (region or DEFAULT_AWS_REGION).strip() or DEFAULT_AWS_REGION
    env["TAPDB_CLIENT_ID"] = (client_id or default_client_id).strip() or default_client_id
    env["TAPDB_DATABASE_NAME"] = (
        namespace or default_namespace
    ).strip() or default_namespace
    env["TAPDB_ENV"] = resolved_env
    env["TAPDB_STRICT_NAMESPACE"] = "1"
    resolved_cfg_path = _resolve_tapdb_config_path(
        namespace=env["TAPDB_DATABASE_NAME"],
        client_id=env["TAPDB_CLIENT_ID"],
    )
    if resolved_cfg_path and not (env.get("TAPDB_CONFIG_PATH") or "").strip():
        env["TAPDB_CONFIG_PATH"] = resolved_cfg_path
    return env


def _resolve_tapdb_config_path(*, namespace: str, client_id: str) -> str | None:
    explicit = (os.environ.get("TAPDB_CONFIG_PATH") or "").strip()
    if explicit:
        return explicit

    default_client_id, default_namespace, _default_tapdb_env = _resolved_default_identity()
    normalized_namespace = (namespace or default_namespace).strip() or default_namespace
    normalized_client_id = (client_id or default_client_id).strip() or default_client_id

    user_scoped = (
        Path.home()
        / ".config"
        / "tapdb"
        / normalized_client_id
        / normalized_namespace
        / "tapdb-config.yaml"
    )
    if user_scoped.exists():
        return str(user_scoped)
    return None


def resolve_tapdb_cli_cwd(cwd: Path | None = None) -> Path | None:
    env_override = (os.environ.get("TAPDB_CLI_CWD") or "").strip()
    if env_override:
        override_path = Path(env_override).expanduser().resolve()
        if override_path.exists():
            return override_path

    base = cwd.resolve() if cwd else Path.cwd().resolve()
    candidates = _local_tapdb_repo_candidates(base)
    for candidate in candidates:
        schema_file = candidate / "schema" / "tapdb_schema.sql"
        if candidate.exists() and schema_file.exists():
            return candidate
    return cwd


def export_database_url_for_target(
    *,
    target: str,
    client_id: str = DEFAULT_TAPDB_CLIENT_ID,
    profile: str = DEFAULT_AWS_PROFILE,
    region: str = DEFAULT_AWS_REGION,
    namespace: str = DEFAULT_TAPDB_DATABASE_NAME,
    tapdb_env: str | None = None,
) -> str:
    ensure_tapdb_version()
    runtime_env = _resolve_runtime_env(
        target=target,
        client_id=client_id,
        profile=profile,
        region=region,
        namespace=namespace,
        tapdb_env=tapdb_env,
    )
    os.environ["AWS_PROFILE"] = runtime_env["AWS_PROFILE"]
    os.environ["AWS_REGION"] = runtime_env["AWS_REGION"]
    os.environ["TAPDB_CLIENT_ID"] = runtime_env["TAPDB_CLIENT_ID"]
    os.environ["TAPDB_DATABASE_NAME"] = runtime_env["TAPDB_DATABASE_NAME"]
    os.environ["TAPDB_ENV"] = runtime_env["TAPDB_ENV"]
    os.environ["TAPDB_STRICT_NAMESPACE"] = runtime_env["TAPDB_STRICT_NAMESPACE"]
    if (runtime_env.get("TAPDB_CONFIG_PATH") or "").strip():
        os.environ["TAPDB_CONFIG_PATH"] = runtime_env["TAPDB_CONFIG_PATH"]
    cfg = _get_tapdb_db_config_for_env(runtime_env["TAPDB_ENV"])
    db_url = _build_sqlalchemy_url(cfg)
    os.environ["DATABASE_URL"] = db_url
    return db_url


def get_tapdb_bundle(
    *,
    target: str = "local",
    client_id: str = DEFAULT_TAPDB_CLIENT_ID,
    profile: str = DEFAULT_AWS_PROFILE,
    region: str = DEFAULT_AWS_REGION,
    namespace: str = DEFAULT_TAPDB_DATABASE_NAME,
    tapdb_env: str | None = None,
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
    )
    os.environ["AWS_PROFILE"] = runtime_env["AWS_PROFILE"]
    os.environ["AWS_REGION"] = runtime_env["AWS_REGION"]
    os.environ["TAPDB_CLIENT_ID"] = runtime_env["TAPDB_CLIENT_ID"]
    os.environ["TAPDB_DATABASE_NAME"] = runtime_env["TAPDB_DATABASE_NAME"]
    os.environ["TAPDB_ENV"] = runtime_env["TAPDB_ENV"]
    os.environ["TAPDB_STRICT_NAMESPACE"] = runtime_env["TAPDB_STRICT_NAMESPACE"]
    config_path = (runtime_env.get("TAPDB_CONFIG_PATH") or "").strip()
    if config_path:
        os.environ["TAPDB_CONFIG_PATH"] = config_path

    cfg = _get_tapdb_db_config_for_env(runtime_env["TAPDB_ENV"])
    connection = TAPDBConnection(
        app_username=str(app_username or runtime_env["TAPDB_CLIENT_ID"]).strip()
        or runtime_env["TAPDB_CLIENT_ID"],
        db_hostname=f"{cfg.get('host', 'localhost')}:{cfg.get('port', '5432')}",
        db_user=cfg.get("user"),
        db_pass=cfg.get("password", ""),
        db_name=cfg.get("database") or runtime_env["TAPDB_DATABASE_NAME"],
        engine_type=cfg.get("engine_type"),
        region=runtime_env["AWS_REGION"],
        secret_arn=cfg.get("secret_arn"),
        iam_auth=str(cfg.get("iam_auth", "true")).strip().lower() not in {"0", "false", "no"},
    )
    template_manager = TemplateManager(Path(config_path) if config_path else None)
    instance_factory = InstanceFactory(template_manager)
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
    )
    runtime_env.setdefault("PYTHONSAFEPATH", "1")
    cmd = [
        sys.executable,
        "-m",
        "daylily_tapdb.cli",
        "--client-id",
        runtime_env["TAPDB_CLIENT_ID"],
        "--database-name",
        runtime_env["TAPDB_DATABASE_NAME"],
    ]
    cmd.extend(args)

    resolved_cwd = resolve_tapdb_cli_cwd(cwd)
    result = subprocess.run(
        cmd,
        cwd=resolved_cwd,
        env=runtime_env,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or "tapdb command failed without output."
        raise TapDBRuntimeError(f"tapdb {' '.join(args)} failed: {details}")
    return result
