from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
import importlib.util
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_DAYLILY_EC_BIN_ENV = "URSA_DAYLILY_EC_BIN"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _job_base_dir() -> Path:
    return Path.home() / ".ursa" / "cluster-create"


def _jobs_dir() -> Path:
    return _job_base_dir() / "jobs"


def _logs_dir() -> Path:
    return _job_base_dir() / "logs"


def _configs_dir() -> Path:
    return _job_base_dir() / "configs"


@dataclass(frozen=True)
class ClusterCreateJob:
    job_id: str
    cluster_name: str
    region_az: str
    aws_profile: Optional[str]
    job_path: Path
    log_path: Path
    status: str


def resolve_daylily_ec() -> Path:
    """Resolve the daylily-ec CLI binary used to create ephemeral clusters.

    Resolution order:
    1) Explicit override: `URSA_DAYLILY_EC_BIN=/abs/path/to/daylily-ec`
    2) `daylily-ec` available on PATH
    """
    override = os.environ.get(_DAYLILY_EC_BIN_ENV, "").strip()
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"{_DAYLILY_EC_BIN_ENV} points to a missing file: {p}")

    exe = shutil.which("daylily-ec")
    if exe:
        return Path(exe)

    raise FileNotFoundError("daylily-ec console script not found on PATH")


def resolve_daylily_ec_command_prefix() -> List[str]:
    """Resolve command prefix for invoking daylily-ec.

    Prefers the console script, but falls back to `python -m daylily_ec`
    when the package is installed without script shims.
    """
    try:
        return [str(resolve_daylily_ec())]
    except FileNotFoundError:
        if importlib.util.find_spec("daylily_ec") is not None:
            return [sys.executable, "-m", "daylily_ec"]

        raise FileNotFoundError(
            "daylily-ec not found.\n"
            "Install dependency `daylily-ephemeral-cluster==0.7.605` in this environment,\n"
            "or set the explicit binary path via:\n"
            f"  {_DAYLILY_EC_BIN_ENV}=/abs/path/to/daylily-ec"
        )


def write_generated_ec_config(
    *,
    dest: Path,
    cluster_name: str,
    ssh_key_name: str,
    s3_bucket_name: str,
    contact_email: Optional[str],
) -> Path:
    """Write a minimal daylily-ec config YAML suitable for --non-interactive create."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = {
        "ephemeral_cluster": {
            "config": {
                "cluster_name": ["USESETVALUE", "", cluster_name],
                "ssh_key_name": ["USESETVALUE", "", ssh_key_name],
                "s3_bucket_name": ["USESETVALUE", "", s3_bucket_name],
                # If omitted, daylily-ec defaults to "true" which is surprising for portal users.
                "enforce_budget": ["USESETVALUE", "", "skip"],
            }
        }
    }

    if contact_email:
        cfg["ephemeral_cluster"]["config"]["budget_email"] = [  # type: ignore[index]
            "USESETVALUE",
            "",
            contact_email,
        ]

    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return dest


def _build_create_command(
    *,
    daylily_ec_command_prefix: List[str],
    region_az: str,
    aws_profile: Optional[str],
    config_path: Path,
    pass_on_warn: bool,
    debug: bool,
) -> List[str]:
    cmd: List[str] = [
        *daylily_ec_command_prefix,
        "create",
        "--region-az",
        region_az,
        "--config",
        str(config_path),
        "--non-interactive",
    ]
    if aws_profile:
        cmd.extend(["--profile", aws_profile])
    if pass_on_warn:
        cmd.append("--pass-on-warn")
    if debug:
        cmd.append("--debug")
    return cmd


def start_create_job(
    *,
    region_az: str,
    cluster_name: str,
    ssh_key_name: str,
    s3_bucket_name: str,
    aws_profile: Optional[str],
    contact_email: Optional[str],
    config_path_override: Optional[str],
    pass_on_warn: bool,
    debug: bool,
) -> ClusterCreateJob:
    """Start a background cluster create job and return job metadata."""
    daylily_ec_command_prefix = resolve_daylily_ec_command_prefix()

    job_id = f"ec_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    jobs_dir = _jobs_dir()
    logs_dir = _logs_dir()
    configs_dir = _configs_dir()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"{job_id}.log"
    job_path = jobs_dir / f"{job_id}.json"

    if config_path_override:
        cfg_path = Path(config_path_override).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
    else:
        cfg_path = write_generated_ec_config(
            dest=(configs_dir / f"{job_id}.yaml"),
            cluster_name=cluster_name,
            ssh_key_name=ssh_key_name,
            s3_bucket_name=s3_bucket_name,
            contact_email=contact_email,
        )

    cmd = _build_create_command(
        daylily_ec_command_prefix=daylily_ec_command_prefix,
        region_az=region_az,
        aws_profile=aws_profile,
        config_path=cfg_path,
        pass_on_warn=pass_on_warn,
        debug=debug,
    )

    env_overrides: Dict[str, str] = {"PYTHONUNBUFFERED": "1"}
    if aws_profile:
        env_overrides["AWS_PROFILE"] = aws_profile
    if contact_email:
        env_overrides["DAY_CONTACT_EMAIL"] = contact_email

    job_doc: Dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _now_iso(),
        "started_at": None,
        "completed_at": None,
        "return_code": None,
        "error": None,
        "cluster_name": cluster_name,
        "region_az": region_az,
        "aws_profile": aws_profile,
        "config_path": str(cfg_path),
        "log_path": str(log_path),
        "command": cmd,
        "env_overrides": env_overrides,
        "runner_pid": None,
    }
    _atomic_write_json(job_path, job_doc)

    proc = subprocess.Popen(
        [sys.executable, "-m", "daylily_ursa.ephemeral_cluster.job_runner", "--job-file", str(job_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        cwd=Path.cwd(),
        env={**os.environ.copy(), **env_overrides},
    )

    # Best-effort: record runner pid and transition to running.
    job_doc["runner_pid"] = proc.pid
    job_doc["status"] = "running"
    job_doc["started_at"] = _now_iso()
    _atomic_write_json(job_path, job_doc)

    return ClusterCreateJob(
        job_id=job_id,
        cluster_name=cluster_name,
        region_az=region_az,
        aws_profile=aws_profile,
        job_path=job_path,
        log_path=log_path,
        status="running",
    )


def run_create_sync(
    *,
    region_az: str,
    aws_profile: Optional[str],
    config_path: str,
    pass_on_warn: bool,
    debug: bool,
    contact_email: Optional[str],
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    """Run daylily-ec synchronously (for monitor usage)."""
    daylily_ec_command_prefix = resolve_daylily_ec_command_prefix()
    cfg_path = Path(config_path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()

    cmd = _build_create_command(
        daylily_ec_command_prefix=daylily_ec_command_prefix,
        region_az=region_az,
        aws_profile=aws_profile,
        config_path=cfg_path,
        pass_on_warn=pass_on_warn,
        debug=debug,
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if aws_profile:
        env["AWS_PROFILE"] = aws_profile
    if contact_email:
        env["DAY_CONTACT_EMAIL"] = contact_email

    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
    )


def list_cluster_create_jobs(*, limit: int = 20) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    d = _jobs_dir()
    if not d.exists():
        return []
    for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            jobs.append(_read_json(p))
        except Exception:
            continue
        if len(jobs) >= limit:
            break
    return jobs


def read_cluster_create_job(job_id: str) -> Dict[str, Any]:
    job_path = _jobs_dir() / f"{job_id}.json"
    if not job_path.is_file():
        raise FileNotFoundError(f"Job not found: {job_id}")
    return _read_json(job_path)


def tail_job_log(job_id: str, *, lines: int = 200) -> str:
    job = read_cluster_create_job(job_id)
    log_path = Path(str(job.get("log_path", "")))
    if not log_path.is_file():
        return ""
    return _tail_file(log_path, lines=lines)


def _tail_file(path: Path, *, lines: int) -> str:
    # Simple tail: read whole file when small, otherwise seek from end.
    if lines <= 0:
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(data.splitlines()[-lines:])
    except Exception:
        return ""
