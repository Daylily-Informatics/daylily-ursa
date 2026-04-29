from __future__ import annotations

from importlib import import_module
from importlib import metadata as importlib_metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, cast


DAYLILY_EC_DISTRIBUTION = "daylily-ephemeral-cluster"
REQUIRED_DAYLILY_EC_VERSION = "2.1.11"


def require_daylily_ec_version() -> str:
    """Require the installed daylily-ec distribution to match Ursa's contract."""

    try:
        installed = importlib_metadata.version(DAYLILY_EC_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"{DAYLILY_EC_DISTRIBUTION} is not installed. Install "
            f"{DAYLILY_EC_DISTRIBUTION}=={REQUIRED_DAYLILY_EC_VERSION} in the active Ursa environment."
        ) from exc
    if installed != REQUIRED_DAYLILY_EC_VERSION:
        raise RuntimeError(
            f"{DAYLILY_EC_DISTRIBUTION} version mismatch: expected "
            f"{REQUIRED_DAYLILY_EC_VERSION}, found {installed}."
        )
    return installed


DAYEC_CLUSTER_CONFIG_FIELDS = (
    "s3_bucket_name",
    "public_subnet_id",
    "private_subnet_id",
    "iam_policy_arn",
    "cluster_name",
    "budget_email",
    "allowed_budget_users",
    "budget_amount",
    "global_allowed_budget_users",
    "global_budget_amount",
    "enforce_budget",
    "cluster_template_yaml",
    "headnode_instance_type",
    "fsx_fs_size",
    "enable_detailed_monitoring",
    "delete_local_root",
    "auto_delete_fsx",
    "heartbeat_email",
    "heartbeat_schedule",
    "heartbeat_scheduler_role_arn",
    "spot_instance_allocation_strategy",
    "max_count_8I",
    "max_count_128I",
    "max_count_192I",
)


def _command_env(
    *,
    aws_profile: Optional[str],
    contact_email: Optional[str] = None,
    extra_env: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if aws_profile:
        env["AWS_PROFILE"] = aws_profile
    if contact_email:
        env["DAY_CONTACT_EMAIL"] = contact_email
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    return env


def _summarize_process_output(
    result: subprocess.CompletedProcess[str], *, max_chars: int = 4000
) -> str:
    output = (result.stderr or "").strip() or (result.stdout or "").strip()
    if not output:
        return f"exit code {result.returncode}"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    tail = "\n".join(lines[-25:]) if lines else output
    if len(tail) > max_chars:
        return tail[-max_chars:]
    return tail


class DaylilyEcClient:
    """Strict Ursa client for the daylily-ephemeral-cluster 2.1.11 contract."""

    def __init__(
        self,
        *,
        aws_profile: Optional[str] = None,
        python_executable: str | None = None,
    ) -> None:
        require_daylily_ec_version()
        self.aws_profile = aws_profile
        self.python_executable = python_executable or sys.executable

    def command(self, args: Iterable[str], *, json_mode: bool = False) -> list[str]:
        argv = [self.python_executable, "-m", "daylily_ec.cli"]
        if json_mode:
            argv.append("--json")
        argv.extend(str(item) for item in args)
        return argv

    def run(
        self,
        args: Iterable[str],
        *,
        json_mode: bool = False,
        aws_profile: Optional[str] = None,
        contact_email: Optional[str] = None,
        cwd: Optional[Path] = None,
        check: bool = False,
        timeout: Optional[int] = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            self.command(args, json_mode=json_mode),
            text=True,
            capture_output=True,
            cwd=str(cwd) if cwd else None,
            env=_command_env(
                aws_profile=aws_profile if aws_profile is not None else self.aws_profile,
                contact_email=contact_email,
                extra_env=extra_env,
            ),
            timeout=timeout,
            check=False,
        )
        if check and result.returncode != 0:
            raise RuntimeError(_summarize_process_output(result))
        return result

    def run_json(
        self,
        args: Iterable[str],
        *,
        aws_profile: Optional[str] = None,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        result = self.run(
            args,
            json_mode=True,
            aws_profile=aws_profile,
            cwd=cwd,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(_summarize_process_output(result))
        if not str(result.stdout or "").strip():
            detail = (result.stderr or "").strip() or "daylily-ec returned empty JSON output"
            raise RuntimeError(detail)
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"daylily-ec returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("daylily-ec returned non-object JSON")
        return cast(Dict[str, Any], payload)

    def repository_commands(self) -> Dict[str, Any]:
        return self.run_json(["repositories", "commands"])

    def cluster_list(self, *, region: str, details: bool = True) -> Dict[str, Any]:
        args = ["cluster", "list", "--region", region]
        if details:
            args.append("--details")
        if self.aws_profile:
            args.extend(["--profile", self.aws_profile])
        return self.run_json(args)

    def cluster_describe(self, *, cluster_name: str, region: str) -> Dict[str, Any]:
        args = ["cluster", "describe", "--region", region, "--cluster", cluster_name]
        if self.aws_profile:
            args.extend(["--profile", self.aws_profile])
        return self.run_json(args)

    def workflow_status(
        self, *, session_name: str, region: str, cluster_name: str
    ) -> Dict[str, Any]:
        args = [
            "workflow",
            "status",
            "--session",
            session_name,
            "--region",
            region,
            "--cluster",
            cluster_name,
        ]
        if self.aws_profile:
            args.extend(["--profile", self.aws_profile])
        return self.run_json(args)

    def workflow_logs(
        self,
        *,
        session_name: str,
        region: str,
        cluster_name: str,
        lines: int = 200,
    ) -> subprocess.CompletedProcess[str]:
        args = [
            "workflow",
            "logs",
            "--session",
            session_name,
            "--region",
            region,
            "--cluster",
            cluster_name,
            "--lines",
            str(lines),
        ]
        if self.aws_profile:
            args.extend(["--profile", self.aws_profile])
        return self.run(args)

    def stage_samples(
        self,
        *,
        analysis_samples: Path,
        reference_bucket: str,
        config_dir: Path,
        region: str,
        stage_target: str | None = None,
        aws_profile: Optional[str] = None,
        debug: bool = False,
        cwd: Optional[Path] = None,
    ) -> subprocess.CompletedProcess[str]:
        args = [
            "samples",
            "stage",
            str(analysis_samples),
            "--reference-bucket",
            reference_bucket,
            "--config-dir",
            str(config_dir),
            "--region",
            region,
        ]
        resolved_profile = aws_profile if aws_profile is not None else self.aws_profile
        if stage_target:
            args.extend(["--stage-target", stage_target])
        if resolved_profile:
            args.extend(["--profile", resolved_profile])
        if debug:
            args.append("--debug")
        return self.run(args, cwd=cwd)

    def workflow_launch(
        self,
        args: Iterable[str],
        *,
        cwd: Optional[Path] = None,
    ) -> subprocess.CompletedProcess[str]:
        return self.run(args, cwd=cwd)

    def delete_dry_run(self, *, cluster_name: str, region: str) -> subprocess.CompletedProcess[str]:
        args = [
            "delete",
            "--dry-run",
            "--cluster-name",
            cluster_name,
            "--region",
            region,
        ]
        if self.aws_profile:
            args.extend(["--profile", self.aws_profile])
        return self.run(args)

    def delete(self, *, cluster_name: str, region: str) -> subprocess.CompletedProcess[str]:
        args = ["delete", "--yes", "--cluster-name", cluster_name, "--region", region]
        if self.aws_profile:
            args.extend(["--profile", self.aws_profile])
        return self.run(args)


def get_daylily_ec_client(*, aws_profile: Optional[str] = None) -> DaylilyEcClient:
    return DaylilyEcClient(aws_profile=aws_profile)


def write_dayec_cluster_config(
    *,
    dest: Path,
    cluster_name: str,
    ssh_key_name: str,
    s3_bucket_name: str,
    contact_email: Optional[str],
    config_values: Mapping[str, Any] | None = None,
) -> Path:
    """Write a non-interactive cluster request through the day-ec 2.1.11 library."""

    require_daylily_ec_version()
    module = import_module("daylily_ec.config")
    builder = getattr(module, "build_noninteractive_cluster_config", None)
    writer = getattr(module, "write_config", None)
    triplet_type = getattr(module, "Triplet", None)
    if not callable(builder) or not callable(writer) or triplet_type is None:
        raise RuntimeError("daylily_ec.config non-interactive config helpers are not available")

    cfg = builder(
        cluster_name=cluster_name,
        ssh_key_name=ssh_key_name,
        s3_bucket_name=s3_bucket_name,
        contact_email=contact_email,
    )
    for key, raw_value in dict(config_values or {}).items():
        if key not in DAYEC_CLUSTER_CONFIG_FIELDS and key != "ssh_key_name":
            raise ValueError(f"Unsupported daylily-ec cluster config field: {key}")
        value = str(raw_value or "").strip()
        if not value:
            continue
        cfg.ephemeral_cluster.config[key] = triplet_type(
            action="USESETVALUE",
            default_value="",
            set_value=value,
        )
    writer(cfg, dest)
    return Path(dest)


def _cluster_command_args(
    verb: str,
    *,
    region_az: str,
    aws_profile: Optional[str],
    config_path: Path,
    pass_on_warn: bool,
    debug: bool,
    repo_overrides: Sequence[str] | None = None,
) -> list[str]:
    command = [
        verb,
        "--region-az",
        region_az,
        "--config",
        str(config_path),
        "--non-interactive",
    ]
    if aws_profile:
        command.extend(["--profile", aws_profile])
    if pass_on_warn:
        command.append("--pass-on-warn")
    if debug:
        command.append("--debug")
    if verb == "create":
        for override in list(repo_overrides or []):
            if str(override or "").strip():
                command.extend(["--repo-override", str(override).strip()])
    return command


def run_preflight_sync(
    *,
    region_az: str,
    aws_profile: Optional[str],
    config_path: Path,
    pass_on_warn: bool,
    debug: bool,
    contact_email: Optional[str],
    repo_overrides: Sequence[str] | None = None,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    client = get_daylily_ec_client(aws_profile=aws_profile)
    return client.run(
        _cluster_command_args(
            "preflight",
            region_az=region_az,
            aws_profile=aws_profile,
            config_path=config_path,
            pass_on_warn=pass_on_warn,
            debug=debug,
            repo_overrides=repo_overrides,
        ),
        contact_email=contact_email,
        cwd=cwd,
    )


def run_create_dry_run_sync(
    *,
    region_az: str,
    aws_profile: Optional[str],
    config_path: str,
    pass_on_warn: bool,
    debug: bool,
    contact_email: Optional[str],
    repo_overrides: Sequence[str] | None = None,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    resolved_config_path = Path(config_path).expanduser()
    if not resolved_config_path.is_absolute():
        resolved_config_path = ((cwd or Path.cwd()) / resolved_config_path).resolve()
    client = get_daylily_ec_client(aws_profile=aws_profile)
    return client.run(
        _cluster_command_args(
            "create",
            region_az=region_az,
            aws_profile=aws_profile,
            config_path=resolved_config_path,
            pass_on_warn=pass_on_warn,
            debug=debug,
            repo_overrides=repo_overrides,
        ),
        contact_email=contact_email,
        cwd=cwd,
        extra_env={"DAY_BREAK": "1"},
    )


def run_aws_validate_all_sync(
    *,
    region_az: str,
    aws_profile: str,
    gap_analysis_path: Path,
    config_path: str | None = None,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    args = [
        "aws",
        "validate",
        "all",
        "--profile",
        aws_profile,
        "--region-az",
        region_az,
    ]
    if config_path:
        args.extend(["--config", config_path])
    args.extend(["--gap-analysis", str(gap_analysis_path)])
    client = get_daylily_ec_client(aws_profile=aws_profile)
    return client.run(args, json_mode=True, cwd=cwd)


def run_create_sync(
    *,
    region_az: str,
    aws_profile: Optional[str],
    config_path: str,
    pass_on_warn: bool,
    debug: bool,
    contact_email: Optional[str],
    repo_overrides: Sequence[str] | None = None,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    resolved_config_path = Path(config_path).expanduser()
    if not resolved_config_path.is_absolute():
        resolved_config_path = ((cwd or Path.cwd()) / resolved_config_path).resolve()
    client = get_daylily_ec_client(aws_profile=aws_profile)
    return client.run(
        _cluster_command_args(
            "create",
            region_az=region_az,
            aws_profile=aws_profile,
            config_path=resolved_config_path,
            pass_on_warn=pass_on_warn,
            debug=debug,
            repo_overrides=repo_overrides,
        ),
        contact_email=contact_email,
        cwd=cwd,
    )


__all__ = [
    "DAYLILY_EC_DISTRIBUTION",
    "REQUIRED_DAYLILY_EC_VERSION",
    "DaylilyEcClient",
    "_summarize_process_output",
    "get_daylily_ec_client",
    "require_daylily_ec_version",
    "run_aws_validate_all_sync",
    "run_create_sync",
    "run_create_dry_run_sync",
    "run_preflight_sync",
    "write_dayec_cluster_config",
]
