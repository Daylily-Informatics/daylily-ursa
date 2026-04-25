from __future__ import annotations

import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from daylib_ursa.cluster_service import ClusterService
from daylib_ursa.ephemeral_cluster.runner import (
    _summarize_process_output,
    run_create_sync,
    run_preflight_sync,
    write_dayec_cluster_config,
)
from daylib_ursa.resource_store import ClusterJobRecord, ResourceStore
from daylib_ursa.tapdb_graph import utc_now_iso

LOGGER = logging.getLogger("daylily.ursa.cluster_jobs")


def region_from_region_az(region_az: str) -> str:
    value = str(region_az or "").strip()
    if len(value) > 1 and value[-1].isalpha() and value[-2].isdigit():
        return value[:-1]
    return value


def run_cluster_create_job(
    *,
    resource_store: ResourceStore,
    cluster_service: ClusterService,
    workspace_root: Path,
    job_euid: str,
) -> None:
    job = resource_store.get_cluster_job(job_euid)
    if job is None:
        raise KeyError(f"cluster job not found: {job_euid}")

    request = dict(job.request or {})
    cluster_name = str(request.get("cluster_name") or job.cluster_name or "").strip()
    region = str(request.get("region") or job.region or "").strip()
    region_az = str(request.get("region_az") or job.region_az or "").strip()
    ssh_key_name = str(request.get("ssh_key_name") or "").strip()
    s3_bucket_name = str(request.get("s3_bucket_name") or "").strip()
    aws_profile = str(request.get("aws_profile") or "").strip() or None
    contact_email = str(request.get("contact_email") or "").strip() or None
    pass_on_warn = bool(request.get("pass_on_warn"))
    debug = bool(request.get("debug"))
    sponsor_user_id = str(job.sponsor_user_id or job.owner_user_id or "").strip()

    started_at = utc_now_iso()
    try:
        resource_store.update_cluster_job_status(
            job_euid=job_euid,
            state="RUNNING",
            created_by=sponsor_user_id,
            started_at=started_at,
        )
        resource_store.add_cluster_job_event(
            job_euid=job_euid,
            event_type="runner",
            status="RUNNING",
            summary=f"Started cluster creation for {cluster_name}",
            details={"region": region, "region_az": region_az},
            created_by=sponsor_user_id,
        )

        with TemporaryDirectory(prefix="ursa-cluster-") as temp_dir:
            scratch_dir = Path(temp_dir)
            config_path = write_dayec_cluster_config(
                dest=scratch_dir / "cluster.yaml",
                cluster_name=cluster_name,
                ssh_key_name=ssh_key_name,
                s3_bucket_name=s3_bucket_name,
                contact_email=contact_email,
            )

            preflight = run_preflight_sync(
                region_az=region_az,
                aws_profile=aws_profile,
                config_path=config_path,
                pass_on_warn=pass_on_warn,
                debug=debug,
                contact_email=contact_email,
                cwd=workspace_root,
            )
            preflight_summary = _summarize_process_output(preflight)
            resource_store.add_cluster_job_event(
                job_euid=job_euid,
                event_type="preflight",
                status="COMPLETED" if preflight.returncode == 0 else "FAILED",
                summary=preflight_summary,
                details={
                    "return_code": int(preflight.returncode),
                    "stdout": (preflight.stdout or "")[-4000:],
                    "stderr": (preflight.stderr or "")[-4000:],
                },
                created_by=sponsor_user_id,
            )
            if preflight.returncode != 0:
                resource_store.update_cluster_job_status(
                    job_euid=job_euid,
                    state="FAILED",
                    created_by=sponsor_user_id,
                    started_at=started_at,
                    completed_at=utc_now_iso(),
                    return_code=int(preflight.returncode),
                    error=preflight_summary,
                    output_summary=preflight_summary,
                )
                return

            result = run_create_sync(
                region_az=region_az,
                aws_profile=aws_profile,
                config_path=str(config_path),
                pass_on_warn=pass_on_warn,
                debug=debug,
                contact_email=contact_email,
                cwd=workspace_root,
            )
            result_summary = _summarize_process_output(result)
            result_status = "COMPLETED" if result.returncode == 0 else "FAILED"
            resource_store.add_cluster_job_event(
                job_euid=job_euid,
                event_type="create",
                status=result_status,
                summary=result_summary,
                details={
                    "return_code": int(result.returncode),
                    "stdout": (result.stdout or "")[-4000:],
                    "stderr": (result.stderr or "")[-4000:],
                },
                created_by=sponsor_user_id,
            )

            cluster_payload: dict[str, Any] = {}
            if result.returncode == 0:
                try:
                    cluster_service.clear_cache()
                    cluster_info = cluster_service.describe_cluster(cluster_name, region)
                    cluster_payload = cluster_info.to_dict(include_sensitive=True)
                except Exception:
                    LOGGER.exception("Failed to refresh cluster state for %s", cluster_name)

            resource_store.update_cluster_job_status(
                job_euid=job_euid,
                state=result_status,
                created_by=sponsor_user_id,
                started_at=started_at,
                completed_at=utc_now_iso(),
                return_code=int(result.returncode),
                error=None if result.returncode == 0 else result_summary,
                output_summary=result_summary,
                cluster=cluster_payload,
            )
    except Exception as exc:
        LOGGER.exception("Cluster create job %s failed", job_euid)
        error_message = f"{type(exc).__name__}: {exc}"
        try:
            resource_store.add_cluster_job_event(
                job_euid=job_euid,
                event_type="runner",
                status="FAILED",
                summary=error_message,
                details={},
                created_by=sponsor_user_id,
            )
            resource_store.update_cluster_job_status(
                job_euid=job_euid,
                state="FAILED",
                created_by=sponsor_user_id,
                started_at=started_at,
                completed_at=utc_now_iso(),
                return_code=1,
                error=error_message,
                output_summary=error_message,
            )
        except Exception:
            LOGGER.exception("Failed to persist terminal cluster job state for %s", job_euid)


class ClusterJobManager:
    """TapDB-backed cluster job launcher that delegates execution to a worker process."""

    def __init__(
        self,
        *,
        resource_store: ResourceStore,
        cluster_service: ClusterService,
        workspace_root: Path | None = None,
        worker_module: str = "daylib_ursa.cluster_job_worker",
        python_executable: str | None = None,
    ) -> None:
        self.resource_store = resource_store
        self.cluster_service = cluster_service
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.worker_module = worker_module
        self.python_executable = python_executable or sys.executable

    def _worker_command(self, *, job_euid: str) -> list[str]:
        return [
            self.python_executable,
            "-m",
            self.worker_module,
            "--job-euid",
            job_euid,
            "--workspace-root",
            str(self.workspace_root),
        ]

    def _spawn_worker(self, *, job_euid: str) -> subprocess.Popen[str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        return subprocess.Popen(
            self._worker_command(job_euid=job_euid),
            cwd=str(self.workspace_root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )

    def start_create_job(
        self,
        *,
        cluster_name: str,
        region_az: str,
        ssh_key_name: str,
        s3_bucket_name: str,
        tenant_id: uuid.UUID,
        owner_user_id: str,
        sponsor_user_id: str,
        aws_profile: str | None,
        contact_email: str | None,
        pass_on_warn: bool,
        debug: bool,
    ) -> ClusterJobRecord:
        region = region_from_region_az(region_az)
        request_payload = {
            "cluster_name": cluster_name,
            "region": region,
            "region_az": region_az,
            "ssh_key_name": ssh_key_name,
            "s3_bucket_name": s3_bucket_name,
            "aws_profile": aws_profile,
            "contact_email": contact_email,
            "pass_on_warn": bool(pass_on_warn),
            "debug": bool(debug),
        }
        job = self.resource_store.create_cluster_job(
            cluster_name=cluster_name,
            region=region,
            region_az=region_az,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            sponsor_user_id=sponsor_user_id,
            request=request_payload,
        )
        self.resource_store.add_cluster_job_event(
            job_euid=job.job_euid,
            event_type="queued",
            status="QUEUED",
            summary=f"Queued cluster creation for {cluster_name} in {region_az}",
            details=request_payload,
            created_by=sponsor_user_id,
        )

        try:
            worker = self._spawn_worker(job_euid=job.job_euid)
        except Exception as exc:
            error_message = f"Failed to launch cluster worker: {type(exc).__name__}: {exc}"
            self.resource_store.add_cluster_job_event(
                job_euid=job.job_euid,
                event_type="worker-launch",
                status="FAILED",
                summary=error_message,
                details={"command": self._worker_command(job_euid=job.job_euid)},
                created_by=sponsor_user_id,
            )
            return self.resource_store.update_cluster_job_status(
                job_euid=job.job_euid,
                state="FAILED",
                created_by=sponsor_user_id,
                completed_at=utc_now_iso(),
                return_code=1,
                error=error_message,
                output_summary=error_message,
            )

        self.resource_store.add_cluster_job_event(
            job_euid=job.job_euid,
            event_type="worker-launch",
            status="QUEUED",
            summary=f"Spawned cluster worker pid {worker.pid}",
            details={
                "pid": int(worker.pid or 0),
                "command": self._worker_command(job_euid=job.job_euid),
            },
            created_by=sponsor_user_id,
        )
        return self.resource_store.get_cluster_job(job.job_euid) or job


__all__ = ["ClusterJobManager", "region_from_region_az", "run_cluster_create_job"]
