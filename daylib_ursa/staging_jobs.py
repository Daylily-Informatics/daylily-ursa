from __future__ import annotations

import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from daylib_ursa.ephemeral_cluster.runner import DaylilyEcClient, _summarize_process_output
from daylib_ursa.resource_store import ManifestRecord, ResourceStore, StagingJobRecord
from daylib_ursa.tapdb_graph import utc_now_iso


_REMOTE_STAGE_RE = re.compile(r"^Remote FSx stage directory:\s*(?P<path>\S+)\s*$", re.MULTILINE)


def _manifest_content(manifest: ManifestRecord) -> str:
    analysis_samples_manifest = dict(
        (manifest.metadata or {}).get("analysis_samples_manifest") or {}
    )
    content = str(analysis_samples_manifest.get("content") or "")
    if not content:
        raise ValueError("Manifest is missing analysis_samples_manifest.content")
    return content


def _parse_stage_dir(stdout: str) -> str:
    match = _REMOTE_STAGE_RE.search(stdout or "")
    if match is None:
        raise RuntimeError("daylily-ec samples stage did not report a remote FSx stage directory")
    return match.group("path").strip()


def _tail_lines(value: str, lines: int) -> str:
    if lines <= 0:
        raise ValueError("lines must be greater than zero")
    split = str(value or "").splitlines(keepends=True)
    return "".join(split[-lines:])


class StagingJobManager:
    """Run Ursa sample-staging jobs through daylily-ec."""

    def __init__(
        self,
        *,
        resource_store: ResourceStore,
        client: DaylilyEcClient,
        workspace_root: Path | None = None,
    ) -> None:
        self.resource_store = resource_store
        self.client = client
        self.workspace_root = (workspace_root or Path.cwd()).resolve()

    def _stage_samples(
        self,
        *,
        manifest: ManifestRecord,
        job: StagingJobRecord,
        temp_dir: Path,
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        request = dict(job.request or {})
        reference_bucket = str(request.get("reference_bucket") or "").strip()
        if not reference_bucket:
            raise ValueError("reference_bucket is required for staging")
        manifest_path = temp_dir / "analysis_samples.tsv"
        manifest_path.write_text(_manifest_content(manifest), encoding="utf-8", newline="\n")
        stage_target = str(request.get("stage_target") or "").strip()
        aws_profile = str(request.get("aws_profile") or self.client.aws_profile or "").strip()
        result = self.client.stage_samples(
            analysis_samples=manifest_path,
            reference_bucket=reference_bucket,
            config_dir=temp_dir,
            region=job.region,
            stage_target=stage_target or None,
            aws_profile=aws_profile or None,
            debug=bool(request.get("debug")),
            cwd=self.workspace_root,
        )
        if result.returncode != 0:
            raise RuntimeError(_summarize_process_output(result))
        return result, _parse_stage_dir(result.stdout or "")

    def run_job(self, job_euid: str, *, actor_user_id: str) -> StagingJobRecord:
        job = self.resource_store.get_staging_job(job_euid)
        if job is None:
            raise KeyError(f"staging job not found: {job_euid}")
        if job.state != "DEFINED":
            raise ValueError("Staging job must be DEFINED before it can run")
        manifest = self.resource_store.get_manifest(job.manifest_euid)
        if manifest is None:
            raise KeyError(f"manifest not found: {job.manifest_euid}")
        started_at = utc_now_iso()
        try:
            self.resource_store.update_staging_job_status(
                job_euid=job_euid,
                state="STAGING",
                created_by=actor_user_id,
                started_at=started_at,
            )
            self.resource_store.add_staging_job_event(
                job_euid=job_euid,
                event_type="stage",
                status="STAGING",
                summary="Staging analysis_samples_manifest",
                details={"manifest_euid": manifest.manifest_euid},
                created_by=actor_user_id,
            )
            with TemporaryDirectory(prefix="ursa-staging-") as temp_dir_name:
                result, stage_dir = self._stage_samples(
                    manifest=manifest,
                    job=job,
                    temp_dir=Path(temp_dir_name),
                )
            completed_at = utc_now_iso()
            stage = {
                "stage_dir": stage_dir,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "argv": list(result.args) if isinstance(result.args, list) else result.args,
            }
            self.resource_store.add_staging_job_event(
                job_euid=job_euid,
                event_type="stage",
                status="COMPLETED",
                summary=f"Staged samples to {stage_dir}",
                details={"stage_dir": stage_dir, "return_code": int(result.returncode)},
                created_by=actor_user_id,
            )
            return self.resource_store.update_staging_job_status(
                job_euid=job_euid,
                state="COMPLETED",
                created_by=actor_user_id,
                started_at=started_at,
                completed_at=completed_at,
                return_code=int(result.returncode),
                output_summary=f"Staged samples to {stage_dir}",
                stage=stage,
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            self.resource_store.add_staging_job_event(
                job_euid=job_euid,
                event_type="runner",
                status="FAILED",
                summary=error_message,
                details={},
                created_by=actor_user_id,
            )
            return self.resource_store.update_staging_job_status(
                job_euid=job_euid,
                state="FAILED",
                created_by=actor_user_id,
                started_at=started_at,
                completed_at=utc_now_iso(),
                return_code=1,
                error=error_message,
                output_summary=error_message,
            )

    def logs(self, job_euid: str, *, lines: int = 200) -> dict[str, Any]:
        job = self.resource_store.get_staging_job(job_euid)
        if job is None:
            raise KeyError(f"staging job not found: {job_euid}")
        stage = dict(job.stage or {})
        if not stage:
            raise ValueError("Staging job has no captured logs")
        return {
            "job_euid": job.job_euid,
            "stage_dir": str(stage.get("stage_dir") or ""),
            "lines": lines,
            "stdout": _tail_lines(str(stage.get("stdout") or ""), lines),
            "stderr": _tail_lines(str(stage.get("stderr") or ""), lines),
        }
