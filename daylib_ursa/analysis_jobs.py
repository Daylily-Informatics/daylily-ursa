from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from daylib_ursa.analysis_commands import get_analysis_command
from daylib_ursa.ephemeral_cluster.runner import DaylilyEcClient, _summarize_process_output
from daylib_ursa.resource_store import AnalysisJobRecord, ResourceStore, StagingJobRecord
from daylib_ursa.tapdb_graph import utc_now_iso


_MARKER_RE = re.compile(r"^__(?P<name>DAYLILY_[A-Z_]+)__=(?P<value>.*)$", re.MULTILINE)


def _safe_session_name(job_euid: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(job_euid or "").strip()).strip("-")
    if not cleaned:
        raise ValueError("job_euid is required")
    return f"ursa-{cleaned}"[:80]


def _parse_launch_markers(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in _MARKER_RE.finditer(stdout or ""):
        values[match.group("name")] = match.group("value").strip()
    session_name = values.get("DAYLILY_SESSION")
    run_dir = values.get("DAYLILY_RUN_DIR")
    repo_path = values.get("DAYLILY_REPO_PATH")
    if not session_name or not run_dir or not repo_path:
        raise RuntimeError("daylily-ec workflow launch did not report launch markers")
    return {
        "session_name": session_name,
        "run_dir": run_dir,
        "repo_path": repo_path,
    }


class AnalysisJobManager:
    """Launch manager for Ursa analysis jobs through daylily-ec 2.1.11."""

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

    @staticmethod
    def _stage_dir_from_completed_staging_job(
        *, analysis_job: AnalysisJobRecord, staging_job: StagingJobRecord
    ) -> str:
        if staging_job.tenant_id != analysis_job.tenant_id:
            raise ValueError("Staging job tenant does not match analysis job tenant")
        if staging_job.workset_euid != analysis_job.workset_euid:
            raise ValueError("Staging job does not belong to analysis job workset")
        if staging_job.manifest_euid != analysis_job.manifest_euid:
            raise ValueError("Staging job does not belong to analysis job manifest")
        if staging_job.state != "COMPLETED":
            raise ValueError("Staging job must be COMPLETED before analysis launch")
        stage_dir = str((staging_job.stage or {}).get("stage_dir") or "").strip()
        if not stage_dir:
            raise ValueError("Staging job is completed but has no stage_dir")
        return stage_dir

    def _run_staging_job_for_analysis(
        self,
        *,
        job: AnalysisJobRecord,
        request: dict[str, Any],
        aws_profile: str | None,
        actor_user_id: str,
    ) -> tuple[StagingJobRecord, str]:
        from daylib_ursa.staging_jobs import StagingJobManager

        reference_bucket = str(request.get("reference_bucket") or "").strip()
        if not reference_bucket:
            raise ValueError("reference_bucket is required for analysis launch staging")
        staging_job = self.resource_store.create_staging_job(
            job_name=f"{job.job_name}:staging",
            workset_euid=job.workset_euid,
            manifest_euid=job.manifest_euid,
            cluster_name=job.cluster_name,
            region=job.region,
            tenant_id=job.tenant_id,
            owner_user_id=job.owner_user_id,
            request={
                "reference_bucket": reference_bucket,
                "stage_target": str(request.get("stage_target") or "").strip()
                or "/data/staged_sample_data",
                "aws_profile": aws_profile,
                "analysis_job_euid": job.job_euid,
            },
        )
        self.resource_store.add_staging_job_event(
            job_euid=staging_job.job_euid,
            event_type="defined",
            status="DEFINED",
            summary="Defined staging job for analysis launch",
            details={"analysis_job_euid": job.job_euid, "manifest_euid": job.manifest_euid},
            created_by=actor_user_id,
        )
        manager = StagingJobManager(
            resource_store=self.resource_store,
            client=self.client,
            workspace_root=self.workspace_root,
        )
        completed = manager.run_job(staging_job.job_euid, actor_user_id=actor_user_id)
        if completed.state != "COMPLETED":
            raise RuntimeError(completed.error or completed.output_summary or "Staging job failed")
        stage_dir = self._stage_dir_from_completed_staging_job(
            analysis_job=job, staging_job=completed
        )
        return completed, stage_dir

    def _launch_workflow(
        self,
        *,
        job: AnalysisJobRecord,
        request: dict[str, Any],
        stage_dir: str,
        aws_profile: str | None,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
        command_id = str(
            request.get("analysis_command_id") or request.get("command_id") or ""
        ).strip()
        optional_features = [
            str(item or "").strip()
            for item in list(request.get("optional_features") or [])
            if str(item or "").strip()
        ]
        command = get_analysis_command(command_id, optional_features=optional_features)
        session_name = str(request.get("session_name") or "").strip() or _safe_session_name(
            job.job_euid
        )
        argv = command.launch_argv(
            profile=aws_profile,
            region=job.region,
            cluster=job.cluster_name,
            stage_dir=stage_dir,
            session_name=session_name,
            project=str(request.get("project") or "").strip() or None,
            dry_run=bool(request.get("dry_run")),
        )
        result = self.client.workflow_launch(argv, cwd=self.workspace_root)
        if result.returncode != 0:
            raise RuntimeError(_summarize_process_output(result))
        markers = _parse_launch_markers(result.stdout or "")
        launch = {
            "command_id": command.command_id,
            "optional_features": optional_features,
            "argv": list(argv),
            "stage_dir": stage_dir,
            "stdout": (result.stdout or "")[-8000:],
            "stderr": (result.stderr or "")[-8000:],
            **markers,
        }
        return result, launch

    def launch_job(self, job_euid: str, *, actor_user_id: str) -> AnalysisJobRecord:
        job = self.resource_store.get_analysis_job(job_euid)
        if job is None:
            raise KeyError(f"analysis job not found: {job_euid}")
        request = dict(job.request or {})
        aws_profile = (
            str(request.get("aws_profile") or self.client.aws_profile or "").strip() or None
        )
        started_at = utc_now_iso()
        try:
            self.resource_store.update_analysis_job_status(
                job_euid=job_euid,
                state="STAGING",
                created_by=actor_user_id,
                started_at=started_at,
            )
            self.resource_store.add_analysis_job_event(
                job_euid=job_euid,
                event_type="stage",
                status="RUNNING",
                summary="Preparing analysis staging",
                details={"manifest_euid": job.manifest_euid},
                created_by=actor_user_id,
            )
            requested_staging_job_euid = str(request.get("staging_job_euid") or "").strip()
            if requested_staging_job_euid:
                staging_job = self.resource_store.get_staging_job(requested_staging_job_euid)
                if staging_job is None:
                    raise KeyError(f"staging job not found: {requested_staging_job_euid}")
                stage_dir = self._stage_dir_from_completed_staging_job(
                    analysis_job=job,
                    staging_job=staging_job,
                )
                staging_job_euid = staging_job.job_euid
                self.resource_store.add_analysis_job_event(
                    job_euid=job_euid,
                    event_type="stage",
                    status="COMPLETED",
                    summary=f"Using staged samples from {stage_dir}",
                    details={
                        "staging_job_euid": staging_job_euid,
                        "stage_dir": stage_dir,
                    },
                    created_by=actor_user_id,
                )
            else:
                staging_job, stage_dir = self._run_staging_job_for_analysis(
                    job=job,
                    request=request,
                    aws_profile=aws_profile,
                    actor_user_id=actor_user_id,
                )
                staging_job_euid = staging_job.job_euid
                self.resource_store.add_analysis_job_event(
                    job_euid=job_euid,
                    event_type="stage",
                    status="COMPLETED",
                    summary=f"Staged samples to {stage_dir}",
                    details={"staging_job_euid": staging_job_euid, "stage_dir": stage_dir},
                    created_by=actor_user_id,
                )
            self.resource_store.update_analysis_job_status(
                job_euid=job_euid,
                state="LAUNCHING",
                created_by=actor_user_id,
                started_at=started_at,
                launch={"stage_dir": stage_dir, "staging_job_euid": staging_job_euid},
            )
            launch_result, launch = self._launch_workflow(
                job=job,
                request=request,
                stage_dir=stage_dir,
                aws_profile=aws_profile,
            )
            launch["staging_job_euid"] = staging_job_euid
            self.resource_store.add_analysis_job_event(
                job_euid=job_euid,
                event_type="launch",
                status="RUNNING",
                summary=f"Workflow session {launch['session_name']} launched",
                details={
                    "session_name": launch["session_name"],
                    "run_dir": launch["run_dir"],
                    "return_code": int(launch_result.returncode),
                    "staging_job_euid": staging_job_euid,
                },
                created_by=actor_user_id,
            )
            return self.resource_store.update_analysis_job_status(
                job_euid=job_euid,
                state="RUNNING",
                created_by=actor_user_id,
                started_at=started_at,
                return_code=int(launch_result.returncode),
                output_summary=f"Workflow session {launch['session_name']} launched",
                launch=launch,
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            self.resource_store.add_analysis_job_event(
                job_euid=job_euid,
                event_type="runner",
                status="FAILED",
                summary=error_message,
                details={},
                created_by=actor_user_id,
            )
            return self.resource_store.update_analysis_job_status(
                job_euid=job_euid,
                state="FAILED",
                created_by=actor_user_id,
                started_at=started_at,
                completed_at=utc_now_iso(),
                return_code=1,
                error=error_message,
                output_summary=error_message,
            )

    def refresh_job(self, job_euid: str, *, actor_user_id: str) -> AnalysisJobRecord:
        job = self.resource_store.get_analysis_job(job_euid)
        if job is None:
            raise KeyError(f"analysis job not found: {job_euid}")
        session_name = str(job.launch.get("session_name") or "").strip()
        if not session_name:
            raise ValueError("Analysis job has not been launched")
        status_payload = self.client.workflow_status(
            session_name=session_name,
            region=job.region,
            cluster_name=job.cluster_name,
        )
        launch = dict(job.launch or {})
        launch["status"] = status_payload
        exit_code = status_payload.get("exit_code")
        completed_at = str(status_payload.get("completed_at") or "").strip() or None
        if exit_code is None:
            state = "RUNNING"
            return_code = job.return_code
            error = None
        else:
            return_code = int(exit_code) if isinstance(exit_code, int) else 1
            state = "COMPLETED" if return_code == 0 else "FAILED"
            error = None if return_code == 0 else f"Workflow exited with status {return_code}"
        record = self.resource_store.update_analysis_job_status(
            job_euid=job_euid,
            state=state,
            created_by=actor_user_id,
            completed_at=completed_at,
            return_code=return_code,
            error=error,
            output_summary=f"Workflow status: {state}",
            launch=launch,
        )
        self.resource_store.add_analysis_job_event(
            job_euid=job_euid,
            event_type="refresh",
            status=state,
            summary=f"Workflow status refreshed: {state}",
            details=status_payload,
            created_by=actor_user_id,
        )
        return record

    def logs(self, job_euid: str, *, lines: int = 200) -> dict[str, Any]:
        job = self.resource_store.get_analysis_job(job_euid)
        if job is None:
            raise KeyError(f"analysis job not found: {job_euid}")
        session_name = str(job.launch.get("session_name") or "").strip()
        if not session_name:
            raise ValueError("Analysis job has not been launched")
        result = self.client.workflow_logs(
            session_name=session_name,
            region=job.region,
            cluster_name=job.cluster_name,
            lines=lines,
        )
        if result.returncode != 0:
            raise RuntimeError(_summarize_process_output(result))
        return {
            "job_euid": job.job_euid,
            "session_name": session_name,
            "lines": lines,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
