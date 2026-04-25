from __future__ import annotations

import subprocess
import uuid
from dataclasses import replace
from pathlib import Path

from daylib_ursa.resource_store import (
    ManifestRecord,
    StagingJobEventRecord,
    StagingJobRecord,
)
from daylib_ursa.staging_jobs import StagingJobManager


TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


class MemoryResourceStore:
    def __init__(self) -> None:
        self.manifest = ManifestRecord(
            manifest_euid="MF-1",
            name="analysis samples",
            workset_euid="WS-1",
            tenant_id=TENANT_ID,
            owner_user_id="user-1",
            artifact_set_euid=None,
            artifact_euids=[],
            input_references=[],
            metadata={"analysis_samples_manifest": {"content": "RUN_ID\tSAMPLE_ID\nR1\tS1\n"}},
            created_at="2026-03-25T00:10:00Z",
            updated_at="2026-03-25T00:10:00Z",
            state="ACTIVE",
        )
        self.job = StagingJobRecord(
            job_euid="SJ-1",
            job_name="stage",
            workset_euid="WS-1",
            manifest_euid="MF-1",
            cluster_name="cluster-1",
            region="us-west-2",
            tenant_id=TENANT_ID,
            owner_user_id="user-1",
            state="DEFINED",
            created_at="2026-03-25T00:35:00Z",
            updated_at="2026-03-25T00:35:00Z",
            started_at=None,
            completed_at=None,
            return_code=None,
            error=None,
            output_summary=None,
            request={
                "reference_bucket": "s3://reference-bucket",
                "stage_target": "/fsx/staged_sample_data",
            },
            stage={},
            events=[],
        )
        self._event_seq = 0

    def get_staging_job(self, job_euid: str) -> StagingJobRecord | None:
        if job_euid != self.job.job_euid:
            return None
        return self.job

    def get_manifest(self, manifest_euid: str) -> ManifestRecord | None:
        if manifest_euid != self.manifest.manifest_euid:
            return None
        return self.manifest

    def update_staging_job_status(
        self,
        *,
        job_euid: str,
        state: str,
        created_by: str,
        started_at: str | None = None,
        completed_at: str | None = None,
        return_code: int | None = None,
        error: str | None = None,
        output_summary: str | None = None,
        stage: dict | None = None,
    ) -> StagingJobRecord:
        _ = (job_euid, created_by)
        self.job = replace(
            self.job,
            state=state,
            updated_at=completed_at or started_at or self.job.updated_at,
            started_at=started_at if started_at is not None else self.job.started_at,
            completed_at=completed_at if completed_at is not None else self.job.completed_at,
            return_code=return_code if return_code is not None else self.job.return_code,
            error=error if error is not None else self.job.error,
            output_summary=output_summary
            if output_summary is not None
            else self.job.output_summary,
            stage=dict(stage if stage is not None else self.job.stage),
        )
        return self.job

    def add_staging_job_event(
        self,
        *,
        job_euid: str,
        event_type: str,
        status: str,
        summary: str,
        details: dict | None = None,
        created_by: str | None = None,
    ) -> StagingJobEventRecord:
        _ = job_euid
        self._event_seq += 1
        event = StagingJobEventRecord(
            event_euid=f"SJE-{self._event_seq}",
            job_euid=self.job.job_euid,
            event_type=event_type,
            status=status,
            summary=summary,
            details=dict(details or {}),
            created_by=created_by,
            created_at=f"2026-03-25T00:35:{self._event_seq:02d}Z",
        )
        self.job = replace(self.job, events=[event, *self.job.events])
        return event


class FakeDaylilyEcClient:
    aws_profile = None

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], Path]] = []

    def stage_samples(self, **kwargs):
        cwd = kwargs.pop("cwd")
        self.calls.append((dict(kwargs), cwd))
        return subprocess.CompletedProcess(
            ["daylily-ec", "samples", "stage"],
            0,
            "stage log\nRemote FSx stage directory: /fsx/staged_sample_data\n",
            "",
        )


def test_staging_job_manager_runs_samples_stage_and_persists_logs(tmp_path: Path) -> None:
    store = MemoryResourceStore()
    client = FakeDaylilyEcClient()
    manager = StagingJobManager(resource_store=store, client=client, workspace_root=tmp_path)

    record = manager.run_job("SJ-1", actor_user_id="user-1")
    logs = manager.logs("SJ-1", lines=10)

    assert record.state == "COMPLETED"
    assert record.return_code == 0
    assert record.stage["stage_dir"] == "/fsx/staged_sample_data"
    assert record.events[0].status == "COMPLETED"
    assert logs["stdout"].startswith("stage log\n")
    assert client.calls[0][0]["stage_target"] == "/fsx/staged_sample_data"
    assert client.calls[0][0]["reference_bucket"] == "s3://reference-bucket"
    assert client.calls[0][1] == tmp_path.resolve()
