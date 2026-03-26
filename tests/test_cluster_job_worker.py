from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from daylib_ursa.cluster_jobs import ClusterJobManager, run_cluster_create_job
from daylib_ursa.cluster_service import ClusterService
from daylib_ursa.resource_store import ClusterJobEventRecord, ClusterJobRecord


class MemoryResourceStore:
    def __init__(self) -> None:
        self.jobs: dict[str, ClusterJobRecord] = {}
        self._job_seq = 0
        self._event_seq = 0

    def create_cluster_job(
        self,
        *,
        cluster_name: str,
        region: str,
        region_az: str,
        atlas_tenant_id: str,
        owner_user_id: str,
        sponsor_user_id: str,
        request: dict | None = None,
    ) -> ClusterJobRecord:
        self._job_seq += 1
        record = ClusterJobRecord(
            job_euid=f"CJ-{self._job_seq}",
            job_name=cluster_name,
            cluster_name=cluster_name,
            region=region,
            region_az=region_az,
            atlas_tenant_id=atlas_tenant_id,
            owner_user_id=owner_user_id,
            sponsor_user_id=sponsor_user_id,
            state="QUEUED",
            created_at="2026-03-25T00:00:00Z",
            updated_at="2026-03-25T00:00:00Z",
            started_at=None,
            completed_at=None,
            return_code=None,
            error=None,
            output_summary=None,
            request=dict(request or {}),
            cluster={},
            events=[],
        )
        self.jobs[record.job_euid] = record
        return record

    def add_cluster_job_event(
        self,
        *,
        job_euid: str,
        event_type: str,
        status: str,
        summary: str,
        details: dict | None = None,
        created_by: str | None = None,
    ) -> ClusterJobEventRecord:
        self._event_seq += 1
        job = self.jobs[job_euid]
        event = ClusterJobEventRecord(
            event_euid=f"CE-{self._event_seq}",
            job_euid=job_euid,
            event_type=event_type,
            status=status,
            summary=summary,
            details=dict(details or {}),
            created_by=created_by,
            created_at=f"2026-03-25T00:00:{self._event_seq:02d}Z",
        )
        self.jobs[job_euid] = replace(job, events=[*job.events, event], updated_at=event.created_at)
        return event

    def update_cluster_job_status(
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
        cluster: dict | None = None,
    ) -> ClusterJobRecord:
        _ = created_by
        job = self.jobs[job_euid]
        updated = replace(
            job,
            state=state,
            updated_at=completed_at or started_at or "2026-03-25T00:10:00Z",
            started_at=started_at if started_at is not None else job.started_at,
            completed_at=completed_at if completed_at is not None else job.completed_at,
            return_code=return_code if return_code is not None else job.return_code,
            error=error if error is not None else job.error,
            output_summary=output_summary if output_summary is not None else job.output_summary,
            cluster=dict(cluster or job.cluster),
        )
        self.jobs[job_euid] = updated
        return updated

    def get_cluster_job(self, job_euid: str) -> ClusterJobRecord | None:
        return self.jobs.get(job_euid)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_cluster_job_manager_spawns_dedicated_worker_process(monkeypatch, tmp_path: Path) -> None:
    store = MemoryResourceStore()
    captured: dict[str, object] = {}

    class _DummyPopen:
        pid = 43210

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _DummyPopen()

    monkeypatch.setattr("daylib_ursa.cluster_jobs.subprocess.Popen", fake_popen)

    manager = ClusterJobManager(
        resource_store=store,
        cluster_service=ClusterService(regions=["us-west-2"]),
        workspace_root=tmp_path,
        python_executable="/usr/bin/python3",
    )

    job = manager.start_create_job(
        cluster_name="cluster-1",
        region_az="us-west-2d",
        ssh_key_name="omics-key",
        s3_bucket_name="ursa-bucket",
        atlas_tenant_id="TEN-1",
        owner_user_id="user-1",
        sponsor_user_id="user-2",
        aws_profile="ursa",
        contact_email="ops@example.com",
        pass_on_warn=False,
        debug=False,
    )

    assert captured["cmd"] == [
        "/usr/bin/python3",
        "-m",
        "daylib_ursa.cluster_job_worker",
        "--job-euid",
        job.job_euid,
        "--workspace-root",
        str(tmp_path.resolve()),
    ]
    assert captured["kwargs"]["cwd"] == str(tmp_path.resolve())
    assert any(event.event_type == "worker-launch" for event in job.events)


def test_run_cluster_create_job_uses_fake_tools_and_leaves_no_home_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    daylily_ec = bin_dir / "daylily-ec"
    _write_executable(
        daylily_ec,
        """#!/bin/sh
set -eu
cmd="$1"
shift
if [ "$cmd" = "preflight" ]; then
  echo "preflight ok"
  exit 0
fi
if [ "$cmd" = "create" ]; then
  echo "create ok"
  exit 0
fi
echo "unexpected command" >&2
exit 2
""",
    )

    pcluster = bin_dir / "pcluster"
    _write_executable(
        pcluster,
        """#!/bin/sh
set -eu
cat <<'JSON'
{"clusterName":"cluster-1","clusterStatus":"CREATE_COMPLETE","computeFleetStatus":"RUNNING","headNode":{"instanceType":"c7i.large","state":"running"},"scheduler":{"type":"slurm"},"tags":[{"key":"aws-parallelcluster-monitor-bucket","value":"s3://ursa-bucket"}]}
JSON
""",
    )

    monkeypatch.setenv("URSA_DAYLILY_EC_BIN", str(daylily_ec))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("HOME", str(home_dir))

    store = MemoryResourceStore()
    job = store.create_cluster_job(
        cluster_name="cluster-1",
        region="us-west-2",
        region_az="us-west-2d",
        atlas_tenant_id="TEN-1",
        owner_user_id="user-1",
        sponsor_user_id="user-2",
        request={
            "cluster_name": "cluster-1",
            "region": "us-west-2",
            "region_az": "us-west-2d",
            "ssh_key_name": "omics-key",
            "s3_bucket_name": "ursa-bucket",
            "aws_profile": None,
            "contact_email": "ops@example.com",
            "pass_on_warn": False,
            "debug": False,
        },
    )

    run_cluster_create_job(
        resource_store=store,
        cluster_service=ClusterService(regions=["us-west-2"]),
        workspace_root=tmp_path,
        job_euid=job.job_euid,
    )

    updated = store.get_cluster_job(job.job_euid)
    assert updated is not None
    assert updated.state == "COMPLETED"
    assert updated.return_code == 0
    assert updated.cluster["cluster_name"] == "cluster-1"
    assert updated.cluster["cluster_status"] == "CREATE_COMPLETE"
    assert [event.event_type for event in updated.events] == ["runner", "preflight", "create"]
    assert not (home_dir / ".ursa" / "cluster-create").exists()
