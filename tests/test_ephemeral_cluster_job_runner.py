"""Unit tests for the ephemeral cluster job runner."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def test_job_runner_executes_command_and_updates_job_file(tmp_path: Path) -> None:
    from daylib.ephemeral_cluster import job_runner

    job_path = tmp_path / "job.json"
    log_path = tmp_path / "job.log"

    job_doc = {
        "job_id": "ec_test_job_1",
        "status": "queued",
        "created_at": _now_iso(),
        "started_at": None,
        "completed_at": None,
        "return_code": None,
        "error": None,
        "cluster_name": "test-cluster",
        "region_az": "us-west-2a",
        "aws_profile": None,
        "config_path": str(tmp_path / "cfg.yaml"),
        "log_path": str(log_path),
        "command": [sys.executable, "-c", "print('ok')"],
        "env_overrides": {"PYTHONUNBUFFERED": "1"},
        "runner_pid": None,
    }
    job_path.write_text(json.dumps(job_doc, indent=2), encoding="utf-8")

    rc = job_runner.main(["--job-file", str(job_path)])

    assert rc == 0
    updated = json.loads(job_path.read_text(encoding="utf-8"))
    assert updated["status"] == "completed"
    assert updated["return_code"] == 0
    assert updated["completed_at"]

    content = log_path.read_text(encoding="utf-8")
    assert "Ursa cluster-create job started" in content
    assert "ok" in content
