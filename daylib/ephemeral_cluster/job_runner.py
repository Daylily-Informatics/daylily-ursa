from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from daylib.ephemeral_cluster.runner import _atomic_write_json, _read_json


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a persisted Ursa cluster-create job.")
    parser.add_argument("--job-file", required=True, help="Path to cluster-create job JSON")
    args = parser.parse_args(argv)

    job_path = Path(args.job_file).expanduser()
    job: Dict[str, Any] = _read_json(job_path)

    job["status"] = "running"
    job["started_at"] = job.get("started_at") or _now_iso()
    job["runner_pid"] = os.getpid()
    _atomic_write_json(job_path, job)

    log_path = Path(str(job.get("log_path") or "")).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env_overrides = job.get("env_overrides") or {}
    env = os.environ.copy()
    for k, v in env_overrides.items():
        if v is None:
            continue
        env[str(k)] = str(v)

    cmd = job.get("command")
    if not isinstance(cmd, list) or not cmd:
        job["status"] = "failed"
        job["completed_at"] = _now_iso()
        job["return_code"] = 2
        job["error"] = "Invalid command in job file"
        _atomic_write_json(job_path, job)
        return 2

    rc = 1
    try:
        with open(log_path, "a", buffering=1, encoding="utf-8") as log_f:
            log_f.write(f"[{_now_iso()}] Ursa cluster-create job started\\n")
            log_f.write(f"job_id={job.get('job_id')}\\n")
            log_f.write(f"command={' '.join(str(x) for x in cmd)}\\n\\n")
            proc = subprocess.run(
                [str(x) for x in cmd],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                check=False,
            )
            rc = int(proc.returncode)
    except Exception as e:
        job["status"] = "failed"
        job["completed_at"] = _now_iso()
        job["return_code"] = 1
        job["error"] = f"{type(e).__name__}: {e}"
        _atomic_write_json(job_path, job)
        return 1

    job["completed_at"] = _now_iso()
    job["return_code"] = rc
    job["status"] = "completed" if rc == 0 else "failed"
    job["error"] = None if rc == 0 else job.get("error")
    _atomic_write_json(job_path, job)
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

