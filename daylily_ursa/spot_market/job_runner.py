from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from daylily_ursa.spot_market.runner import (
    _atomic_write_json,
    _read_json,
    compute_and_store_snapshot,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a persisted Ursa spot-market poll job.")
    parser.add_argument("--job-file", required=True, help="Path to spot-market job JSON")
    args = parser.parse_args(argv)

    job_path = Path(args.job_file).expanduser()
    job: Dict[str, Any] = _read_json(job_path)

    job["status"] = "running"
    job["started_at"] = job.get("started_at") or _now_iso()
    job["runner_pid"] = os.getpid()
    _atomic_write_json(job_path, job)

    log_path = Path(str(job.get("log_path") or "")).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    region = str(job.get("region") or "")
    aws_profile = job.get("aws_profile")
    cfg = job.get("cfg") or {}

    rc = 1
    try:
        with open(log_path, "a", buffering=1, encoding="utf-8") as log_f:
            log_f.write(f"[{_now_iso()}] Ursa spot-market poll started\\n")
            log_f.write(
                f"job_id={job.get('job_id')} region={region} aws_profile={aws_profile}\\n\\n"
            )

            snapshot_path = compute_and_store_snapshot(
                region=region,
                aws_profile=str(aws_profile) if aws_profile else None,
                cfg=cfg,
            )
            job["snapshot_path"] = str(snapshot_path)
            log_f.write(f"snapshot_path={snapshot_path}\\n")
            rc = 0
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
