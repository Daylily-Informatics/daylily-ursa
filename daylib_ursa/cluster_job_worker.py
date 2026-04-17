from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from daylib_ursa.cluster_jobs import run_cluster_create_job
from daylib_ursa.cluster_service import ClusterService
from daylib_ursa.config import get_settings
from daylib_ursa.resource_store import ResourceStore

LOGGER = logging.getLogger("daylily.ursa.cluster_job_worker")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single Ursa cluster job worker process")
    parser.add_argument("--job-euid", required=True, help="TapDB EUID for the cluster job")
    parser.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
        help="Workspace root used for short-lived scratch execution",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    settings = get_settings()
    resource_store = ResourceStore()
    job = resource_store.get_cluster_job(args.job_euid)
    if job is None:
        raise RuntimeError(f"Cluster job not found: {args.job_euid}")

    request_payload = dict(job.request or {})
    cluster_service = ClusterService(
        regions=settings.get_allowed_regions(),
        aws_profile=str(request_payload.get("aws_profile") or settings.aws_profile or "").strip()
        or None,
    )
    run_cluster_create_job(
        resource_store=resource_store,
        cluster_service=cluster_service,
        workspace_root=Path(args.workspace_root).resolve(),
        job_euid=args.job_euid,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
