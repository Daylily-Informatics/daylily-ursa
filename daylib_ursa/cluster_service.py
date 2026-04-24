"""Cluster service backed by the daylily-ephemeral-cluster 2.1.3 contract."""

from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

from daylib_ursa.ephemeral_cluster.runner import DaylilyEcClient, get_daylily_ec_client
from daylib_ursa.security import sanitize_for_log


_CLUSTER_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9-]{0,59}$")
_global_service: Optional["ClusterService"] = None
_global_service_lock = threading.Lock()


@dataclass
class BudgetInfo:
    project_name: Optional[str] = None
    region: Optional[str] = None
    reference_bucket: Optional[str] = None
    total_budget: Optional[float] = None
    used_budget: Optional[float] = None
    percent_used: Optional[float] = None
    fetched_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "region": self.region,
            "reference_bucket": self.reference_bucket,
            "total_budget": self.total_budget,
            "used_budget": self.used_budget,
            "percent_used": self.percent_used,
            "fetched_at": self.fetched_at,
            "error": self.error,
        }


@dataclass
class JobInfo:
    job_id: str
    partition: str
    cpus: int
    state: str
    state_short: str
    nodelist: str
    min_memory: str
    time_used: str
    nodes: int
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "partition": self.partition,
            "cpus": self.cpus,
            "state": self.state,
            "state_short": self.state_short,
            "nodelist": self.nodelist,
            "min_memory": self.min_memory,
            "time_used": self.time_used,
            "nodes": self.nodes,
            "name": self.name,
        }


@dataclass
class JobQueueSummary:
    total_jobs: int = 0
    running_jobs: int = 0
    pending_jobs: int = 0
    configuring_jobs: int = 0
    other_jobs: int = 0
    total_cpus: int = 0
    jobs: List[JobInfo] = field(default_factory=list)
    fetched_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_jobs": self.total_jobs,
            "running_jobs": self.running_jobs,
            "pending_jobs": self.pending_jobs,
            "configuring_jobs": self.configuring_jobs,
            "other_jobs": self.other_jobs,
            "total_cpus": self.total_cpus,
            "jobs": [job.to_dict() for job in self.jobs],
            "fetched_at": self.fetched_at,
            "error": self.error,
        }


@dataclass
class HeadNode:
    instance_type: str = ""
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    state: str = "unknown"
    instance_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeadNode":
        return cls(
            instance_type=str(data.get("instanceType") or ""),
            public_ip=str(data.get("publicIpAddress") or "").strip() or None,
            private_ip=str(data.get("privateIpAddress") or "").strip() or None,
            state=str(data.get("state") or "unknown"),
            instance_id=str(data.get("instanceId") or "").strip() or None,
        )


@dataclass
class ClusterInfo:
    cluster_name: str
    region: str
    cluster_status: str = "UNKNOWN"
    compute_fleet_status: str = "UNKNOWN"
    creation_time: Optional[str] = None
    last_updated_time: Optional[str] = None
    head_node: Optional[HeadNode] = None
    scheduler_type: str = "slurm"
    tags: Dict[str, str] = field(default_factory=dict)
    version: Optional[str] = None
    error_message: Optional[str] = None
    budget_info: Optional[BudgetInfo] = None
    job_queue: Optional[JobQueueSummary] = None

    MONITOR_BUCKET_TAG = "aws-parallelcluster-monitor-bucket"

    @classmethod
    def from_dict(cls, data: Dict[str, Any], region: str) -> "ClusterInfo":
        head_node_data = data.get("headNode")
        head_node = HeadNode.from_dict(head_node_data) if isinstance(head_node_data, dict) else None
        scheduler = data.get("scheduler", {})
        tags_list = data.get("tags", [])
        tags = {
            str(item.get("key") or ""): str(item.get("value") or "")
            for item in tags_list
            if isinstance(item, dict) and item.get("key")
        }
        return cls(
            cluster_name=str(data.get("clusterName") or data.get("name") or ""),
            region=region,
            cluster_status=str(data.get("clusterStatus") or data.get("status") or "UNKNOWN"),
            compute_fleet_status=str(data.get("computeFleetStatus") or "UNKNOWN"),
            creation_time=str(data.get("creationTime") or data.get("created_at") or "").strip()
            or None,
            last_updated_time=str(
                data.get("lastUpdatedTime") or data.get("updated_at") or ""
            ).strip()
            or None,
            head_node=head_node,
            scheduler_type=str(scheduler.get("type") or "slurm")
            if isinstance(scheduler, dict)
            else "slurm",
            tags=tags,
            version=str(data.get("version") or "").strip() or None,
        )

    @classmethod
    def from_dayec_row(cls, row: Dict[str, Any], region: str) -> "ClusterInfo":
        details = row.get("details")
        if isinstance(details, dict) and details:
            return cls.from_dict(details, region=region)
        public_ip = str(row.get("ip") or "").strip()
        instance_id = str(row.get("instance_id") or "").strip()
        return cls(
            cluster_name=str(row.get("name") or ""),
            region=region,
            cluster_status=str(row.get("status") or "UNKNOWN"),
            creation_time=str(row.get("created_at") or "").strip() or None,
            last_updated_time=str(row.get("updated_at") or "").strip() or None,
            head_node=HeadNode(public_ip=public_ip or None, instance_id=instance_id or None),
        )

    def get_monitor_bucket(self) -> Optional[str]:
        return self.tags.get(self.MONITOR_BUCKET_TAG)

    def get_monitor_bucket_name(self) -> Optional[str]:
        bucket = self.get_monitor_bucket()
        if not bucket:
            return None
        if bucket.startswith("s3://"):
            bucket = bucket[5:]
        return bucket.split("/")[0]

    def to_dict(self, include_sensitive: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "cluster_name": self.cluster_name,
            "region": self.region,
            "cluster_status": self.cluster_status,
            "compute_fleet_status": self.compute_fleet_status,
            "creation_time": self.creation_time,
            "last_updated_time": self.last_updated_time,
            "head_node": {
                "instance_type": self.head_node.instance_type,
                "public_ip": self.head_node.public_ip,
                "private_ip": self.head_node.private_ip,
                "state": self.head_node.state,
                "instance_id": self.head_node.instance_id,
            }
            if self.head_node
            else None,
            "scheduler_type": self.scheduler_type,
            "tags": self.tags,
            "version": self.version,
            "error_message": self.error_message,
            "budget_info": None,
            "job_queue": None,
            "monitor_bucket": self.get_monitor_bucket(),
        }
        if include_sensitive:
            result["budget_info"] = self.budget_info.to_dict() if self.budget_info else None
            result["job_queue"] = self.job_queue.to_dict() if self.job_queue else None
        return result


class ClusterService:
    """Cluster inventory and lifecycle operations via daylily-ec."""

    def __init__(
        self,
        regions: List[str],
        aws_profile: Optional[str] = None,
        cache_ttl_seconds: int = 300,
        client: DaylilyEcClient | None = None,
    ) -> None:
        if not regions:
            raise ValueError("At least one AWS region is required for ClusterService")
        self.regions = [str(region).strip() for region in regions if str(region).strip()]
        if not self.regions:
            raise ValueError("At least one AWS region is required for ClusterService")
        self.aws_profile = aws_profile
        self.cache_ttl_seconds = cache_ttl_seconds
        self.client = client or get_daylily_ec_client(aws_profile=aws_profile)
        self._cache: Dict[str, Any] = {}
        self._cache_time: float = 0
        self._cluster_region_map: Dict[str, str] = {}
        self._delete_tokens: Dict[str, tuple[str, str]] = {}

    def _validate_cluster_name(self, name: str) -> None:
        if not name or not _CLUSTER_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid cluster name: {sanitize_for_log(name, 50)}. "
                "Must start with a letter, contain only letters, numbers, and hyphens, "
                "and be 1-60 characters."
            )

    def get_region_for_cluster(self, cluster_name: str) -> Optional[str]:
        return self._cluster_region_map.get(cluster_name)

    def list_clusters_in_region(self, region: str) -> List[str]:
        payload = self.client.cluster_list(region=region, details=False)
        rows = payload.get("clusters")
        if not isinstance(rows, list):
            raise RuntimeError("daylily-ec cluster list returned invalid clusters payload")
        return [
            str(row.get("name") or row.get("clusterName") or "")
            for row in rows
            if isinstance(row, dict) and (row.get("name") or row.get("clusterName"))
        ]

    def describe_cluster(self, cluster_name: str, region: str) -> ClusterInfo:
        self._validate_cluster_name(cluster_name)
        payload = self.client.cluster_describe(cluster_name=cluster_name, region=region)
        return ClusterInfo.from_dict(payload, region=region)

    def create_delete_plan(self, cluster_name: str, region: str) -> Dict[str, Any]:
        self._validate_cluster_name(cluster_name)
        result = self.client.delete_dry_run(cluster_name=cluster_name, region=region)
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or result.stdout.strip() or "delete dry-run failed"
            )
        token = secrets.token_urlsafe(24)
        self._delete_tokens[token] = (cluster_name, region)
        return {
            "cluster_name": cluster_name,
            "region": region,
            "confirmation_token": token,
            "dry_run_stdout": result.stdout,
            "dry_run_stderr": result.stderr,
        }

    def delete_cluster(
        self,
        cluster_name: str,
        region: str,
        *,
        confirmation_token: str,
        confirm_cluster_name: str,
    ) -> Dict[str, Any]:
        self._validate_cluster_name(cluster_name)
        if confirm_cluster_name != cluster_name:
            raise ValueError("confirm_cluster_name must exactly match cluster_name")
        expected = self._delete_tokens.pop(confirmation_token, None)
        if expected != (cluster_name, region):
            raise ValueError("Invalid or expired cluster delete confirmation token")
        result = self.client.delete(cluster_name=cluster_name, region=region)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "delete failed")
        self.clear_cache()
        return {
            "cluster_name": cluster_name,
            "region": region,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def _scan_region(self, region: str) -> List[ClusterInfo]:
        payload = self.client.cluster_list(region=region, details=True)
        rows = payload.get("clusters")
        if not isinstance(rows, list):
            raise RuntimeError("daylily-ec cluster list returned invalid clusters payload")
        return [
            ClusterInfo.from_dayec_row(cast(Dict[str, Any], row), region=region)
            for row in rows
            if isinstance(row, dict)
        ]

    def get_all_clusters(self, force_refresh: bool = False) -> List[ClusterInfo]:
        now = time.time()
        if not force_refresh and self._cache and (now - self._cache_time) < self.cache_ttl_seconds:
            return cast(List[ClusterInfo], self._cache["clusters"])

        clusters: List[ClusterInfo] = []
        for region in self.regions:
            clusters.extend(self._scan_region(region))
        self._cache = {"clusters": clusters}
        self._cache_time = now
        self._cluster_region_map = {
            cluster.cluster_name: cluster.region for cluster in clusters if cluster.cluster_name
        }
        return clusters

    def get_clusters_by_region(self, force_refresh: bool = False) -> Dict[str, List[ClusterInfo]]:
        grouped = {region: [] for region in self.regions}
        for cluster in self.get_all_clusters(force_refresh=force_refresh):
            grouped.setdefault(cluster.region, []).append(cluster)
        return grouped

    def get_cluster_by_name(
        self, cluster_name: str, force_refresh: bool = False
    ) -> Optional[ClusterInfo]:
        for cluster in self.get_all_clusters(force_refresh=force_refresh):
            if cluster.cluster_name == cluster_name:
                return cluster
        return None

    def get_bucket_for_cluster(
        self, cluster_name: str, force_refresh: bool = False
    ) -> Optional[str]:
        cluster = self.get_cluster_by_name(cluster_name, force_refresh=force_refresh)
        return cluster.get_monitor_bucket_name() if cluster else None

    def _parse_squeue_output(self, output: str) -> JobQueueSummary:
        summary = JobQueueSummary(fetched_at=datetime.now(timezone.utc).isoformat())
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        header_idx = next(
            (idx for idx, line in enumerate(lines) if line.startswith("JOBID")),
            None,
        )
        if header_idx is None:
            return summary
        for line in lines[header_idx + 1 :]:
            parts = line.split()
            if len(parts) < 11:
                continue
            job = JobInfo(
                job_id=parts[0],
                partition=parts[1],
                cpus=int(parts[2]) if parts[2].isdigit() else 0,
                state_short=parts[3],
                nodelist=parts[4],
                min_memory=parts[7],
                time_used=parts[8],
                nodes=int(parts[9]) if parts[9].isdigit() else 0,
                name=parts[10],
                state=parts[6],
            )
            summary.jobs.append(job)
            summary.total_jobs += 1
            summary.total_cpus += job.cpus
            state_upper = job.state.upper()
            if state_upper == "RUNNING" or job.state_short == "R":
                summary.running_jobs += 1
            elif state_upper == "PENDING" or job.state_short == "PD":
                summary.pending_jobs += 1
            elif state_upper == "CONFIGURING" or job.state_short == "CF":
                summary.configuring_jobs += 1
            else:
                summary.other_jobs += 1
        return summary

    def fetch_headnode_status(self, cluster: ClusterInfo) -> ClusterInfo:
        result = self.client.run(
            [
                "headnode",
                "jobs",
                "--region",
                cluster.region,
                "--cluster",
                cluster.cluster_name,
            ]
            + (["--profile", self.aws_profile] if self.aws_profile else [])
        )
        if result.returncode != 0:
            cluster.job_queue = JobQueueSummary(
                fetched_at=datetime.now(timezone.utc).isoformat(),
                error=result.stderr.strip() or result.stdout.strip() or "headnode jobs failed",
            )
            return cluster
        cluster.job_queue = self._parse_squeue_output(result.stdout)
        return cluster

    def get_all_clusters_with_status(
        self,
        force_refresh: bool = False,
        fetch_ssh_status: bool = True,
        ssh_key_pattern: str = "",
    ) -> List[ClusterInfo]:
        _ = ssh_key_pattern
        clusters = self.get_all_clusters(force_refresh=force_refresh)
        if fetch_ssh_status:
            for cluster in clusters:
                self.fetch_headnode_status(cluster)
        return clusters

    def clear_cache(self) -> None:
        self._cache = {}
        self._cache_time = 0


def get_cluster_service(
    regions: Optional[List[str]] = None,
    aws_profile: Optional[str] = None,
    cache_ttl_seconds: int = 300,
) -> ClusterService:
    global _global_service
    with _global_service_lock:
        if _global_service is None:
            resolved_regions = list(regions or [])
            resolved_profile = aws_profile
            if not resolved_regions:
                from daylib_ursa.ursa_config import get_ursa_config

                ursa_config = get_ursa_config()
                if ursa_config.is_configured:
                    resolved_regions = ursa_config.get_allowed_regions()
                    resolved_profile = resolved_profile or ursa_config.aws_profile
            if not resolved_regions:
                raise RuntimeError("No Ursa cluster regions configured")
            _global_service = ClusterService(
                regions=resolved_regions,
                aws_profile=resolved_profile,
                cache_ttl_seconds=cache_ttl_seconds,
            )
        return _global_service


def reset_cluster_service() -> None:
    global _global_service
    with _global_service_lock:
        _global_service = None
