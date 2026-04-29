"""Cluster service backed by the daylily-ephemeral-cluster 2.1.12 contract."""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

from daylib_ursa.ephemeral_cluster.runner import (
    REQUIRED_DAYLILY_EC_VERSION,
    DaylilyEcClient,
    get_daylily_ec_client,
)
from daylib_ursa.security import sanitize_for_log


_CLUSTER_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9-]{0,59}$")
_STATIC_PROBE_TTL_SECONDS = 24 * 60 * 60
_DYNAMIC_PROBE_TTL_SECONDS = 5 * 60
_global_service: Optional["ClusterService"] = None
_global_service_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _aws_console_url(region: str, instance_id: str | None) -> str | None:
    if not instance_id:
        return None
    return (
        f"https://{region}.console.aws.amazon.com/ec2/home?region={region}"
        f"#InstanceDetails:instanceId={instance_id}"
    )


def _section_between(text: str, start_marker: str, end_marker: str) -> str:
    start_idx = text.find(start_marker)
    if start_idx < 0:
        return ""
    start_idx += len(start_marker)
    end_idx = text.find(end_marker, start_idx)
    if end_idx < 0:
        end_idx = len(text)
    return text[start_idx:end_idx].strip()


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
class HeadnodeProbeResult:
    probe_type: str
    cluster_name: str
    region: str
    instance_id: Optional[str]
    captured_at: str
    cache_expires_at: str
    ttl_seconds: int
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self, *, cached: bool = False) -> Dict[str, Any]:
        return {
            "probe_type": self.probe_type,
            "cluster_name": self.cluster_name,
            "region": self.region,
            "instance_id": self.instance_id,
            "captured_at": self.captured_at,
            "cache_expires_at": self.cache_expires_at,
            "ttl_seconds": self.ttl_seconds,
            "cached": cached,
            "data": dict(self.data),
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
    headnode_probes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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

    def aws_console_url(self) -> str | None:
        instance_id = self.head_node.instance_id if self.head_node else None
        return _aws_console_url(self.region, instance_id)

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
            "daylily_ec_pinned_version": REQUIRED_DAYLILY_EC_VERSION,
            "aws_console_url": self.aws_console_url(),
            "headnode_probes": dict(self.headnode_probes),
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
        self._probe_cache: Dict[tuple[str, str, str], tuple[float, HeadnodeProbeResult]] = {}
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
        names = [
            str(row.get("name") or row.get("clusterName") or "")
            for row in rows
            if isinstance(row, dict) and (row.get("name") or row.get("clusterName"))
        ]
        for name in names:
            self._cluster_region_map[name] = region
        return names

    def describe_cluster(self, cluster_name: str, region: str) -> ClusterInfo:
        self._validate_cluster_name(cluster_name)
        payload = self.client.cluster_describe(cluster_name=cluster_name, region=region)
        cluster = ClusterInfo.from_dict(payload, region=region)
        self._attach_cached_probes(cluster)
        return cluster

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
        clusters = [
            ClusterInfo.from_dayec_row(cast(Dict[str, Any], row), region=region)
            for row in rows
            if isinstance(row, dict)
        ]
        for cluster in clusters:
            self._attach_cached_probes(cluster)
        return clusters

    def get_all_clusters(self, force_refresh: bool = False) -> List[ClusterInfo]:
        now = time.time()
        if not force_refresh and self._cache and (now - self._cache_time) < self.cache_ttl_seconds:
            clusters = cast(List[ClusterInfo], self._cache["clusters"])
            for cluster in clusters:
                self._attach_cached_probes(cluster)
            return clusters

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
        for cluster in clusters:
            if fetch_ssh_status:
                self.fetch_headnode_status(cluster)
            self._attach_cached_probes(cluster)
        return clusters

    def clear_cache(self) -> None:
        self._cache = {}
        self._cache_time = 0

    def _attach_cached_probes(self, cluster: ClusterInfo) -> None:
        probes: Dict[str, Dict[str, Any]] = {}
        now = time.time()
        for probe_type in ("static", "scheduler", "fsx"):
            cached = self._probe_cache.get((probe_type, cluster.region, cluster.cluster_name))
            if not cached:
                continue
            expires_at, result = cached
            if expires_at <= now:
                continue
            probes[probe_type] = result.to_dict(cached=True)
        cluster.headnode_probes = probes

    def _resolve_probe_target(self, cluster_name: str, region: str) -> ClusterInfo:
        cluster = self.describe_cluster(cluster_name, region)
        instance_id = cluster.head_node.instance_id if cluster.head_node else None
        if not instance_id:
            raise ValueError(f"Headnode EC2 instance id is unavailable for {cluster_name}")
        return cluster

    def _cached_probe(
        self,
        *,
        probe_type: str,
        cluster_name: str,
        region: str,
        ttl_seconds: int,
        refresh: bool,
    ) -> HeadnodeProbeResult | None:
        if refresh:
            return None
        cached = self._probe_cache.get((probe_type, region, cluster_name))
        if not cached:
            return None
        expires_at, result = cached
        if expires_at <= time.time():
            return None
        _ = ttl_seconds
        return result

    def _store_probe(self, result: HeadnodeProbeResult) -> None:
        self._probe_cache[(result.probe_type, result.region, result.cluster_name)] = (
            time.time() + result.ttl_seconds,
            result,
        )

    def _probe_error_result(
        self,
        *,
        probe_type: str,
        cluster_name: str,
        region: str,
        instance_id: str | None,
        ttl_seconds: int,
        error: str,
        data: Dict[str, Any] | None = None,
    ) -> HeadnodeProbeResult:
        captured_at = _utc_now_iso()
        return HeadnodeProbeResult(
            probe_type=probe_type,
            cluster_name=cluster_name,
            region=region,
            instance_id=instance_id,
            captured_at=captured_at,
            cache_expires_at=datetime.fromtimestamp(
                time.time() + ttl_seconds,
                tz=timezone.utc,
            ).isoformat(),
            ttl_seconds=ttl_seconds,
            data=dict(data or {}),
            error=error,
        )

    def _run_headnode_script(
        self,
        *,
        cluster_name: str,
        region: str,
        probe_type: str,
        script: str,
        ttl_seconds: int,
        timeout: int,
    ) -> tuple[ClusterInfo | None, Any | None, HeadnodeProbeResult | None]:
        from daylily_ec.aws.ssm import (
            SsmCommandFailedError,
            SsmError,
            run_shell,
            wait_for_ssm_online,
        )

        instance_id: str | None = None
        try:
            cluster = self._resolve_probe_target(cluster_name, region)
            instance_id = cluster.head_node.instance_id if cluster.head_node else None
            if not instance_id:
                raise ValueError(f"Headnode EC2 instance id is unavailable for {cluster_name}")

            wait_for_ssm_online(
                instance_id,
                region,
                profile=self.aws_profile,
                timeout=30,
                poll_interval=3,
            )
            result = run_shell(
                instance_id,
                region,
                script,
                profile=self.aws_profile,
                timeout=timeout,
                comment=f"Ursa {probe_type} headnode probe for {cluster_name}",
            )
            return cluster, result, None
        except SsmCommandFailedError as exc:
            error = exc.result.stderr.strip() or exc.result.stdout.strip() or str(exc)
            return (
                None,
                None,
                self._probe_error_result(
                    probe_type=probe_type,
                    cluster_name=cluster_name,
                    region=region,
                    instance_id=instance_id,
                    ttl_seconds=ttl_seconds,
                    error=error,
                    data={
                        "stdout": exc.result.stdout,
                        "stderr": exc.result.stderr,
                        "response_code": exc.result.response_code,
                        "status": exc.result.status,
                    },
                ),
            )
        except (SsmError, TimeoutError, RuntimeError, ValueError) as exc:
            return (
                None,
                None,
                self._probe_error_result(
                    probe_type=probe_type,
                    cluster_name=cluster_name,
                    region=region,
                    instance_id=instance_id,
                    ttl_seconds=ttl_seconds,
                    error=str(exc),
                ),
            )

    def _build_probe_result(
        self,
        *,
        probe_type: str,
        cluster_name: str,
        region: str,
        instance_id: str | None,
        ttl_seconds: int,
        data: Dict[str, Any],
        error: str | None = None,
    ) -> HeadnodeProbeResult:
        captured_at = _utc_now_iso()
        return HeadnodeProbeResult(
            probe_type=probe_type,
            cluster_name=cluster_name,
            region=region,
            instance_id=instance_id,
            captured_at=captured_at,
            cache_expires_at=datetime.fromtimestamp(
                time.time() + ttl_seconds,
                tz=timezone.utc,
            ).isoformat(),
            ttl_seconds=ttl_seconds,
            data=data,
            error=error,
        )

    def fetch_headnode_static_probe(
        self,
        *,
        cluster_name: str,
        region: str,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        cached = self._cached_probe(
            probe_type="static",
            cluster_name=cluster_name,
            region=region,
            ttl_seconds=_STATIC_PROBE_TTL_SECONDS,
            refresh=refresh,
        )
        if cached:
            return cached.to_dict(cached=True)

        script = "\n".join(
            [
                "set +e",
                'printf "__DAYLILY_EC_VERSION_BEGIN__\\n"',
                "if command -v daylily-ec >/dev/null 2>&1; then",
                "  daylily-ec --json version",
                "else",
                '  printf "daylily-ec unavailable\\n"',
                "fi",
                'printf "\\n__DAYLILY_EC_VERSION_END__\\n"',
                'printf "__DAY_CLONE_HELP_BEGIN__\\n"',
                "if command -v day-clone >/dev/null 2>&1; then",
                "  day-clone --help",
                "  rc=$?",
                '  if [ "$rc" -ne 0 ]; then printf "\\nday-clone --help exited %s\\n" "$rc"; fi',
                "else",
                '  printf "day-clone unavailable\\n"',
                "fi",
                'printf "\\n__DAY_CLONE_HELP_END__\\n"',
            ]
        )
        cluster, result, error_result = self._run_headnode_script(
            cluster_name=cluster_name,
            region=region,
            probe_type="static",
            script=script,
            ttl_seconds=_STATIC_PROBE_TTL_SECONDS,
            timeout=90,
        )
        if error_result:
            self._store_probe(error_result)
            return error_result.to_dict(cached=False)
        assert result is not None
        assert cluster is not None

        version_text = _section_between(
            result.stdout,
            "__DAYLILY_EC_VERSION_BEGIN__",
            "__DAYLILY_EC_VERSION_END__",
        )
        day_clone_help = _section_between(
            result.stdout,
            "__DAY_CLONE_HELP_BEGIN__",
            "__DAY_CLONE_HELP_END__",
        )
        remote_version = ""
        remote_git_hash = ""
        if version_text:
            try:
                version_payload = cast(Dict[str, Any], json.loads(version_text))
                if isinstance(version_payload, dict):
                    remote_version = str(version_payload.get("version") or "").strip()
                    remote_git_hash = str(
                        version_payload.get("git_hash")
                        or version_payload.get("git_sha")
                        or version_payload.get("commit")
                        or ""
                    ).strip()
            except Exception:
                remote_version = version_text.splitlines()[0].strip()

        data = {
            "daylily_ec_pinned_version": REQUIRED_DAYLILY_EC_VERSION,
            "remote_daylily_ec_version": remote_version,
            "remote_git_hash": remote_git_hash,
            "day_clone_available": bool(day_clone_help)
            and "day-clone unavailable" not in day_clone_help,
            "day_clone_help": day_clone_help,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        probe_result = self._build_probe_result(
            probe_type="static",
            cluster_name=cluster_name,
            region=region,
            instance_id=cluster.head_node.instance_id if cluster.head_node else None,
            ttl_seconds=_STATIC_PROBE_TTL_SECONDS,
            data=data,
        )
        self._store_probe(probe_result)
        return probe_result.to_dict(cached=False)

    def fetch_headnode_scheduler_probe(
        self,
        *,
        cluster_name: str,
        region: str,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        cached = self._cached_probe(
            probe_type="scheduler",
            cluster_name=cluster_name,
            region=region,
            ttl_seconds=_DYNAMIC_PROBE_TTL_SECONDS,
            refresh=refresh,
        )
        if cached:
            return cached.to_dict(cached=True)

        script = "\n".join(
            [
                "set +e",
                'printf "__SQUEUE_BEGIN__\\n"',
                "squeue",
                'printf "\\n__SQUEUE_END__\\n"',
                'printf "__SINFO_BEGIN__\\n"',
                "sinfo",
                'printf "\\n__SINFO_END__\\n"',
            ]
        )
        cluster, result, error_result = self._run_headnode_script(
            cluster_name=cluster_name,
            region=region,
            probe_type="scheduler",
            script=script,
            ttl_seconds=_DYNAMIC_PROBE_TTL_SECONDS,
            timeout=90,
        )
        if error_result:
            self._store_probe(error_result)
            return error_result.to_dict(cached=False)
        assert result is not None
        assert cluster is not None
        probe_result = self._build_probe_result(
            probe_type="scheduler",
            cluster_name=cluster_name,
            region=region,
            instance_id=cluster.head_node.instance_id if cluster.head_node else None,
            ttl_seconds=_DYNAMIC_PROBE_TTL_SECONDS,
            data={
                "squeue_output": _section_between(
                    result.stdout,
                    "__SQUEUE_BEGIN__",
                    "__SQUEUE_END__",
                ),
                "sinfo_output": _section_between(
                    result.stdout,
                    "__SINFO_BEGIN__",
                    "__SINFO_END__",
                ),
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        self._store_probe(probe_result)
        return probe_result.to_dict(cached=False)

    def fetch_headnode_fsx_probe(
        self,
        *,
        cluster_name: str,
        region: str,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        cached = self._cached_probe(
            probe_type="fsx",
            cluster_name=cluster_name,
            region=region,
            ttl_seconds=_DYNAMIC_PROBE_TTL_SECONDS,
            refresh=refresh,
        )
        if cached:
            return cached.to_dict(cached=True)

        script = "\n".join(
            [
                "set +e",
                'printf "__DF_FSX_BEGIN__\\n"',
                "df -h /fsx",
                'printf "\\n__DF_FSX_END__\\n"',
            ]
        )
        cluster, result, error_result = self._run_headnode_script(
            cluster_name=cluster_name,
            region=region,
            probe_type="fsx",
            script=script,
            ttl_seconds=_DYNAMIC_PROBE_TTL_SECONDS,
            timeout=60,
        )
        if error_result:
            self._store_probe(error_result)
            return error_result.to_dict(cached=False)
        assert result is not None
        assert cluster is not None
        probe_result = self._build_probe_result(
            probe_type="fsx",
            cluster_name=cluster_name,
            region=region,
            instance_id=cluster.head_node.instance_id if cluster.head_node else None,
            ttl_seconds=_DYNAMIC_PROBE_TTL_SECONDS,
            data={
                "df_output": _section_between(
                    result.stdout,
                    "__DF_FSX_BEGIN__",
                    "__DF_FSX_END__",
                ),
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        self._store_probe(probe_result)
        return probe_result.to_dict(cached=False)


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
