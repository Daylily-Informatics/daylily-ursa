"""Cluster service for fetching AWS ParallelCluster information.

Uses pcluster CLI to list and describe clusters across multiple regions.
Implements caching to reduce API calls.
Includes SSH-based budget and job queue fetching for running headnodes.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

LOGGER = logging.getLogger("daylily.cluster_service")

# Global singleton instance and lock
_global_service: Optional["ClusterService"] = None
_global_service_lock = threading.Lock()


@dataclass
class BudgetInfo:
    """AWS Budget information from headnode."""

    project_name: Optional[str] = None
    region: Optional[str] = None
    reference_bucket: Optional[str] = None
    total_budget: Optional[float] = None
    used_budget: Optional[float] = None
    percent_used: Optional[float] = None
    fetched_at: Optional[str] = None
    error: Optional[str] = None
    ssh_command: Optional[str] = None  # For debugging

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
            "ssh_command": self.ssh_command,
        }


@dataclass
class JobInfo:
    """Slurm job information."""

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
    """Summary of Slurm job queue."""

    total_jobs: int = 0
    running_jobs: int = 0
    pending_jobs: int = 0
    configuring_jobs: int = 0
    other_jobs: int = 0
    total_cpus: int = 0
    jobs: List[JobInfo] = field(default_factory=list)
    fetched_at: Optional[str] = None
    error: Optional[str] = None
    ssh_command: Optional[str] = None  # For debugging

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_jobs": self.total_jobs,
            "running_jobs": self.running_jobs,
            "pending_jobs": self.pending_jobs,
            "configuring_jobs": self.configuring_jobs,
            "other_jobs": self.other_jobs,
            "total_cpus": self.total_cpus,
            "jobs": [j.to_dict() for j in self.jobs],
            "fetched_at": self.fetched_at,
            "error": self.error,
            "ssh_command": self.ssh_command,
        }


@dataclass
class HeadNode:
    """Head node information."""

    instance_type: str = ""
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    state: str = "unknown"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeadNode":
        """Create HeadNode from pcluster API response."""
        return cls(
            instance_type=data.get("instanceType", ""),
            public_ip=data.get("publicIpAddress"),
            private_ip=data.get("privateIpAddress"),
            state=data.get("state", "unknown"),
        )


@dataclass
class ClusterInfo:
    """Cluster information from pcluster describe-cluster."""

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any], region: str) -> "ClusterInfo":
        """Create ClusterInfo from pcluster API response."""
        head_node_data = data.get("headNode")
        head_node = HeadNode.from_dict(head_node_data) if head_node_data else None
        scheduler = data.get("scheduler", {})
        tags_list = data.get("tags", [])
        tags_dict = {t.get("key", ""): t.get("value", "") for t in tags_list if isinstance(t, dict)}
        return cls(
            cluster_name=data.get("clusterName", ""),
            region=region,
            cluster_status=data.get("clusterStatus", "UNKNOWN"),
            compute_fleet_status=data.get("computeFleetStatus", "UNKNOWN"),
            creation_time=data.get("creationTime"),
            last_updated_time=data.get("lastUpdatedTime"),
            head_node=head_node,
            scheduler_type=scheduler.get("type", "slurm") if isinstance(scheduler, dict) else "slurm",
            tags=tags_dict,
            version=data.get("version"),
        )

    # Tag key for monitor bucket (set during cluster creation)
    MONITOR_BUCKET_TAG = "aws-parallelcluster-monitor-bucket"

    def get_monitor_bucket(self) -> Optional[str]:
        """Get the monitor bucket from cluster tags.

        Returns:
            S3 bucket URI (e.g., 's3://my-bucket') or None if tag not set.
        """
        return self.tags.get(self.MONITOR_BUCKET_TAG)

    def get_monitor_bucket_name(self) -> Optional[str]:
        """Get the monitor bucket name (without s3:// prefix) from cluster tags.

        Returns:
            Bucket name or None if tag not set.
        """
        bucket = self.get_monitor_bucket()
        if not bucket:
            return None
        if bucket.startswith("s3://"):
            bucket = bucket[5:]
        # Remove any path component
        return bucket.split("/")[0]

    def to_dict(self, include_sensitive: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Args:
            include_sensitive: If False, omit/blank fields that expose cost/budget/queue details.
        """
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
            } if self.head_node else None,
            "scheduler_type": self.scheduler_type,
            "tags": self.tags,
            "version": self.version,
            "error_message": self.error_message,
            # Always provide keys so templates/callers don't KeyError.
            "budget_info": None,
            "job_queue": None,
            "monitor_bucket": self.get_monitor_bucket(),
        }

        if include_sensitive:
            result["budget_info"] = self.budget_info.to_dict() if self.budget_info else None
            result["job_queue"] = self.job_queue.to_dict() if self.job_queue else None

        return result


class ClusterService:
    """Service for fetching ParallelCluster information."""

    def __init__(
        self,
        regions: List[str],
        aws_profile: Optional[str] = None,
        cache_ttl_seconds: int = 300,
    ):
        """Initialize cluster service.

        Args:
            regions: List of AWS regions to scan for clusters.
            aws_profile: AWS profile name to use.
            cache_ttl_seconds: Cache TTL in seconds (default 5 minutes).
        """
        self.regions = regions
        self.aws_profile = aws_profile
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._cache_time: float = 0
        # Fast lookup cache: cluster_name -> region (persists longer than full cache)
        self._cluster_region_map: Dict[str, str] = {}
        self._cluster_region_map_time: float = 0
        self._cluster_region_map_ttl: int = 3600  # 1 hour TTL for name->region mapping

    def get_region_for_cluster(self, cluster_name: str) -> Optional[str]:
        """Fast lookup of region for a cluster name (uses cached mapping).

        This is much faster than get_all_clusters() as it uses a long-lived
        name->region cache that doesn't require pcluster CLI calls.

        Args:
            cluster_name: Name of the cluster.

        Returns:
            Region string if known, None otherwise.
        """
        now = time.time()

        # Check if cluster_region_map is still valid
        if (now - self._cluster_region_map_time) < self._cluster_region_map_ttl:
            if cluster_name in self._cluster_region_map:
                LOGGER.debug(f"Fast cache hit for cluster {cluster_name}")
                return self._cluster_region_map[cluster_name]

        # Check full cluster cache (might be more recent)
        if self._cache and (now - self._cache_time) < self.cache_ttl_seconds:
            cached_clusters = cast(List[ClusterInfo], self._cache.get("clusters", []))
            for cluster in cached_clusters:
                if cluster.cluster_name == cluster_name:
                    # Update region map
                    self._cluster_region_map[cluster_name] = cluster.region
                    self._cluster_region_map_time = now
                    return cluster.region

        # Not in any cache - return None (caller can decide to refresh)
        return None

    def _run_pcluster_command(self, args: List[str], timeout: int = 30) -> Dict[str, Any]:
        """Run a pcluster CLI command and return parsed JSON output."""
        import os
        import shutil

        # Find pcluster binary - prefer conda env path
        pcluster_path = shutil.which("pcluster")
        if not pcluster_path:
            LOGGER.error("pcluster CLI not found in PATH")
            return {"error": "pcluster CLI not installed"}

        cmd = [pcluster_path] + args
        # Set AWS_PROFILE - use explicit value or fall back to env (no 'default' fallback)
        profile = self.aws_profile or os.environ.get("AWS_PROFILE")
        if profile:
            LOGGER.info(f"Running: AWS_PROFILE={profile} {' '.join(cmd)}")
        else:
            LOGGER.warning(
                "No AWS profile configured for pcluster command. "
                "Set aws_profile in ~/.ursa/ursa-config.yaml"
            )
            LOGGER.info(f"Running: {' '.join(cmd)}")
        LOGGER.debug(f"PATH: {os.environ.get('PATH', 'not set')[:200]}")
        try:
            env = os.environ.copy()
            if profile:
                env["AWS_PROFILE"] = profile
            # Use explicit PIPE for stdin to avoid "bad file descriptor" errors
            # when running from ThreadPoolExecutor threads
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=env,
            )
            # Log stderr always for debugging
            if result.stderr.strip():
                LOGGER.info(f"pcluster stderr: {result.stderr.strip()[:500]}")

            # Try to parse stdout as JSON first - pcluster may emit warnings to stderr
            # even on success (e.g., pkg_resources deprecation warnings)
            if result.stdout.strip():
                try:
                    parsed = cast(Dict[str, Any], json.loads(result.stdout))
                    return parsed
                except json.JSONDecodeError:
                    LOGGER.warning(f"Failed to parse stdout as JSON: {result.stdout[:200]}")
                    pass  # Fall through to error handling

            # No valid JSON output - check return code
            if result.returncode != 0:
                LOGGER.warning(f"pcluster command failed (exit {result.returncode}): {result.stderr}")
                return {"error": result.stderr.strip() or f"Exit code {result.returncode}"}

            # Return code 0 but no output
            return {}
        except subprocess.TimeoutExpired:
            LOGGER.error(f"pcluster command timed out after {timeout}s")
            return {"error": f"Command timed out after {timeout}s"}
        except json.JSONDecodeError as e:
            LOGGER.error(f"Failed to parse pcluster output: {e}")
            return {"error": f"Invalid JSON output: {e}"}
        except FileNotFoundError:
            LOGGER.error("pcluster CLI not found")
            return {"error": "pcluster CLI not installed"}
        except Exception as e:
            LOGGER.error(f"pcluster command error: {e}")
            return {"error": str(e)}

    def list_clusters_in_region(self, region: str) -> List[str]:
        """List all cluster names in a region."""
        result = self._run_pcluster_command(["list-clusters", "--region", region])
        LOGGER.info(f"list-clusters {region} returned: {result}")
        if "error" in result:
            LOGGER.warning(f"Failed to list clusters in {region}: {result['error']}")
            return []
        clusters = result.get("clusters", [])
        names = [c.get("clusterName", "") for c in clusters if c.get("clusterName")]
        LOGGER.info(f"Found {len(names)} clusters in {region}: {names}")
        return names

    def describe_cluster(self, cluster_name: str, region: str) -> ClusterInfo:
        """Get detailed information about a cluster."""
        result = self._run_pcluster_command(
            ["describe-cluster", "--region", region, "-n", cluster_name]
        )
        if "error" in result:
            return ClusterInfo(
                cluster_name=cluster_name,
                region=region,
                error_message=result["error"],
            )
        return ClusterInfo.from_dict(result, region)

    def delete_cluster(self, cluster_name: str, region: str) -> Dict[str, Any]:
        """Initiate deletion of a ParallelCluster cluster.

        Note: ParallelCluster deletion is asynchronous; this method returns once the
        delete request is accepted by the CLI.
        """
        result = self._run_pcluster_command(
            ["delete-cluster", "--region", region, "-n", cluster_name],
            timeout=60,
        )
        if "error" not in result:
            # Cluster set has changed; clear caches so subsequent list calls refresh.
            self._cache = {}
            self._cache_time = 0
            self._cluster_region_map.pop(cluster_name, None)
            self._cluster_region_map_time = time.time()
        return result

    def _scan_region(self, region: str) -> List[ClusterInfo]:
        """Scan a single region for clusters (used by parallel executor).

        Args:
            region: AWS region to scan.

        Returns:
            List of ClusterInfo for clusters in that region.
        """
        LOGGER.debug(f"Scanning region: {region}")
        cluster_names = self.list_clusters_in_region(region)
        if not cluster_names:
            LOGGER.debug(f"No clusters found in {region}")
            return []

        # Parallelize describe_cluster calls within this region
        # Each pcluster describe-cluster takes ~1.5-2s, so this saves significant time
        clusters = []
        if len(cluster_names) == 1:
            # Single cluster - no need for thread overhead
            cluster_info = self.describe_cluster(cluster_names[0], region)
            clusters.append(cluster_info)
            LOGGER.debug(f"Found cluster: {cluster_names[0]} ({cluster_info.cluster_status})")
        else:
            # Multiple clusters - parallelize describe calls
            with ThreadPoolExecutor(max_workers=min(len(cluster_names), 4)) as executor:
                future_to_name = {
                    executor.submit(self.describe_cluster, name, region): name
                    for name in cluster_names
                }
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        cluster_info = future.result()
                        clusters.append(cluster_info)
                        LOGGER.debug(f"Found cluster: {name} ({cluster_info.cluster_status})")
                    except Exception as e:
                        LOGGER.warning(f"Failed to describe cluster {name} in {region}: {e}")
        return clusters

    def get_all_clusters(self, force_refresh: bool = False) -> List[ClusterInfo]:
        """Get all clusters across all configured regions.

        Uses caching to avoid excessive API calls. Scans regions in parallel
        for faster response times.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            List of ClusterInfo objects.
        """
        now = time.time()
        if not force_refresh and self._cache and (now - self._cache_time) < self.cache_ttl_seconds:
            LOGGER.debug("Returning cached cluster data (age: %.1fs)", now - self._cache_time)
            return cast(List[ClusterInfo], self._cache.get("clusters", []))

        LOGGER.info(f"Fetching clusters from {len(self.regions)} regions in parallel: {self.regions}")
        start_time = time.time()
        all_clusters: List[ClusterInfo] = []

        # Use ThreadPoolExecutor for parallel region scanning
        # Each region scan takes 2-5 seconds, so parallelizing saves significant time
        max_workers = min(len(self.regions), 5)  # Cap at 5 concurrent workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_region = {
                executor.submit(self._scan_region, region): region
                for region in self.regions
            }
            for future in as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    clusters = future.result()
                    all_clusters.extend(clusters)
                except Exception as e:
                    LOGGER.warning(f"Failed to scan region {region}: {e}")

        # Update caches
        self._cache = {"clusters": all_clusters}
        self._cache_time = now

        # Update cluster_name -> region map for fast lookups
        for cluster in all_clusters:
            self._cluster_region_map[cluster.cluster_name] = cluster.region
        self._cluster_region_map_time = now

        elapsed = time.time() - start_time
        LOGGER.info(f"Found {len(all_clusters)} clusters across {len(self.regions)} regions in {elapsed:.2f}s")
        return all_clusters

    def get_clusters_by_region(self, force_refresh: bool = False) -> Dict[str, List[ClusterInfo]]:
        """Get clusters grouped by region.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            Dictionary mapping region to list of clusters.
        """
        all_clusters = self.get_all_clusters(force_refresh=force_refresh)
        by_region: Dict[str, List[ClusterInfo]] = {}
        for region in self.regions:
            by_region[region] = []
        for cluster in all_clusters:
            if cluster.region in by_region:
                by_region[cluster.region].append(cluster)
            else:
                by_region[cluster.region] = [cluster]
        return by_region

    def get_cluster_by_name(self, cluster_name: str, force_refresh: bool = False) -> Optional[ClusterInfo]:
        """Get a specific cluster by name.

        Args:
            cluster_name: Name of the cluster to find.
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            ClusterInfo if found, None otherwise.
        """
        all_clusters = self.get_all_clusters(force_refresh=force_refresh)
        for cluster in all_clusters:
            if cluster.cluster_name == cluster_name:
                return cluster
        return None

    def get_bucket_for_cluster(self, cluster_name: str, force_refresh: bool = False) -> Optional[str]:
        """Get the monitor bucket name for a cluster.

        Args:
            cluster_name: Name of the cluster.
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            Bucket name (without s3:// prefix) or None if cluster not found or has no bucket tag.
        """
        cluster = self.get_cluster_by_name(cluster_name, force_refresh=force_refresh)
        if cluster:
            return cluster.get_monitor_bucket_name()
        return None

    def _run_ssh_command(
        self,
        host: str,
        command: str,
        region: str,
        timeout: int = 30,
        ssh_key_pattern: str = "~/.ssh/lsmc-omics-{region}.pem",
        user: str = "ubuntu",
    ) -> tuple[bool, str, str]:
        """Run a command on a remote host via SSH.

        Args:
            host: IP address or hostname.
            command: Command to run on remote host.
            region: AWS region (used for SSH key path).
            timeout: Command timeout in seconds.
            ssh_key_pattern: Pattern for SSH key path with {region} placeholder.
            user: SSH user.

        Returns:
            Tuple of (success, output_or_error, full_ssh_command_string).
        """
        import os

        ssh_key = os.path.expanduser(ssh_key_pattern.format(region=region))
        # Wrap in bash -lc to ensure login shell environment (PATH, etc.)
        remote_cmd = f"bash -lc {shlex.quote(command)}"
        ssh_cmd = [
            "ssh",
            "-i", ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            f"{user}@{host}",
            remote_cmd,
        ]
        ssh_cmd_str = " ".join(shlex.quote(part) for part in ssh_cmd)
        LOGGER.debug(f"Running SSH command: {ssh_cmd_str}")
        try:
            result = subprocess.run(
                ssh_cmd, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                error = result.stderr.strip() or f"Exit code {result.returncode}"
                LOGGER.warning(f"SSH command failed: {error}")
                return False, error, ssh_cmd_str
            return True, result.stdout, ssh_cmd_str
        except subprocess.TimeoutExpired:
            LOGGER.error(f"SSH command timed out after {timeout}s")
            return False, f"Command timed out after {timeout}s", ssh_cmd_str
        except FileNotFoundError:
            LOGGER.error("SSH not found")
            return False, "SSH not installed", ssh_cmd_str
        except Exception as e:
            LOGGER.error(f"SSH error: {e}")
            return False, str(e), ssh_cmd_str

    def _parse_budget_output(self, output: str) -> BudgetInfo:
        """Parse budget information from bash shell initialization output.

        Expected format:
        Day CLI initialized for project 'da-us-west-2d-lfg-usw2d' in region 'us-west-2'.
        ...
        The reference S3 bucket is set to: $reference_bucket (lsmc-dayoa-omics-analysis-us-west-2)
        ...
        AWS Budget for project 'da-us-west-2d-lfg-usw2d' in region 'us-west-2':
          Total: 200.0 USD
          Used: 670.906 USD
          Percent Used: 335.453%
        """
        budget = BudgetInfo(fetched_at=datetime.now(timezone.utc).isoformat())

        # Extract project and region from Day CLI init line
        cli_init_match = re.search(
            r"Day CLI initialized for project '([^']+)' in region '([^']+)'", output
        )
        if cli_init_match:
            budget.project_name = cli_init_match.group(1)
            budget.region = cli_init_match.group(2)

        # Extract reference bucket - format: "The reference S3 bucket is set to: $reference_bucket (actual-bucket-name)"
        ref_bucket_match = re.search(
            r"The reference S3 bucket is set to:[^(]*\(([^)]+)\)", output
        )
        if ref_bucket_match:
            budget.reference_bucket = ref_bucket_match.group(1)

        # Fallback: Extract project name from older format
        if not budget.project_name:
            project_match = re.search(r"Project:\s*(\S+)", output)
            if project_match:
                budget.project_name = project_match.group(1)

        # Extract budget details (fallback for project/region if not found above)
        budget_header = re.search(
            r"AWS Budget for project '([^']+)' in region '([^']+)':", output
        )
        if budget_header:
            budget.project_name = budget.project_name or budget_header.group(1)
            budget.region = budget.region or budget_header.group(2)

        total_match = re.search(r"Total:\s*([\d.]+)\s*USD", output)
        if total_match:
            budget.total_budget = float(total_match.group(1))

        used_match = re.search(r"Used:\s*([\d.]+)\s*USD", output)
        if used_match:
            budget.used_budget = float(used_match.group(1))

        percent_match = re.search(r"Percent Used:\s*([\d.]+)%", output)
        if percent_match:
            budget.percent_used = float(percent_match.group(1))

        return budget

    def _parse_squeue_output(self, output: str) -> JobQueueSummary:
        """Parse squeue output into job queue summary.

        The output contains bash initialization messages before the actual squeue data.
        We skip everything until after "The reference S3 bucket is set to:" line,
        then look for the JOBID header line.

        Expected squeue format:
        JOBID  PARTITION  CPUS  ST  NODELIST  MIN_CPUS  STATE  MIN_MEMORY  TIME  NODES  NAME
        554  i8  1  CF  i8-dy-r7gb64-1  1  CONFIGURING  0  0:02  1  sleeptest.sh
        """
        summary = JobQueueSummary(fetched_at=datetime.now(timezone.utc).isoformat())
        lines = output.strip().split("\n")

        if len(lines) < 1:
            return summary

        # Find the start of actual squeue output
        # Skip everything until after "The reference S3 bucket is set to:" line
        start_idx = 0
        for i, line in enumerate(lines):
            if "The reference S3 bucket is set to:" in line:
                start_idx = i + 1
                break

        # Skip any blank lines after the reference bucket line
        while start_idx < len(lines) and not lines[start_idx].strip():
            start_idx += 1

        # Now find the JOBID header line
        header_idx = None
        for i in range(start_idx, len(lines)):
            if lines[i].strip().startswith("JOBID"):
                header_idx = i
                break

        if header_idx is None:
            # No header found, try parsing from start_idx assuming first line is header
            if start_idx < len(lines) and lines[start_idx].strip():
                header_idx = start_idx
            else:
                return summary

        # Parse job lines after header
        for line in lines[header_idx + 1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 11:
                continue

            try:
                job = JobInfo(
                    job_id=parts[0],
                    partition=parts[1],
                    cpus=int(parts[2]) if parts[2].isdigit() else 0,
                    state_short=parts[3],
                    nodelist=parts[4],
                    min_memory=parts[7],
                    time_used=parts[8],
                    nodes=int(parts[9]) if parts[9].isdigit() else 0,
                    name=parts[10] if len(parts) > 10 else "",
                    state=parts[6],
                )
                summary.jobs.append(job)
                summary.total_jobs += 1
                summary.total_cpus += job.cpus

                # Categorize by state
                state_upper = job.state.upper()
                if state_upper == "RUNNING" or job.state_short == "R":
                    summary.running_jobs += 1
                elif state_upper == "PENDING" or job.state_short == "PD":
                    summary.pending_jobs += 1
                elif state_upper == "CONFIGURING" or job.state_short == "CF":
                    summary.configuring_jobs += 1
                else:
                    summary.other_jobs += 1
            except (ValueError, IndexError) as e:
                LOGGER.warning(f"Failed to parse squeue line: {line} - {e}")
                continue

        return summary

    def fetch_headnode_status(
        self,
        cluster: ClusterInfo,
        ssh_key_pattern: str = "~/.ssh/lsmc-omics-{region}.pem",
    ) -> ClusterInfo:
        """Fetch budget and job queue info from a cluster headnode via SSH.

        Args:
            cluster: ClusterInfo object with headnode IP.
            ssh_key_pattern: Pattern for SSH key path.

        Returns:
            Updated ClusterInfo with budget_info and job_queue populated.
        """
        if not cluster.head_node or not cluster.head_node.public_ip:
            LOGGER.debug(f"No public IP for cluster {cluster.cluster_name}, skipping SSH")
            return cluster

        if cluster.head_node.state != "running":
            LOGGER.debug(f"Headnode not running for {cluster.cluster_name}, skipping SSH")
            return cluster

        host = cluster.head_node.public_ip
        region = cluster.region

        # Fetch budget info by running bash (which triggers .bashrc initialization)
        success, output, budget_ssh_cmd = self._run_ssh_command(
            host=host,
            command="bash -i -c 'exit'",
            region=region,
            timeout=45,
            ssh_key_pattern=ssh_key_pattern,
        )
        if success:
            cluster.budget_info = self._parse_budget_output(output)
            cluster.budget_info.ssh_command = budget_ssh_cmd
        else:
            cluster.budget_info = BudgetInfo(
                error=output,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                ssh_command=budget_ssh_cmd,
            )

        # Fetch job queue (source bashrc for slurm PATH)
        squeue_cmd = "source ~/.bashrc && squeue -o '%i  %P  %C  %t  %N  %c  %T  %m  %M  %D  %j'"
        success, output, squeue_ssh_cmd = self._run_ssh_command(
            host=host,
            command=squeue_cmd,
            region=region,
            timeout=30,
            ssh_key_pattern=ssh_key_pattern,
        )
        if success:
            cluster.job_queue = self._parse_squeue_output(output)
            cluster.job_queue.ssh_command = squeue_ssh_cmd
        else:
            cluster.job_queue = JobQueueSummary(
                error=output,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                ssh_command=squeue_ssh_cmd,
            )

        return cluster

    def get_all_clusters_with_status(
        self,
        force_refresh: bool = False,
        fetch_ssh_status: bool = True,
        ssh_key_pattern: str = "~/.ssh/lsmc-omics-{region}.pem",
    ) -> List[ClusterInfo]:
        """Get all clusters with optional SSH-based status info.

        Args:
            force_refresh: If True, bypass cache.
            fetch_ssh_status: If True, fetch budget/job info via SSH.
            ssh_key_pattern: Pattern for SSH key path.

        Returns:
            List of ClusterInfo objects with status populated.
        """
        clusters = self.get_all_clusters(force_refresh=force_refresh)

        if fetch_ssh_status:
            for cluster in clusters:
                try:
                    self.fetch_headnode_status(cluster, ssh_key_pattern)
                except Exception as e:
                    LOGGER.error(f"Failed to fetch SSH status for {cluster.cluster_name}: {e}")

        return clusters

    def clear_cache(self) -> None:
        """Clear the cluster cache."""
        self._cache = {}
        self._cache_time = 0
        LOGGER.debug("Cluster cache cleared")


def get_cluster_service(
    regions: Optional[List[str]] = None,
    aws_profile: Optional[str] = None,
    cache_ttl_seconds: int = 300,
) -> ClusterService:
    """Get the global ClusterService singleton.

    Creates the singleton on first call. Subsequent calls return the same
    instance (ignoring parameters). This ensures cache is shared across
    all API calls.

    Args:
        regions: List of AWS regions (only used on first call).
        aws_profile: AWS profile name (only used on first call).
        cache_ttl_seconds: Cache TTL in seconds (only used on first call).

    Returns:
        The global ClusterService instance.
    """
    global _global_service

    with _global_service_lock:
        if _global_service is None:
            if not regions:
                # Try to get regions from config
                try:
                    from daylib.ursa_config import get_ursa_config
                    ursa_config = get_ursa_config()
                    if ursa_config.is_configured:
                        regions = ursa_config.get_allowed_regions()
                        aws_profile = aws_profile or ursa_config.aws_profile
                except Exception as e:
                    LOGGER.warning(f"Failed to load ursa config: {e}")

            if not regions:
                # Fallback to environment
                import os
                region_str = os.environ.get("URSA_ALLOWED_REGIONS", "")
                regions = [r.strip() for r in region_str.split(",") if r.strip()]
                if not regions:
                    # Final fallback - use us-west-2
                    # Note: AWS_DEFAULT_REGION is intentionally not used.
                    # In a multi-region architecture, regions must be explicit.
                    regions = ["us-west-2"]

            LOGGER.info(f"Creating global ClusterService for regions: {regions}")
            _global_service = ClusterService(
                regions=regions,
                aws_profile=aws_profile,
                cache_ttl_seconds=cache_ttl_seconds,
            )

        return _global_service


def reset_cluster_service() -> None:
    """Reset the global ClusterService singleton.

    Used for testing or when configuration changes require a new instance.
    """
    global _global_service
    with _global_service_lock:
        _global_service = None
        LOGGER.info("Global ClusterService reset")
