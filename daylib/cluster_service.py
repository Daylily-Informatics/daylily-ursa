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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("daylily.cluster_service")


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
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
            "budget_info": self.budget_info.to_dict() if self.budget_info else None,
            "job_queue": self.job_queue.to_dict() if self.job_queue else None,
        }


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

    def _run_pcluster_command(self, args: List[str], timeout: int = 30) -> Dict[str, Any]:
        """Run a pcluster CLI command and return parsed JSON output."""
        cmd = ["pcluster"] + args
        if self.aws_profile:
            env_vars = {"AWS_PROFILE": self.aws_profile}
        else:
            env_vars = None
        LOGGER.debug(f"Running: {' '.join(cmd)}")
        try:
            import os
            env = os.environ.copy()
            if env_vars:
                env.update(env_vars)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=env
            )
            if result.returncode != 0:
                LOGGER.warning(f"pcluster command failed: {result.stderr}")
                return {"error": result.stderr.strip() or f"Exit code {result.returncode}"}
            if not result.stdout.strip():
                return {}
            return json.loads(result.stdout)
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
        if "error" in result:
            LOGGER.warning(f"Failed to list clusters in {region}: {result['error']}")
            return []
        clusters = result.get("clusters", [])
        return [c.get("clusterName", "") for c in clusters if c.get("clusterName")]

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

    def get_all_clusters(self, force_refresh: bool = False) -> List[ClusterInfo]:
        """Get all clusters across all configured regions.

        Uses caching to avoid excessive API calls.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            List of ClusterInfo objects.
        """
        now = time.time()
        if not force_refresh and self._cache and (now - self._cache_time) < self.cache_ttl_seconds:
            LOGGER.debug("Returning cached cluster data")
            return self._cache.get("clusters", [])

        LOGGER.info(f"Fetching clusters from regions: {self.regions}")
        all_clusters: List[ClusterInfo] = []

        for region in self.regions:
            LOGGER.debug(f"Scanning region: {region}")
            cluster_names = self.list_clusters_in_region(region)
            if not cluster_names:
                LOGGER.debug(f"No clusters found in {region}")
                continue
            for name in cluster_names:
                cluster_info = self.describe_cluster(name, region)
                all_clusters.append(cluster_info)
                LOGGER.debug(f"Found cluster: {name} ({cluster_info.cluster_status})")

        self._cache = {"clusters": all_clusters}
        self._cache_time = now
        LOGGER.info(f"Found {len(all_clusters)} clusters across {len(self.regions)} regions")
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

