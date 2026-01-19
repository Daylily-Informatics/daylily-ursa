"""Cluster service for fetching AWS ParallelCluster information.

Uses pcluster CLI to list and describe clusters across multiple regions.
Implements caching to reduce API calls.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("daylily.cluster_service")


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

    def clear_cache(self) -> None:
        """Clear the cluster cache."""
        self._cache = {}
        self._cache_time = 0
        LOGGER.debug("Cluster cache cleared")

