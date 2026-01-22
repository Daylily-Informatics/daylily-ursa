"""Ursa configuration loader for ~/.ursa/ursa.yaml.

This module provides:
- List of AWS regions to scan for ParallelCluster instances
- AWS profile and Cognito settings (overridden by environment variables)

S3 buckets are discovered from cluster tags (aws-parallelcluster-monitor-bucket)
rather than being configured statically per region.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

LOGGER = logging.getLogger(__name__)

# Config search paths (in priority order)
CONFIG_SEARCH_PATHS = [
    Path.home() / ".ursa" / "ursa.yaml",
    Path.home() / ".ursa" / "config.yaml",
]


@dataclass
class UrsaConfig:
    """Ursa configuration loaded from ~/.ursa/ursa.yaml.

    S3 buckets are NOT configured here - they are discovered dynamically from
    cluster tags (aws-parallelcluster-monitor-bucket) when a cluster is selected.
    """

    regions: List[str] = field(default_factory=list)
    """List of AWS regions to scan for ParallelCluster instances."""

    aws_profile: Optional[str] = None
    """AWS profile to use (overridden by AWS_PROFILE env var)."""

    cognito_user_pool_id: Optional[str] = None
    """Cognito User Pool ID (overridden by COGNITO_USER_POOL_ID env var)."""

    cognito_app_client_id: Optional[str] = None
    """Cognito App Client ID (overridden by COGNITO_APP_CLIENT_ID env var)."""

    _config_path: Optional[Path] = None
    """Path where config was loaded from."""

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "UrsaConfig":
        """Load configuration from YAML file.

        Environment variables take precedence over config file values:
        - AWS_PROFILE overrides aws_profile
        - COGNITO_USER_POOL_ID overrides cognito_user_pool_id
        - COGNITO_APP_CLIENT_ID overrides cognito_app_client_id

        Args:
            config_path: Path to config file. If not provided, searches
                         CONFIG_SEARCH_PATHS in order.

        Returns:
            UrsaConfig instance (empty regions list if file doesn't exist).
        """
        # Find config file
        if config_path:
            path = config_path
        else:
            path = None
            for candidate in CONFIG_SEARCH_PATHS:
                if candidate.exists():
                    path = candidate
                    break

        if not path or not path.exists():
            LOGGER.warning("Ursa config not found (searched: %s)",
                          ", ".join(str(p) for p in CONFIG_SEARCH_PATHS))
            return cls(_config_path=CONFIG_SEARCH_PATHS[0])

        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            LOGGER.error("Failed to load Ursa config from %s: %s", path, e)
            return cls(_config_path=path)

        # Parse regions - support both list format and legacy dict format
        regions_data = data.get("regions", [])
        if isinstance(regions_data, list):
            # New format: simple list of region names
            regions = [r for r in regions_data if isinstance(r, str)]
        elif isinstance(regions_data, dict):
            # Legacy format: dict with region -> bucket mappings
            # Extract just the region names, ignore bucket mappings
            regions = list(regions_data.keys())
            LOGGER.warning(
                "Legacy region-to-bucket config format detected in %s. "
                "Buckets are now discovered from cluster tags. "
                "Consider updating to: regions: [%s]",
                path, ", ".join(regions)
            )
        else:
            regions = []

        # Environment variables take precedence over config file
        aws_profile = os.environ.get("AWS_PROFILE") or data.get("aws_profile")
        cognito_user_pool_id = os.environ.get("COGNITO_USER_POOL_ID") or data.get("cognito_user_pool_id")
        cognito_app_client_id = os.environ.get("COGNITO_APP_CLIENT_ID") or data.get("cognito_app_client_id")

        config = cls(
            regions=regions,
            aws_profile=aws_profile,
            cognito_user_pool_id=cognito_user_pool_id,
            cognito_app_client_id=cognito_app_client_id,
            _config_path=path,
        )

        LOGGER.info("Loaded Ursa config from %s with %d regions", path, len(regions))
        return config

    def get_allowed_regions(self) -> List[str]:
        """Get list of region names to scan for clusters."""
        return self.regions

    @property
    def is_configured(self) -> bool:
        """Check if config has any regions defined."""
        return len(self.regions) > 0

    @property
    def config_path(self) -> Optional[Path]:
        """Get the path where config was loaded from."""
        return self._config_path


# Global singleton instance (lazy-loaded)
_global_config: Optional[UrsaConfig] = None


def get_ursa_config(reload: bool = False) -> UrsaConfig:
    """Get the global UrsaConfig instance.

    Args:
        reload: If True, reload from disk even if already loaded.

    Returns:
        UrsaConfig instance.
    """
    global _global_config
    if _global_config is None or reload:
        _global_config = UrsaConfig.load()
    return _global_config

