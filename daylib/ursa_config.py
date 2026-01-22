"""Ursa configuration loader for ~/.ursa/config.yaml.

This module provides region-to-bucket mappings and other Ursa-wide settings.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

LOGGER = logging.getLogger(__name__)

# Config search paths (in priority order)
CONFIG_SEARCH_PATHS = [
    Path.home() / ".ursa" / "config.yaml",
    Path.home() / ".ursa" / "ursa.yaml",
]


@dataclass
class RegionConfig:
    """Configuration for a single AWS region."""

    bucket: str
    """S3 bucket URI for this region (e.g., s3://my-bucket-us-west-2)."""

    enabled: bool = True
    """Whether this region is enabled for cluster discovery."""

    @property
    def bucket_name(self) -> str:
        """Return bucket name without s3:// prefix."""
        b = self.bucket
        if b.startswith("s3://"):
            b = b[5:]
        # Remove trailing path if present
        return b.split("/")[0]


@dataclass
class UrsaConfig:
    """Ursa configuration loaded from ~/.ursa/config.yaml."""

    regions: Dict[str, RegionConfig] = field(default_factory=dict)
    """Region name -> RegionConfig mapping."""

    aws_profile: Optional[str] = None
    """Default AWS profile to use."""

    _config_path: Optional[Path] = None
    """Path where config was loaded from."""

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "UrsaConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file. If not provided, searches
                         CONFIG_SEARCH_PATHS in order.

        Returns:
            UrsaConfig instance (empty if file doesn't exist).
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

        regions = {}
        for region_name, region_data in data.get("regions", {}).items():
            if isinstance(region_data, dict):
                regions[region_name] = RegionConfig(
                    bucket=region_data.get("bucket", ""),
                    enabled=region_data.get("enabled", True),
                )
            elif isinstance(region_data, str):
                # Simple format: region: bucket_uri
                regions[region_name] = RegionConfig(bucket=region_data)

        config = cls(
            regions=regions,
            aws_profile=data.get("aws_profile"),
            _config_path=path,
        )

        LOGGER.info("Loaded Ursa config from %s with %d regions", path, len(regions))
        return config

    def get_allowed_regions(self) -> List[str]:
        """Get list of enabled region names."""
        return [r for r, cfg in self.regions.items() if cfg.enabled]

    def get_bucket_for_region(self, region: str) -> Optional[str]:
        """Get the bucket URI for a region.

        Args:
            region: AWS region name (e.g., 'us-west-2').

        Returns:
            Bucket URI (e.g., 's3://my-bucket') or None if region not configured.
        """
        cfg = self.regions.get(region)
        return cfg.bucket if cfg else None

    def get_bucket_name_for_region(self, region: str) -> Optional[str]:
        """Get the bucket name (without s3://) for a region.

        Args:
            region: AWS region name.

        Returns:
            Bucket name or None if region not configured.
        """
        cfg = self.regions.get(region)
        return cfg.bucket_name if cfg else None

    def get_region_for_bucket(self, bucket: str) -> Optional[str]:
        """Look up region for a bucket name.

        Args:
            bucket: Bucket name (with or without s3:// prefix).

        Returns:
            Region name or None if bucket not found in config.
        """
        # Normalize bucket name
        if bucket.startswith("s3://"):
            bucket = bucket[5:]
        bucket = bucket.split("/")[0]

        for region, cfg in self.regions.items():
            if cfg.bucket_name == bucket:
                return region
        return None

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

