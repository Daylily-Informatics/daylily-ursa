"""Ursa configuration loader for ~/.ursa/ursa-config.yaml.

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
from typing import Dict, List, Optional, Tuple

import yaml

LOGGER = logging.getLogger(__name__)

# Canonical config path
DEFAULT_CONFIG_PATH = Path.home() / ".ursa" / "ursa-config.yaml"

# Legacy paths to check (for backward compatibility during migration)
LEGACY_CONFIG_PATHS = [
    Path.home() / ".ursa" / "ursa.yaml",
    Path.home() / ".ursa" / "config.yaml",
]

# Expected schema fields
VALID_FIELDS = {
    "regions": (list, "List of AWS regions to scan"),
    "aws_profile": (str, "AWS profile name"),
    "cognito_region": (str, "AWS region for Cognito"),
    "cognito_user_pool_id": (str, "Cognito User Pool ID"),
    "cognito_app_client_id": (str, "Cognito App Client ID"),
}


def validate_config_file(path: Path) -> Tuple[bool, List[str], List[str]]:
    """Validate a config file for correct YAML format and schema.

    Args:
        path: Path to the config file.

    Returns:
        Tuple of (is_valid, errors, warnings).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not path.exists():
        errors.append(f"Config file not found: {path}")
        return False, errors, warnings

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML syntax: {e}")
        return False, errors, warnings

    if data is None:
        errors.append("Config file is empty")
        return False, errors, warnings

    if not isinstance(data, dict):
        errors.append(f"Config must be a YAML mapping, got {type(data).__name__}")
        return False, errors, warnings

    # Check for unknown fields
    known_fields = set(VALID_FIELDS.keys())
    for key in data.keys():
        if key not in known_fields:
            warnings.append(f"Unknown field '{key}' (will be ignored)")

    # Validate regions field
    if "regions" in data:
        regions = data["regions"]
        if isinstance(regions, list):
            for i, r in enumerate(regions):
                if not isinstance(r, str):
                    errors.append(f"regions[{i}] must be a string, got {type(r).__name__}")
        elif isinstance(regions, dict):
            warnings.append("Legacy region-to-bucket format detected; consider updating to list format")
        else:
            errors.append(f"'regions' must be a list, got {type(regions).__name__}")

    # Validate string fields
    for field_name in ["aws_profile", "cognito_region", "cognito_user_pool_id", "cognito_app_client_id"]:
        if field_name in data and data[field_name] is not None:
            if not isinstance(data[field_name], str):
                errors.append(f"'{field_name}' must be a string, got {type(data[field_name]).__name__}")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


@dataclass
class UrsaConfig:
    """Ursa configuration loaded from ~/.ursa/ursa-config.yaml.

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

    cognito_region: Optional[str] = None
    """AWS region where Cognito User Pool is deployed (overridden by COGNITO_REGION env var)."""

    _config_path: Optional[Path] = None
    """Path where config was loaded from."""

    _from_legacy_path: bool = False
    """Whether config was loaded from a legacy path."""

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "UrsaConfig":
        """Load configuration from YAML file.

        Environment variables take precedence over config file values:
        - AWS_PROFILE overrides aws_profile
        - COGNITO_USER_POOL_ID overrides cognito_user_pool_id
        - COGNITO_APP_CLIENT_ID overrides cognito_app_client_id
        - COGNITO_REGION overrides cognito_region

        Args:
            config_path: Path to config file. If not provided, looks for
                         ~/.ursa/ursa-config.yaml first, then legacy paths.

        Returns:
            UrsaConfig instance (empty regions list if file doesn't exist).
        """
        from_legacy = False

        # Find config file
        if config_path:
            path = config_path
        else:
            # Check canonical path first
            if DEFAULT_CONFIG_PATH.exists():
                path = DEFAULT_CONFIG_PATH
            else:
                # Check legacy paths
                path = None
                for candidate in LEGACY_CONFIG_PATHS:
                    if candidate.exists():
                        path = candidate
                        from_legacy = True
                        LOGGER.warning(
                            "Using legacy config path %s. "
                            "Consider moving to %s",
                            path, DEFAULT_CONFIG_PATH
                        )
                        break

        if not path or not path.exists():
            LOGGER.warning("Ursa config not found at %s", DEFAULT_CONFIG_PATH)
            return cls(_config_path=DEFAULT_CONFIG_PATH)

        # Validate the config file
        is_valid, errors, warnings = validate_config_file(path)
        for warn in warnings:
            LOGGER.warning("%s: %s", path, warn)

        if not is_valid:
            for err in errors:
                LOGGER.error("%s: %s", path, err)
            return cls(_config_path=path)

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
        cognito_region = os.environ.get("COGNITO_REGION") or data.get("cognito_region")

        config = cls(
            regions=regions,
            aws_profile=aws_profile,
            cognito_user_pool_id=cognito_user_pool_id,
            cognito_app_client_id=cognito_app_client_id,
            cognito_region=cognito_region,
            _config_path=path,
            _from_legacy_path=from_legacy,
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

    @property
    def from_legacy_path(self) -> bool:
        """Check if config was loaded from a legacy path."""
        return self._from_legacy_path

    def get_effective_aws_profile(self) -> Optional[str]:
        """Get the effective AWS profile (env var or config).

        Returns:
            AWS profile name, or None if not configured.
        """
        return os.environ.get("AWS_PROFILE") or self.aws_profile

    def get_effective_cognito_region(self) -> Optional[str]:
        """Get the effective Cognito region (env var or config).

        Returns:
            Cognito region, or None if not configured.
        """
        return os.environ.get("COGNITO_REGION") or self.cognito_region

    def get_value_source(self, field: str) -> str:
        """Get the source of a configuration value.

        Args:
            field: Field name (aws_profile, cognito_region, etc.)

        Returns:
            Source description: 'env', 'config', or 'not set'
        """
        env_map = {
            "aws_profile": "AWS_PROFILE",
            "cognito_region": "COGNITO_REGION",
            "cognito_user_pool_id": "COGNITO_USER_POOL_ID",
            "cognito_app_client_id": "COGNITO_APP_CLIENT_ID",
        }

        env_var = env_map.get(field)
        if env_var and os.environ.get(env_var):
            return "env"

        config_val = getattr(self, field, None)
        if config_val:
            return "config"

        return "not set"


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

