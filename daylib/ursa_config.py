"""Ursa configuration loader for ~/.config/ursa/ursa-config.yaml.

This module provides:
- List of AWS regions to scan for ParallelCluster instances
- Per-region SSH key configuration for multi-region cluster access
- AWS profile and Cognito settings (overridden by environment variables)

S3 buckets are discovered from cluster tags (aws-parallelcluster-monitor-bucket)
rather than being configured statically per region.

Configuration follows XDG Base Directory conventions:
- Config file: ~/.config/ursa/ursa-config.yaml
- Legacy paths (~/.ursa/*) are checked for backward compatibility
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml  # type: ignore[import-untyped]

LOGGER = logging.getLogger(__name__)


@dataclass
class RegionConfig:
    """Configuration for a single AWS region.

    Attributes:
        name: AWS region name (e.g., 'us-west-2', 'eu-central-1')
        ssh_pem: Path to SSH private key for this region's clusters.
                 If None, falls back to global ssh_identity_file in monitor config.
    """

    name: str
    ssh_pem: Optional[str] = None

    def get_expanded_ssh_pem(self) -> Optional[str]:
        """Get the SSH key path with ~ expanded."""
        if not self.ssh_pem:
            return None
        return str(Path(self.ssh_pem).expanduser())

# Canonical config path (XDG Base Directory convention)
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "ursa" / "ursa-config.yaml"

# Legacy paths to check (for backward compatibility during migration)
LEGACY_CONFIG_PATHS = [
    Path.home() / ".ursa" / "ursa-config.yaml",  # Previous canonical path
    Path.home() / ".ursa" / "ursa.yaml",
    Path.home() / ".ursa" / "config.yaml",
]

# Expected schema fields
VALID_FIELDS = {
    "regions": (list, "List of AWS regions to scan"),
    "aws_profile": (str, "AWS profile name"),
    "dynamo_db_region": (str, "AWS region for DynamoDB tables (single source of truth)"),
    "cognito_region": (str, "AWS region for Cognito"),
    "cognito_user_pool_id": (str, "Cognito User Pool ID"),
    "cognito_app_client_id": (str, "Cognito App Client ID"),
    "whitelist_domains": (str, "Comma-separated list of allowed email domains"),
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

    # Validate regions field - accepts multiple formats:
    # 1. Simple list of strings: ["us-west-2", "eu-central-1"]
    # 2. List of dicts with region config: [{"us-west-2": {"ssh_pem": "~/.ssh/key.pem"}}]
    # 3. Legacy dict format: {"us-west-2": "bucket-name"}
    if "regions" in data:
        regions = data["regions"]
        if isinstance(regions, list):
            for i, r in enumerate(regions):
                if isinstance(r, str):
                    pass  # Valid: simple region name
                elif isinstance(r, dict):
                    # Valid: region with config like {"us-west-2": {"ssh_pem": "..."}}
                    for region_name, region_opts in r.items():
                        if not isinstance(region_name, str):
                            errors.append(f"regions[{i}] key must be a string, got {type(region_name).__name__}")
                        if region_opts is not None and not isinstance(region_opts, dict):
                            errors.append(f"regions[{i}]['{region_name}'] must be a dict or null, got {type(region_opts).__name__}")
                else:
                    errors.append(f"regions[{i}] must be a string or dict, got {type(r).__name__}")
        elif isinstance(regions, dict):
            warnings.append("Legacy region-to-bucket format detected; consider updating to list format")
        else:
            errors.append(f"'regions' must be a list, got {type(regions).__name__}")

    # Validate string fields
    for field_name in ["aws_profile", "dynamo_db_region", "cognito_region", "cognito_user_pool_id", "cognito_app_client_id"]:
        if field_name in data and data[field_name] is not None:
            if not isinstance(data[field_name], str):
                errors.append(f"'{field_name}' must be a string, got {type(data[field_name]).__name__}")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


@dataclass
class UrsaConfig:
    """Ursa configuration loaded from ~/.config/ursa/ursa-config.yaml.

    S3 buckets are NOT configured here - they are discovered dynamically from
    cluster tags (aws-parallelcluster-monitor-bucket) when a cluster is selected.

    Configuration follows XDG Base Directory conventions. Legacy paths under
    ~/.ursa/ are checked for backward compatibility.
    """

    regions: List[RegionConfig] = field(default_factory=list)
    """List of region configurations to scan for ParallelCluster instances."""

    aws_profile: Optional[str] = None
    """AWS profile to use (overridden by AWS_PROFILE env var)."""

    dynamo_db_region: Optional[str] = None
    """AWS region for DynamoDB tables - single source of truth for all worksets.

    In a multi-region architecture, worksets may have S3 data and compute clusters
    in different regions, but all workset state is stored in a single DynamoDB table
    in this region. This ensures the API server and monitor see the same worksets.

    Overridden by DYNAMO_DB_REGION env var. Defaults to 'us-west-2' if not set.
    """

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

    _region_map: Dict[str, RegionConfig] = field(default_factory=dict, repr=False)
    """Internal map from region name to RegionConfig for fast lookup."""

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
                         ~/.config/ursa/ursa-config.yaml first, then legacy paths.

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

        # Parse regions - support multiple formats for backward compatibility
        regions_data = data.get("regions", [])
        region_configs: List[RegionConfig] = []
        region_map: Dict[str, RegionConfig] = {}

        if isinstance(regions_data, list):
            for item in regions_data:
                if isinstance(item, str):
                    # Simple string format: "us-west-2"
                    rc = RegionConfig(name=item)
                    region_configs.append(rc)
                    region_map[item] = rc
                elif isinstance(item, dict):
                    # Dict format: {"us-west-2": {"ssh_pem": "~/.ssh/key.pem"}}
                    # or {"us-west-2": null} for region without SSH key
                    for region_name, region_opts in item.items():
                        if isinstance(region_name, str):
                            ssh_pem = None
                            if isinstance(region_opts, dict):
                                ssh_pem = region_opts.get("ssh_pem")
                            rc = RegionConfig(name=region_name, ssh_pem=ssh_pem)
                            region_configs.append(rc)
                            region_map[region_name] = rc
        elif isinstance(regions_data, dict):
            # Legacy format: dict with region -> bucket mappings
            # Extract just the region names, ignore bucket mappings
            for region_name in regions_data.keys():
                rc = RegionConfig(name=region_name)
                region_configs.append(rc)
                region_map[region_name] = rc
            LOGGER.warning(
                "Legacy region-to-bucket config format detected in %s. "
                "Buckets are now discovered from cluster tags. "
                "Consider updating to: regions: [%s]",
                path, ", ".join(region_map.keys())
            )

        # Environment variables take precedence over config file
        aws_profile = os.environ.get("AWS_PROFILE") or data.get("aws_profile")
        dynamo_db_region = os.environ.get("DYNAMO_DB_REGION") or data.get("dynamo_db_region")
        cognito_user_pool_id = os.environ.get("COGNITO_USER_POOL_ID") or data.get("cognito_user_pool_id")
        cognito_app_client_id = os.environ.get("COGNITO_APP_CLIENT_ID") or data.get("cognito_app_client_id")
        cognito_region = os.environ.get("COGNITO_REGION") or data.get("cognito_region")

        config = cls(
            regions=region_configs,
            aws_profile=aws_profile,
            dynamo_db_region=dynamo_db_region,
            cognito_user_pool_id=cognito_user_pool_id,
            cognito_app_client_id=cognito_app_client_id,
            cognito_region=cognito_region,
            _config_path=path,
            _from_legacy_path=from_legacy,
            _region_map=region_map,
        )

        LOGGER.info("Loaded Ursa config from %s with %d regions", path, len(region_configs))
        return config

    def get_allowed_regions(self) -> List[str]:
        """Get list of region names to scan for clusters."""
        return [rc.name for rc in self.regions]

    def get_region_config(self, region: str) -> Optional[RegionConfig]:
        """Get the RegionConfig for a specific region.

        Args:
            region: AWS region name (e.g., 'us-west-2', 'eu-central-1')

        Returns:
            RegionConfig if found, None otherwise.
        """
        return self._region_map.get(region)

    def get_ssh_key_for_region(self, region: str) -> Optional[str]:
        """Get the SSH key path for a specific region.

        Args:
            region: AWS region name (e.g., 'us-west-2', 'eu-central-1')

        Returns:
            Expanded path to SSH private key file, or None if not configured.
        """
        rc = self._region_map.get(region)
        if rc:
            return rc.get_expanded_ssh_pem()
        return None

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

    def get_effective_dynamo_db_region(self) -> str:
        """Get the effective DynamoDB region (env var, config, or default).

        This is the single region where all DynamoDB tables (worksets, customers,
        manifests, etc.) are stored, regardless of which AWS regions worksets
        execute in.

        Priority:
            1. DYNAMO_DB_REGION environment variable
            2. dynamo_db_region from config file
            3. Default: 'us-west-2'

        Returns:
            DynamoDB region string.
        """
        return os.environ.get("DYNAMO_DB_REGION") or self.dynamo_db_region or "us-west-2"

    def get_value_source(self, field: str) -> str:
        """Get the source of a configuration value.

        Args:
            field: Field name (aws_profile, cognito_region, dynamo_db_region, etc.)

        Returns:
            Source description: 'env', 'config', or 'not set'
        """
        env_map = {
            "aws_profile": "AWS_PROFILE",
            "dynamo_db_region": "DYNAMO_DB_REGION",
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

