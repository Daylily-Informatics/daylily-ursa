"""Ursa configuration loader for ~/.config/ursa-<deployment>/ursa-config-<deployment>.yaml.

This module provides:
- List of AWS regions to scan for ParallelCluster instances
- Per-region SSH key configuration for multi-region cluster access
- AWS profile plus YAML-owned Cognito and deployment settings

S3 buckets are discovered from cluster tags (aws-parallelcluster-monitor-bucket)
rather than being configured statically per region.

Configuration follows XDG Base Directory conventions:
- Config file: ~/.config/ursa-<deployment>/ursa-config-<deployment>.yaml
"""

import colorsys
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml  # type: ignore[import-untyped]

LOGGER = logging.getLogger(__name__)
DEFAULT_DEPLOYMENT_BANNER_COLOR = "#AFEEEE"
PRODUCTION_DEPLOYMENT_NAMES = {"prod", "production"}


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


def _sanitize_deployment_code(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9-]+", "-", str(value or "").strip()).strip("-")
    return cleaned or "local"


def _resolve_deployment_code() -> str:
    return _sanitize_deployment_code(
        os.environ.get("URSA_DEPLOYMENT_CODE")
        or os.environ.get("DEPLOYMENT_CODE")
        or os.environ.get("LSMC_DEPLOYMENT_CODE")
        or "local"
    )


def get_config_dir() -> Path:
    xdg_config_home = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return xdg_config_home / f"ursa-{_resolve_deployment_code()}"


def get_config_file_path() -> Path:
    deployment = _resolve_deployment_code()
    return get_config_dir() / f"ursa-config-{deployment}.yaml"


def parse_regions_csv(regions_csv: str) -> List[str]:
    """Normalize a comma-separated region list into unique region names."""
    seen: set[str] = set()
    regions: List[str] = []
    for raw_value in str(regions_csv or "").split(","):
        normalized = str(raw_value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        regions.append(normalized)
    if not regions:
        raise ValueError("At least one AWS region is required")
    return regions


def update_config_regions(
    *,
    regions: List[str],
    config_path: Optional[Path] = None,
) -> "UrsaConfig":
    """Persist the configured scan regions while preserving existing region-specific options."""
    normalized_regions = parse_regions_csv(",".join(regions))
    path = config_path or get_config_file_path()
    if not path.exists():
        raise FileNotFoundError(f"Ursa config file not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Ursa config must be a YAML mapping: {path}")

    preserved_entries: Dict[str, object] = {}
    for entry in list(payload.get("regions") or []):
        if isinstance(entry, str):
            region_name = str(entry or "").strip()
            if region_name and region_name not in preserved_entries:
                preserved_entries[region_name] = region_name
            continue
        if isinstance(entry, dict):
            for raw_region_name, region_options in entry.items():
                region_name = str(raw_region_name or "").strip()
                if region_name and region_name not in preserved_entries:
                    preserved_entries[region_name] = {region_name: region_options}
                break

    payload["regions"] = [
        preserved_entries.get(region_name, region_name) for region_name in normalized_regions
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return UrsaConfig.load(path)


def _stable_color_hex(name: str, *, hue_shift: int = 0, lightness: float, saturation: float) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    hue = (int.from_bytes(digest[:8], "big") + hue_shift) % 360
    red, green, blue = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)
    return "#{:02x}{:02x}{:02x}".format(
        round(red * 255),
        round(green * 255),
        round(blue * 255),
    )


def _stable_deployment_color_hex(name: str) -> str:
    return _stable_color_hex(name, lightness=0.46, saturation=0.72)


def _stable_region_color_hex(name: str) -> str:
    return _stable_color_hex(name, hue_shift=180, lightness=0.62, saturation=0.45)


def _resolve_deployment_chrome(
    *,
    name: str | None,
    color: str | None,
    default_name: str | None = None,
) -> dict[str, object]:
    resolved_name = str(name or "").strip() or str(default_name or "").strip()
    _ = color
    resolved_color = (
        _stable_deployment_color_hex(resolved_name)
        if resolved_name
        else DEFAULT_DEPLOYMENT_BANNER_COLOR
    )
    return {
        "name": resolved_name,
        "color": resolved_color,
        "is_production": resolved_name.lower() in PRODUCTION_DEPLOYMENT_NAMES,
    }


# Expected schema fields
VALID_FIELDS = {
    "regions": (list, "List of AWS regions to scan"),
    "aws_profile": (str, "AWS profile name"),
    "cognito_group_role_map": (dict, "Canonical Cognito group-to-role mapping"),
    "ursa_internal_output_bucket": (str, "Ursa-managed internal S3 bucket"),
    "tapdb_client_id": (str, "TapDB client identifier"),
    "tapdb_database_name": (str, "TapDB namespace / database name"),
    "tapdb_env": (str, "TapDB environment selector"),
    "tapdb_config_path": (str, "Explicit TapDB config path"),
    "tapdb_domain_registry_path": (
        str,
        "Explicit TapDB domain registry path",
    ),
    "tapdb_prefix_ownership_registry_path": (
        str,
        "Explicit TapDB prefix ownership registry path",
    ),
    "cognito_region": (str, "AWS region for Cognito"),
    "cognito_user_pool_id": (str, "Cognito User Pool ID"),
    "cognito_app_client_id": (str, "Cognito App Client ID"),
    "cognito_app_client_secret": (str, "Cognito App Client Secret"),
    "cognito_domain": (str, "Cognito Hosted UI domain"),
    "cognito_callback_url": (str, "Cognito Hosted UI callback URL"),
    "cognito_logout_url": (str, "Cognito Hosted UI logout redirect URL"),
    "session_secret_key": (str, "Session secret key for web sessions"),
    "api_host": (str, "API bind host"),
    "api_port": (int, "API bind port"),
    "bloom_base_url": (str, "Bloom base URL"),
    "bloom_verify_ssl": (bool, "Verify Bloom TLS certificates"),
    "atlas_base_url": (str, "Atlas base URL"),
    "atlas_verify_ssl": (bool, "Verify Atlas TLS certificates"),
    "dewey_enabled": (bool, "Enable Dewey integration"),
    "dewey_base_url": (str, "Dewey base URL"),
    "dewey_api_token": (str, "Dewey API bearer token"),
    "dewey_verify_ssl": (bool, "Verify Dewey TLS certificates"),
    "ursa_internal_api_key": (str, "Ursa internal API key"),
    "whitelist_domains": (str, "Allowed email domains for registration/login"),
    "deployment": (dict, "Deployment metadata for non-production UI chrome"),
    "ui_show_environment_chrome": (bool, "Toggle GUI deployment and region chrome"),
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
        if key in known_fields:
            continue
        warnings.append(f"Unknown field '{key}' (will be ignored)")

    # Validate regions field — accepted formats:
    # 1. Simple list of strings: ["us-west-2", "eu-central-1"]
    # 2. List of dicts with region config: [{"us-west-2": {"ssh_pem": "~/.ssh/key.pem"}}]
    # Dict format (e.g. {"us-west-2": "bucket-name"}) is rejected.
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
                            errors.append(
                                f"regions[{i}] key must be a string, got {type(region_name).__name__}"
                            )
                        if region_opts is not None and not isinstance(region_opts, dict):
                            errors.append(
                                f"regions[{i}]['{region_name}'] must be a dict or null, got {type(region_opts).__name__}"
                            )
                else:
                    errors.append(f"regions[{i}] must be a string or dict, got {type(r).__name__}")
        elif isinstance(regions, dict):
            errors.append(
                "'regions' must be a list, not a dict. "
                'Update to list format: regions: ["us-west-2", ...]'
            )
        else:
            errors.append(f"'regions' must be a list, got {type(regions).__name__}")

    # Validate string fields
    for field_name in [
        "aws_profile",
        "ursa_internal_output_bucket",
        "tapdb_client_id",
        "tapdb_database_name",
        "tapdb_env",
        "tapdb_config_path",
        "tapdb_domain_registry_path",
        "tapdb_prefix_ownership_registry_path",
        "cognito_region",
        "cognito_user_pool_id",
        "cognito_app_client_id",
        "cognito_app_client_secret",
        "cognito_domain",
        "cognito_callback_url",
        "cognito_logout_url",
        "session_secret_key",
        "api_host",
        "bloom_base_url",
        "atlas_base_url",
        "dewey_base_url",
        "whitelist_domains",
        "ursa_internal_api_key",
    ]:
        if field_name in data and data[field_name] is not None:
            if not isinstance(data[field_name], str):
                errors.append(
                    f"'{field_name}' must be a string, got {type(data[field_name]).__name__}"
                )

    for field_name in ["api_port"]:
        if field_name in data and data[field_name] is not None:
            if not isinstance(data[field_name], int):
                errors.append(
                    f"'{field_name}' must be an integer, got {type(data[field_name]).__name__}"
                )

    for field_name in ["bloom_verify_ssl", "atlas_verify_ssl", "dewey_enabled", "dewey_verify_ssl"]:
        if field_name in data and data[field_name] is not None:
            if not isinstance(data[field_name], bool):
                errors.append(
                    f"'{field_name}' must be a boolean, got {type(data[field_name]).__name__}"
                )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


@dataclass
class UrsaConfig:
    """Ursa configuration loaded from ~/.config/ursa-<deployment>/ursa-config-<deployment>.yaml.

    S3 buckets are NOT configured here - they are discovered dynamically from
    cluster tags (aws-parallelcluster-monitor-bucket) when a cluster is selected.

    Configuration follows deployment-scoped XDG Base Directory conventions.
    """

    regions: List[RegionConfig] = field(default_factory=list)
    """List of region configurations to scan for ParallelCluster instances."""

    aws_profile: Optional[str] = None
    """AWS profile to use (AWS_PROFILE may still override this)."""

    cognito_group_role_map: Optional[Dict[str, str]] = None
    """Optional Cognito group-to-role mapping override loaded from YAML config."""

    ursa_internal_output_bucket: Optional[str] = None
    """Ursa-managed internal output bucket read from YAML config."""

    tapdb_client_id: Optional[str] = None
    """TapDB client identifier read from YAML config."""

    tapdb_database_name: Optional[str] = None
    """TapDB namespace / database name read from YAML config."""

    tapdb_env: Optional[str] = None
    """TapDB environment selector read from YAML config."""

    tapdb_config_path: Optional[str] = None
    """Explicit TapDB config path read from YAML config."""

    tapdb_domain_registry_path: Optional[str] = None
    """Explicit TapDB domain registry path read from YAML config."""

    tapdb_prefix_ownership_registry_path: Optional[str] = None
    """Explicit TapDB prefix ownership registry path read from YAML config."""

    cognito_user_pool_id: Optional[str] = None
    """Cognito User Pool ID read from YAML config."""

    cognito_app_client_id: Optional[str] = None
    """Cognito App Client ID read from YAML config."""

    cognito_app_client_secret: Optional[str] = None
    """Cognito App Client Secret read from YAML config."""

    cognito_domain: Optional[str] = None
    """Cognito Hosted UI domain read from YAML config."""

    cognito_region: Optional[str] = None
    """AWS region where Cognito User Pool is deployed, read from YAML config."""

    cognito_callback_url: Optional[str] = None
    """Cognito Hosted UI callback URL, read from YAML config."""

    cognito_logout_url: Optional[str] = None
    """Cognito Hosted UI logout redirect URL, read from YAML config."""

    session_secret_key: Optional[str] = None
    """Session secret key for web sessions read from YAML config."""

    api_host: Optional[str] = None
    """API bind host read from YAML config."""

    api_port: Optional[int] = None
    """API bind port read from YAML config."""

    bloom_base_url: Optional[str] = None
    """Bloom base URL read from YAML config."""

    bloom_verify_ssl: Optional[bool] = None
    """Bloom TLS verification flag read from YAML config."""

    atlas_base_url: Optional[str] = None
    """Atlas base URL read from YAML config."""

    atlas_verify_ssl: Optional[bool] = None
    """Atlas TLS verification flag read from YAML config."""

    dewey_enabled: Optional[bool] = None
    """Whether Dewey integration is enabled in YAML config."""

    dewey_base_url: Optional[str] = None
    """Dewey base URL read from YAML config."""

    dewey_api_token: Optional[str] = None
    """Dewey API bearer token read from YAML config."""

    dewey_verify_ssl: Optional[bool] = None
    """Dewey TLS verification flag read from YAML config."""

    ursa_internal_api_key: Optional[str] = None
    """Ursa internal API key read from YAML config."""

    whitelist_domains: Optional[str] = None
    """Allowed registration/login email domains (overridden by WHITELIST_DOMAINS env var)."""

    deployment_name: str = ""
    """Deployment name shown in non-production UI chrome."""

    deployment_color: str = ""
    """Deployment banner color derived from deployment name."""

    deployment_is_production: bool = False
    """Whether this deployment is considered production-like."""

    ui_show_environment_chrome: bool = True
    """Whether deployment and region chrome should render in the GUI."""

    _config_path: Optional[Path] = None
    """Path where config was loaded from."""

    _region_map: Dict[str, RegionConfig] = field(default_factory=dict, repr=False)
    """Internal map from region name to RegionConfig for fast lookup."""

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "UrsaConfig":
        """Load configuration from YAML file.

        AWS_PROFILE and WHITELIST_DOMAINS may override config file values.
        Cognito runtime settings are read from YAML only.

        Args:
            config_path: Path to config file. If not provided, looks for
                         ~/.config/ursa-<deployment>/ursa-config-<deployment>.yaml.

        Returns:
            UrsaConfig instance (empty regions list if file doesn't exist).
        """
        # Find config file
        if config_path:
            path = config_path
        else:
            path = get_config_file_path()

        if not path or not path.exists():
            LOGGER.warning("Ursa config not found at %s", get_config_file_path())
            return cls(_config_path=get_config_file_path())

        # Validate the config file
        is_valid, errors, warnings = validate_config_file(path)
        for warn in warnings:
            LOGGER.warning("%s: %s", path, warn)

        if not is_valid:
            detail = "; ".join(errors)
            raise ValueError(f"{path}: {detail}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            LOGGER.error("Failed to load Ursa config from %s: %s", path, e)
            return cls(_config_path=path)

        # Parse regions — list format only
        regions_data = data.get("regions", [])
        region_configs: List[RegionConfig] = []
        region_map: Dict[str, RegionConfig] = {}

        if isinstance(regions_data, dict):
            raise ValueError(
                f"'regions' in {path} must be a list, not a dict. "
                f"Update to: regions: [{', '.join(regions_data.keys())}]"
            )

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

        deployment = data.get("deployment") or {}
        if not isinstance(deployment, dict):
            deployment = {}

        # Environment variables take precedence only for non-Cognito runtime knobs.
        aws_profile = os.environ.get("AWS_PROFILE") or data.get("aws_profile")
        cognito_group_role_map = data.get("cognito_group_role_map")
        ursa_internal_output_bucket = data.get("ursa_internal_output_bucket")
        tapdb_client_id = data.get("tapdb_client_id")
        tapdb_database_name = data.get("tapdb_database_name")
        tapdb_env = data.get("tapdb_env")
        tapdb_config_path = data.get("tapdb_config_path")
        tapdb_domain_registry_path = data.get("tapdb_domain_registry_path")
        tapdb_prefix_ownership_registry_path = data.get("tapdb_prefix_ownership_registry_path")
        cognito_user_pool_id = data.get("cognito_user_pool_id")
        cognito_app_client_id = data.get("cognito_app_client_id")
        cognito_app_client_secret = data.get("cognito_app_client_secret")
        cognito_domain = data.get("cognito_domain")
        cognito_region = data.get("cognito_region")
        cognito_callback_url = data.get("cognito_callback_url")
        cognito_logout_url = data.get("cognito_logout_url")
        session_secret_key = data.get("session_secret_key")
        api_host = data.get("api_host")
        api_port = data.get("api_port")
        bloom_base_url = data.get("bloom_base_url")
        bloom_verify_ssl = data.get("bloom_verify_ssl")
        atlas_base_url = data.get("atlas_base_url")
        atlas_verify_ssl = data.get("atlas_verify_ssl")
        dewey_enabled = data.get("dewey_enabled")
        dewey_base_url = data.get("dewey_base_url")
        dewey_api_token = data.get("dewey_api_token")
        dewey_verify_ssl = data.get("dewey_verify_ssl")
        ursa_internal_api_key = data.get("ursa_internal_api_key")
        whitelist_domains = os.environ.get("WHITELIST_DOMAINS") or data.get("whitelist_domains")
        ui_show_environment_chrome = data.get("ui_show_environment_chrome")

        deployment_chrome = _resolve_deployment_chrome(
            name=str(deployment.get("name") or ""),
            color=str(deployment.get("color") or ""),
            default_name=_resolve_deployment_code(),
        )

        config = cls(
            regions=region_configs,
            aws_profile=aws_profile,
            cognito_group_role_map=cognito_group_role_map,
            ursa_internal_output_bucket=ursa_internal_output_bucket,
            tapdb_client_id=tapdb_client_id,
            tapdb_database_name=tapdb_database_name,
            tapdb_env=tapdb_env,
            tapdb_config_path=tapdb_config_path,
            tapdb_domain_registry_path=tapdb_domain_registry_path,
            tapdb_prefix_ownership_registry_path=tapdb_prefix_ownership_registry_path,
            cognito_user_pool_id=cognito_user_pool_id,
            cognito_app_client_id=cognito_app_client_id,
            cognito_app_client_secret=cognito_app_client_secret,
            cognito_domain=cognito_domain,
            cognito_region=cognito_region,
            cognito_callback_url=cognito_callback_url,
            cognito_logout_url=cognito_logout_url,
            session_secret_key=session_secret_key,
            api_host=api_host,
            api_port=api_port,
            bloom_base_url=bloom_base_url,
            bloom_verify_ssl=bloom_verify_ssl,
            atlas_base_url=atlas_base_url,
            atlas_verify_ssl=atlas_verify_ssl,
            dewey_enabled=dewey_enabled,
            dewey_base_url=dewey_base_url,
            dewey_api_token=dewey_api_token,
            dewey_verify_ssl=dewey_verify_ssl,
            ursa_internal_api_key=ursa_internal_api_key,
            whitelist_domains=whitelist_domains,
            deployment_name=str(deployment_chrome["name"]),
            deployment_color=str(deployment_chrome["color"]),
            deployment_is_production=bool(deployment_chrome["is_production"]),
            ui_show_environment_chrome=bool(
                True if ui_show_environment_chrome is None else ui_show_environment_chrome
            ),
            _config_path=path,
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

    def get_effective_aws_profile(self) -> Optional[str]:
        """Get the effective AWS profile (env var or config).

        Returns:
            AWS profile name, or None if not configured.
        """
        return os.environ.get("AWS_PROFILE") or self.aws_profile

    def get_effective_cognito_region(self) -> Optional[str]:
        """Get the configured Cognito region.

        Returns:
            Cognito region, or None if not configured.
        """
        return self.cognito_region

    def get_effective_cognito_domain(self) -> Optional[str]:
        """Get the configured Cognito Hosted UI domain."""
        return self.cognito_domain

    def get_value_source(self, field: str) -> str:
        """Get the source of a configuration value.

        Args:
            field: Field name (aws_profile, cognito_region, etc.)

        Returns:
            Source description: 'env', 'config', or 'not set'
        """
        env_map = {
            "aws_profile": "AWS_PROFILE",
            "whitelist_domains": "WHITELIST_DOMAINS",
            "session_secret_key": "SESSION_SECRET_KEY",
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
