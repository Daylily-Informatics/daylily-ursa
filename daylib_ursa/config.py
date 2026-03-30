"""Centralized configuration management for Daylily.

Uses Pydantic BaseSettings for environment variable loading with validation.
Configuration is loaded once at startup and injected via dependency.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from daylib_ursa.domain_access import (
    APPROVED_WEB_DOMAIN_SUFFIXES,
    is_allowed_origin,
)

DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8914
DEFAULT_BLOOM_BASE_URL = "https://localhost:8001"
DEFAULT_ATLAS_BASE_URL = "https://localhost:8000"
DEFAULT_URSA_COGNITO_CALLBACK_URL = f"https://localhost:{DEFAULT_API_PORT}/auth/callback"
DEFAULT_URSA_COGNITO_LOGOUT_URL = f"https://localhost:{DEFAULT_API_PORT}/login"


def _yaml_seed_from_ursa_config() -> dict[str, object]:
    """Seed YAML-owned runtime settings from Ursa config before env resolution."""
    try:
        from daylib_ursa.ursa_config import get_ursa_config

        cfg = get_ursa_config()
    except Exception:
        return {}

    seeded = {
        "aws_profile": cfg.aws_profile,
        "ursa_portal_default_customer_id": cfg.ursa_portal_default_customer_id,
        "cognito_group_role_map": cfg.cognito_group_role_map,
        "ursa_internal_output_bucket": cfg.ursa_internal_output_bucket,
        "tapdb_client_id": cfg.tapdb_client_id,
        "tapdb_database_name": cfg.tapdb_database_name,
        "tapdb_env": cfg.tapdb_env,
        "cognito_user_pool_id": cfg.cognito_user_pool_id,
        "cognito_app_client_id": cfg.cognito_app_client_id,
        "cognito_app_client_secret": cfg.cognito_app_client_secret,
        "cognito_domain": cfg.cognito_domain,
        "cognito_region": cfg.cognito_region,
        "cognito_callback_url": cfg.cognito_callback_url,
        "cognito_logout_url": cfg.cognito_logout_url,
        "api_host": cfg.api_host,
        "api_port": cfg.api_port,
        "bloom_base_url": cfg.bloom_base_url,
        "bloom_verify_ssl": cfg.bloom_verify_ssl,
        "atlas_base_url": cfg.atlas_base_url,
        "atlas_verify_ssl": cfg.atlas_verify_ssl,
        "dewey_enabled": cfg.dewey_enabled,
        "dewey_base_url": cfg.dewey_base_url,
        "dewey_api_token": cfg.dewey_api_token,
        "dewey_verify_ssl": cfg.dewey_verify_ssl,
        "deployment_name": cfg.deployment_name,
        "deployment_color": cfg.deployment_color,
        "deployment_is_production": cfg.deployment_is_production,
    }
    return {key: value for key, value in seeded.items() if value is not None}


def _require_https_url(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if not normalized.startswith("https://"):
        raise ValueError(f"{field_name} must use an absolute https:// URL")
    return normalized.rstrip("/")


def _validate_optional_https_url(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return ""
    if not normalized.startswith("https://"):
        raise ValueError(f"{field_name} must use an absolute https:// URL")
    return normalized.rstrip("/")


def normalize_bucket_name(bucket: Optional[str]) -> Optional[str]:
    """Strip s3:// prefix and trailing slashes from bucket name.

    Allows bucket names to be specified with or without the s3:// prefix.

    Args:
        bucket: Bucket name, optionally with s3:// prefix

    Returns:
        Bucket name without prefix, or None if input is None/empty
    """
    if not bucket:
        return None
    bucket = bucket.strip()
    if bucket.startswith("s3://"):
        bucket = bucket[5:]
    # Strip any path component (just get bucket name)
    if "/" in bucket:
        bucket = bucket.split("/")[0]
    return bucket if bucket else None


class Settings(BaseSettings):
    """Daylily application settings.

    All settings can be overridden via environment variables.
    Environment variable names are uppercase versions of the field names.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ========== AWS Configuration ==========
    aws_profile: Optional[str] = Field(
        default=None,
        description="AWS profile name (None uses default credentials chain)",
    )
    aws_account_id: Optional[str] = Field(
        default=None,
        description="AWS account ID for resource ARNs",
    )
    ursa_allowed_regions: str = Field(
        default="us-west-2",
        description="Comma-separated list of AWS regions to scan for ParallelCluster instances",
    )
    ursa_portal_default_customer_id: str = Field(
        default="default-customer",
        description="Fallback customer ID used by the lightweight portal surface",
    )
    ursa_internal_output_bucket: str = Field(
        default="",
        description="Ursa-managed internal S3 bucket for analysis outputs",
    )
    database_backend: str = Field(
        default="tapdb",
        description="Database backend for Ursa runtime",
    )
    database_target: str = Field(
        default="local",
        description="TapDB database target to resolve (local or aurora)",
    )
    tapdb_client_id: str = Field(
        default="local",
        description="TapDB client identifier",
    )
    tapdb_database_name: str = Field(
        default="ursa",
        description="TapDB namespace / database name",
    )
    tapdb_env: str = Field(
        default="dev",
        description="TapDB environment selector",
    )
    ursa_cost_monitor_regions: str = Field(
        default="us-west-2",
        description="Comma-separated regions used for pricing snapshots",
    )
    ursa_cost_monitor_partitions: str = Field(
        default="i192",
        description="Comma-separated production partitions used for pricing snapshots",
    )
    ursa_cost_monitor_interval_hours: int = Field(
        default=24,
        description="Recurring pricing snapshot interval in hours",
    )
    ursa_cost_monitor_enabled: bool = Field(
        default=False,
        description="Enable background scheduled pricing snapshot capture",
    )
    ursa_cost_monitor_config_path: Optional[str] = Field(
        default=None,
        description="Optional cluster YAML path passed to daylily-ec pricing snapshot",
    )

    def get_allowed_regions(self) -> List[str]:
        """Get list of allowed regions from comma-separated string."""
        return [r.strip() for r in self.ursa_allowed_regions.split(",") if r.strip()]

    def get_cost_monitor_regions(self) -> List[str]:
        """Get list of pricing-monitor regions from comma-separated string."""
        return [r.strip() for r in self.ursa_cost_monitor_regions.split(",") if r.strip()]

    def get_cost_monitor_partitions(self) -> List[str]:
        """Get list of pricing-monitor partitions from comma-separated string."""
        return [p.strip() for p in self.ursa_cost_monitor_partitions.split(",") if p.strip()]

    # ========== S3 Configuration ==========
    # NOTE: S3 buckets are discovered from cluster tags (aws-parallelcluster-monitor-bucket).
    # No bucket env vars are needed - each cluster's tag specifies its bucket.
    s3_prefix: str = Field(
        default="worksets/",
        description="Default S3 prefix for workset data",
    )

    # ========== Authentication ==========
    cognito_user_pool_id: Optional[str] = Field(
        default=None,
        description="AWS Cognito User Pool ID",
    )
    cognito_app_client_id: Optional[str] = Field(
        default=None,
        description="AWS Cognito App Client ID",
    )
    cognito_app_client_secret: Optional[str] = Field(
        default=None,
        description="AWS Cognito App Client Secret (optional, required for secret-enabled app clients)",
    )
    cognito_domain: Optional[str] = Field(
        default=None,
        description="AWS Cognito Hosted UI domain (optional, used for SSO/OAuth flows)",
    )
    cognito_region: Optional[str] = Field(
        default=None,
        description="AWS region where the Cognito User Pool is deployed",
    )
    cognito_callback_url: Optional[str] = Field(
        default=None,
        description="Explicit HTTPS callback URL registered for Cognito Hosted UI",
    )
    cognito_logout_url: Optional[str] = Field(
        default=None,
        description="Explicit HTTPS logout redirect URL registered for Cognito Hosted UI",
    )
    cognito_group_role_map: dict[str, str] = Field(
        default_factory=lambda: {
            "platform-admin": "ADMIN",
            "ursa-admin": "ADMIN",
            "ursa-internal": "INTERNAL_USER",
            "ursa-external-admin": "EXTERNAL_USER_ADMIN",
            "ursa-external": "EXTERNAL_USER",
            "ursa-readwrite": "READ_WRITE",
            "ursa-readonly": "READ_ONLY",
        },
        description="Mapping from Cognito group names to Ursa auth roles",
    )
    enable_auth: bool = Field(
        default=True,
        description="Authentication is mandatory and always enabled",
    )
    session_secret_key: str = Field(
        default="daylily-dev-secret-change-in-production",
        description="Secret key for session encryption (CHANGE IN PRODUCTION)",
    )
    whitelist_domains: str = Field(
        default="all",
        description=(
            "Email domain whitelist for registration and login. "
            "Set to 'all' to allow any domain, or a comma-separated list of allowed domains "
            "(e.g., 'company.com,partner.org'). Case-insensitive."
        ),
    )

    # ========== CORS Configuration ==========
    cors_origins: str = Field(
        default=",".join(f"https://{item}" for item in APPROVED_WEB_DOMAIN_SUFFIXES),
        description="Comma-separated list of allowed CORS origins (* for all)",
    )
    daylily_env: str = Field(
        default="development",
        description="Environment: development, staging, production",
    )
    deployment_name: str = Field(
        default="",
        description="Deployment name shown in non-production UI chrome",
    )
    deployment_color: str = Field(
        default="#0f766e",
        description="Deployment banner color shown in non-production UI chrome",
    )
    deployment_is_production: bool = Field(
        default=False,
        description="Whether this deployment should hide non-production chrome",
    )

    # ========== Demo Mode ==========
    demo_mode: bool = Field(
        default=False,
        description="Enable demo mode (allows first-customer fallback). NEVER enable in production.",
    )

    # ========== API Server ==========
    api_host: str = Field(
        default=DEFAULT_API_HOST,
        description="API server host",
    )
    api_port: int = Field(
        default=DEFAULT_API_PORT,
        description="API server port",
    )
    ursa_tapdb_mount_enabled: bool = Field(
        default=True,
        description="Mount TapDB admin UI/API inside Ursa under an admin-only path",
    )
    ursa_tapdb_mount_path: str = Field(
        default="/admin/tapdb",
        description="Ursa path prefix used to mount TapDB admin sub-application",
    )
    ursa_internal_api_key: str = Field(
        default="ursa-dev-internal-key",
        description="Internal API key for Ursa beta write endpoints",
    )
    bloom_base_url: str = Field(
        default=DEFAULT_BLOOM_BASE_URL,
        description="Bloom base URL for run/index resolver requests",
    )
    bloom_api_token: Optional[str] = Field(
        default=None,
        description="Bearer token for Bloom beta API access",
    )
    bloom_verify_ssl: bool = Field(
        default=True,
        description="Verify Bloom HTTPS certificates for resolver requests",
    )
    atlas_base_url: str = Field(
        default=DEFAULT_ATLAS_BASE_URL,
        description="Atlas base URL for result return requests",
    )
    atlas_internal_api_key: Optional[str] = Field(
        default=None,
        description="Atlas internal API key used by Ursa result return",
    )
    atlas_verify_ssl: bool = Field(
        default=True,
        description="Verify Atlas HTTPS certificates for result return requests",
    )
    dewey_enabled: bool = Field(
        default=False,
        description="Enable Ursa <-> Dewey artifact resolve/register integration",
    )
    dewey_base_url: str = Field(
        default="",
        description="Dewey base URL for artifact resolve/register requests",
    )
    dewey_api_token: Optional[str] = Field(
        default=None,
        description="Bearer token used for Dewey API access",
    )
    dewey_verify_ssl: bool = Field(
        default=True,
        description="Verify Dewey HTTPS certificates for artifact requests",
    )
    dewey_timeout_seconds: float = Field(
        default=10.0,
        description="Dewey API timeout in seconds",
    )

    # ========== Notifications ==========
    sns_topic_arn: Optional[str] = Field(
        default=None,
        description="SNS topic ARN for notifications",
    )
    daylily_sns_topic_arn: Optional[str] = Field(
        default=None,
        description="Daylily-specific SNS topic ARN",
    )
    linear_api_key: Optional[str] = Field(
        default=None,
        description="Linear API key for issue tracking integration",
    )
    linear_team_id: Optional[str] = Field(
        default=None,
        description="Linear team ID for issue tracking",
    )

    # ========== Daylily Project Config ==========
    day_project: Optional[str] = Field(default=None, description="Daylily project name")
    day_aws_region: Optional[str] = Field(default=None, description="Daylily AWS region override")
    day_ex_cfg: Optional[str] = Field(default=None, description="Daylily execution config")
    daylily_primary_region: Optional[str] = Field(
        default=None, description="Primary region for multi-region"
    )
    daylily_multi_region: bool = Field(default=False, description="Enable multi-region support")
    day_biome: Optional[str] = Field(default=None, description="Daylily biome setting")
    day_root: Optional[str] = Field(default=None, description="Daylily root directory")
    apptainer_home: Optional[str] = Field(default=None, description="Apptainer home directory")

    # ========== Logging ==========
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR",
    )

    # ========== Rate Limiting ==========
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting",
    )
    rate_limit_auth_per_minute: int = Field(
        default=10,
        description="Rate limit for auth endpoints (requests per minute per IP)",
    )
    rate_limit_read_per_minute: int = Field(
        default=100,
        description="Rate limit for read endpoints (requests per minute per user)",
    )
    rate_limit_write_per_minute: int = Field(
        default=30,
        description="Rate limit for write endpoints (requests per minute per user)",
    )
    rate_limit_admin_per_minute: int = Field(
        default=20,
        description="Rate limit for admin endpoints (requests per minute per user)",
    )
    rate_limit_storage_uri: Optional[str] = Field(
        default=None,
        description="Redis URI for rate limit storage (None = in-memory)",
    )
    rate_limit_whitelist: str = Field(
        default="",
        description="Comma-separated list of whitelisted IPs or user IDs",
    )

    # ========== Validation ==========
    validation_strictness: str = Field(
        default="strict",
        description="Validation strictness level: strict, permissive",
    )
    validation_required: bool = Field(
        default=True,
        description="Require validation for workset creation/updates",
    )

    # ========== Pipeline Monitoring (SSH to headnode) ==========
    pipeline_ssh_user: str = Field(
        default="ubuntu",
        description="SSH user for connecting to headnodes",
    )
    pipeline_ssh_identity_file: Optional[str] = Field(
        default=None,
        description="Path to SSH identity file (PEM) for headnode connection",
    )
    pipeline_ssh_timeout: int = Field(
        default=5,
        description="SSH connection timeout in seconds",
    )
    pipeline_clone_dest_root: str = Field(
        default="/fsx/analysis_results/ubuntu",
        description="Root directory where pipeline worksets are cloned on headnode",
    )
    pipeline_repo_dir_name: str = Field(
        default="daylily-omics-analysis",
        description="Name of the pipeline repository directory",
    )
    pipeline_monitor_config_path: Optional[str] = Field(
        default=None,
        description="Path to workset-monitor-config.yaml for loading additional SSH settings",
    )

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_in_production(cls, v: str, info) -> str:
        """Warn if wildcard CORS is used in production."""
        # Note: We can't access other fields in field_validator easily,
        # so production check is done at runtime in get_cors_origins()
        return v

    @field_validator("daylily_env")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of the allowed values."""
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"daylily_env must be one of: {allowed}")
        return v.lower()

    @field_validator("database_backend")
    @classmethod
    def validate_database_backend(cls, v: str) -> str:
        allowed = {"tapdb", "postgres", "postgresql"}
        normalized = str(v or "").strip().lower()
        if normalized not in allowed:
            raise ValueError(f"database_backend must be one of: {allowed}")
        return normalized

    @field_validator("database_target")
    @classmethod
    def validate_database_target(cls, v: str) -> str:
        allowed = {"local", "aurora"}
        normalized = str(v or "").strip().lower()
        if normalized not in allowed:
            raise ValueError(f"database_target must be one of: {allowed}")
        return normalized

    @field_validator("enable_auth", mode="before")
    @classmethod
    def enforce_auth_always_enabled(cls, _v) -> bool:
        """Authentication is always enabled for Ursa."""
        return True

    @field_validator("ursa_tapdb_mount_path")
    @classmethod
    def validate_tapdb_mount_path(cls, v: str) -> str:
        """Validate mount path shape for mounted TapDB admin."""
        path = str(v or "").strip()
        if not path:
            raise ValueError("ursa_tapdb_mount_path must not be empty")
        if not path.startswith("/"):
            raise ValueError("ursa_tapdb_mount_path must start with '/'")
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        return path

    @field_validator("bloom_base_url", "atlas_base_url")
    @classmethod
    def validate_https_service_urls(cls, v: str, info) -> str:
        return _require_https_url(v, field_name=str(info.field_name))

    @field_validator("dewey_base_url")
    @classmethod
    def validate_dewey_base_url(cls, v: str) -> str:
        return _validate_optional_https_url(v, field_name="dewey_base_url")

    @field_validator("cognito_callback_url", "cognito_logout_url")
    @classmethod
    def validate_optional_cognito_urls(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return None
        return _validate_optional_https_url(v, field_name=str(info.field_name))

    @field_validator("cognito_group_role_map", mode="before")
    @classmethod
    def validate_cognito_group_role_map(cls, value: object) -> dict[str, str]:
        from daylib_ursa.auth.rbac import Role

        if value is None:
            return {
                "platform-admin": "ADMIN",
                "ursa-admin": "ADMIN",
                "ursa-internal": "INTERNAL_USER",
                "ursa-external-admin": "EXTERNAL_USER_ADMIN",
                "ursa-external": "EXTERNAL_USER",
                "ursa-readwrite": "READ_WRITE",
                "ursa-readonly": "READ_ONLY",
            }
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise ValueError("cognito_group_role_map must be a mapping")

        allowed_roles = {role.value for role in Role}
        normalized: dict[str, str] = {}
        for group_name, role_name in value.items():
            group = str(group_name or "").strip()
            role = str(role_name or "").strip().upper()
            if not group:
                raise ValueError("cognito_group_role_map contains an empty group name")
            if role not in allowed_roles:
                raise ValueError(f"Unsupported Ursa role in cognito_group_role_map: {role_name!r}")
            normalized[group] = role
        return normalized

    @model_validator(mode="after")
    def validate_dewey_integration(self) -> "Settings":
        if self.dewey_enabled:
            if not str(self.dewey_base_url or "").strip():
                raise ValueError("dewey_base_url is required when dewey_enabled=true")
            if not str(self.dewey_api_token or "").strip():
                raise ValueError("dewey_api_token is required when dewey_enabled=true")
        if not str(self.ursa_internal_output_bucket or "").strip():
            raise ValueError("ursa_internal_output_bucket is required")
        return self

    def get_cors_origins(self) -> List[str]:
        """Get list of CORS origins from comma-separated string.

        Raises ValueError if an origin falls outside the approved allowlist.
        """
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        if not origins:
            origins = [f"https://{item}" for item in APPROVED_WEB_DOMAIN_SUFFIXES]
        for origin in origins:
            if not is_allowed_origin(origin, allow_local=not self.is_production):
                raise ValueError(
                    "CORS_ORIGINS entries must stay within the approved domain allowlist. "
                    f"Invalid origin: {origin}"
                )
        return origins

    def get_effective_region(self) -> str:
        """Get the effective AWS region.

        Priority order:
        1. DAY_AWS_REGION (Daylily-specific override)
        2. AWS_REGION (standard AWS SDK env var)
        3. Fallback to 'us-west-2'

        Note: AWS_DEFAULT_REGION is intentionally not used. In a multi-region
        architecture, regions must be explicitly specified per API call.
        """
        return self.day_aws_region or os.environ.get("AWS_REGION") or "us-west-2"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.daylily_env == "production" or self.deployment_is_production

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.daylily_env == "development"

    def validate_demo_mode(self) -> None:
        """Validate demo mode is not enabled in production.

        Raises:
            ValueError: If demo_mode is True in production environment.
        """
        if self.demo_mode and self.is_production:
            raise ValueError(
                "Demo mode (DEMO_MODE=true) is not allowed in production. "
                "Set DAYLILY_ENV to 'development' or 'staging' to use demo mode, "
                "or disable demo mode by setting DEMO_MODE=false."
            )

    @property
    def auth_configured(self) -> bool:
        """Check if authentication is properly configured."""
        return bool(self.cognito_user_pool_id and self.cognito_app_client_id)

    @property
    def deployment(self) -> dict[str, object]:
        return {
            "name": self.deployment_name,
            "color": self.deployment_color,
            "is_production": self.deployment_is_production,
        }

    def get_rate_limit_whitelist(self) -> List[str]:
        """Get list of whitelisted IPs/user IDs for rate limiting."""
        if not self.rate_limit_whitelist:
            return []
        return [w.strip() for w in self.rate_limit_whitelist.split(",") if w.strip()]

    def is_rate_limit_whitelisted(self, identifier: str) -> bool:
        """Check if an identifier (IP or user ID) is whitelisted."""
        return identifier in self.get_rate_limit_whitelist()

    @field_validator("validation_strictness")
    @classmethod
    def validate_strictness_level(cls, v: str) -> str:
        """Validate strictness level is one of the allowed values."""
        allowed = {"strict", "permissive"}
        if v.lower() not in allowed:
            raise ValueError(f"validation_strictness must be one of: {allowed}")
        return v.lower()

    @property
    def is_validation_strict(self) -> bool:
        """Check if validation is in strict mode."""
        return self.validation_strictness == "strict"

    def get_whitelist_domains(self) -> List[str]:
        """Get list of whitelisted email domains.

        Returns:
            List of domain strings (lowercase)
            Empty list [] = allow all domains
            ["__BLOCK_ALL__"] = block all domains (when empty string)
        """
        # Empty string = block all domains
        if self.whitelist_domains == "":
            return ["__BLOCK_ALL__"]

        # "all" or "*" = allow all domains
        if not self.whitelist_domains or self.whitelist_domains.strip().lower() in ("all", "*"):
            return []

        return [d.strip().lower() for d in self.whitelist_domains.split(",") if d.strip()]

    def is_domain_whitelisted(self, email: str) -> bool:
        """Check if an email address's domain is whitelisted.

        Args:
            email: Email address to check

        Returns:
            True if domain is allowed (whitelisted or no whitelist configured),
            False if domain is blocked.
        """
        whitelist = self.get_whitelist_domains()

        # Block all sentinel (when whitelist_domains="")
        if whitelist == ["__BLOCK_ALL__"]:
            return False

        if not whitelist:
            # No whitelist = all domains allowed
            return True

        # Extract domain from email
        if not email or "@" not in email:
            return False

        domain = email.split("@")[-1].strip().lower()
        if not domain:
            return False

        return domain in whitelist

    def validate_email_domain(self, email: str) -> tuple[bool, str]:
        """Validate email domain against whitelist.

        Args:
            email: Email address to validate

        Returns:
            Tuple of (is_valid, error_message).
            error_message is empty string if valid.
        """
        if not email:
            return False, "Email address is required"

        if "@" not in email:
            return False, "Invalid email address format"

        domain = email.split("@")[-1].strip().lower()
        if not domain:
            return False, "Invalid email address: missing domain"

        if not self.is_domain_whitelisted(email):
            whitelist = self.get_whitelist_domains()
            return False, (
                f"Email domain '{domain}' is not allowed. "
                f"Registration is restricted to: {', '.join(whitelist)}"
            )

        return True, ""


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Settings are loaded once and cached for the lifetime of the application.
    Use this function as a FastAPI dependency.
    """
    return Settings(**_yaml_seed_from_ursa_config())


def clear_settings_cache() -> None:
    """Clear the settings cache.

    Useful for testing when you need to reload settings with different
    environment variables.
    """
    get_settings.cache_clear()


def get_settings_for_testing(**overrides) -> Settings:
    """Create settings instance with overrides for testing.

    This bypasses the cache, allowing tests to use custom configuration.
    """
    payload = _yaml_seed_from_ursa_config()
    payload.update(overrides)
    if payload.get("ursa_portal_default_customer_id") is None:
        payload.pop("ursa_portal_default_customer_id", None)
    return Settings(**payload)
