"""Centralized configuration management for Daylily.

Uses Pydantic BaseSettings for environment variable loading with validation.
Configuration is loaded once at startup and injected via dependency.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    ursa_cost_monitor_regions: str = Field(
        default="us-west-2,us-east-1,eu-central-1",
        description="Comma-separated regions used for pricing snapshots",
    )
    ursa_cost_monitor_partitions: str = Field(
        default="i8,i128,i192,i192mem,i192bigmem",
        description="Comma-separated production partitions used for pricing snapshots",
    )
    ursa_cost_monitor_interval_hours: int = Field(
        default=6,
        description="Recurring pricing snapshot interval in hours",
    )
    ursa_cost_monitor_config_path: Optional[str] = Field(
        default=None,
        description="Optional cluster YAML path passed to daylily-ec pricing snapshot",
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
        default="*",
        description="Comma-separated list of allowed CORS origins (* for all)",
    )
    daylily_env: str = Field(
        default="development",
        description="Environment: development, staging, production",
    )

    # ========== Demo Mode ==========
    demo_mode: bool = Field(
        default=False,
        description="Enable demo mode (allows first-customer fallback). NEVER enable in production.",
    )

    # ========== API Server ==========
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    api_port: int = Field(
        default=8914,
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
        default="http://localhost:8001",
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
        default="http://localhost:8000",
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

    def get_cors_origins(self) -> List[str]:
        """Get list of CORS origins from comma-separated string.

        Raises ValueError if wildcard is used in production.
        """
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        if self.daylily_env == "production" and "*" in origins:
            raise ValueError(
                "Wildcard CORS origin (*) is not allowed in production. "
                "Set CORS_ORIGINS to a comma-separated list of allowed origins."
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
        return self.daylily_env == "production"

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
    return Settings()


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
    return Settings(**overrides)
