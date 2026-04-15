"""Ursa TapDB JSON template-pack helpers."""

from __future__ import annotations

from pathlib import Path

from daylily_tapdb import (
    find_tapdb_core_config_dir,
    resolve_seed_config_dirs,
    seed_templates,
    validate_template_configs,
)
from daylib_ursa.integrations.tapdb_runtime import (
    DEFAULT_TAPDB_DOMAIN_CODE,
    DEFAULT_TAPDB_DOMAIN_REGISTRY_PATH,
    DEFAULT_TAPDB_OWNER_REPO,
    DEFAULT_TAPDB_PREFIX_REGISTRY_PATH,
)


def template_config_root() -> Path:
    """Return the repo-local Ursa TapDB template pack directory."""
    return Path(__file__).resolve().parents[1] / "config" / "tapdb_templates"


def seed_ursa_templates(session) -> None:
    """Load the canonical Ursa JSON template pack through TapDB."""
    config_dirs = resolve_seed_config_dirs(template_config_root())
    templates, issues = validate_template_configs(config_dirs, strict=True)
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        joined = "; ".join(issue.message for issue in errors)
        raise RuntimeError(f"Ursa template pack validation failed: {joined}")
    seed_templates(
        session,
        templates,
        overwrite=True,
        core_config_dir=find_tapdb_core_config_dir(),
        domain_code=DEFAULT_TAPDB_DOMAIN_CODE,
        owner_repo_name=DEFAULT_TAPDB_OWNER_REPO,
        domain_registry_path=DEFAULT_TAPDB_DOMAIN_REGISTRY_PATH,
        prefix_registry_path=DEFAULT_TAPDB_PREFIX_REGISTRY_PATH,
    )
