"""Ursa TapDB JSON template-pack helpers."""

from __future__ import annotations

from pathlib import Path

from daylily_tapdb import (
    find_tapdb_core_config_dir,
    resolve_seed_config_dirs,
    seed_templates,
    validate_template_configs,
)
from daylily_tapdb.euid import resolve_client_scoped_core_prefix


def template_config_root() -> Path:
    """Return the repo-local Ursa TapDB template pack directory."""
    return Path(__file__).resolve().parents[1] / "config" / "tapdb_templates"


def seed_ursa_templates(session) -> None:
    """Load the canonical Ursa JSON template pack through TapDB."""
    core_prefix = resolve_client_scoped_core_prefix("R")
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
        core_instance_prefix=core_prefix,
    )
