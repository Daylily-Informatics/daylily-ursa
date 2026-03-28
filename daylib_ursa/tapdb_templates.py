"""Ursa TapDB JSON template-pack helpers."""

from __future__ import annotations

from pathlib import Path

from daylily_tapdb import resolve_seed_config_dirs, seed_templates, validate_template_configs


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
    seed_templates(session, templates, overwrite=True)

