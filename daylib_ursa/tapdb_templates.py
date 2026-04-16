"""Ursa TapDB JSON template-pack helpers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from daylily_tapdb.euid import (
    AUDIT_LOG_PREFIX,
    GENERIC_INSTANCE_LINEAGE_PREFIX,
    GENERIC_TEMPLATE_PREFIX,
    SYSTEM_MESSAGE_PREFIX,
    SYSTEM_USER_PREFIX,
)
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

_TAPDB_CORE_PREFIXES = {
    GENERIC_TEMPLATE_PREFIX,
    GENERIC_INSTANCE_LINEAGE_PREFIX,
    AUDIT_LOG_PREFIX,
    SYSTEM_USER_PREFIX,
    SYSTEM_MESSAGE_PREFIX,
}


def template_config_root() -> Path:
    """Return the repo-local Ursa TapDB template pack directory."""
    return Path(__file__).resolve().parents[1] / "config" / "tapdb_templates"


def _load_json_object(path: Path, *, required_key: str) -> dict[str, Any]:
    if not path.exists():
        if required_key == "ownership":
            return {"version": "0.4.0", "ownership": {}}
        raise RuntimeError(f"Required TapDB registry file is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive parsing guard
        raise RuntimeError(f"Failed to read TapDB registry JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"TapDB registry JSON must be an object: {path}")
    section = payload.get(required_key)
    if not isinstance(section, dict):
        raise RuntimeError(f"TapDB registry JSON must define object {required_key!r}: {path}")
    return payload


def _write_json_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _claimable_template_prefixes(templates: list[dict[str, Any]]) -> list[str]:
    prefixes = {
        str(template.get("instance_prefix") or "").strip().upper()
        for template in templates
        if str(template.get("instance_prefix") or "").strip()
    }
    return sorted(prefixes.difference(_TAPDB_CORE_PREFIXES))


def _owner_claim_value(claim: Any) -> str:
    if not isinstance(claim, dict):
        return ""
    return str(
        claim.get("issuer_app_code") or claim.get("owner_repo_name") or claim.get("repo_name") or ""
    ).strip()


def _resolve_registry_paths(
    *,
    domain_registry_path: Path | str | None = None,
    prefix_registry_path: Path | str | None = None,
) -> tuple[Path, Path]:
    try:
        from daylib_ursa.config import get_settings

        settings = get_settings()
    except Exception:
        settings = None

    resolved_domain_registry_path = (
        Path(
            domain_registry_path
            or getattr(settings, "tapdb_domain_registry_path", DEFAULT_TAPDB_DOMAIN_REGISTRY_PATH)
        )
        .expanduser()
        .resolve()
    )
    resolved_prefix_registry_path = (
        Path(
            prefix_registry_path
            or getattr(
                settings,
                "tapdb_prefix_ownership_registry_path",
                DEFAULT_TAPDB_PREFIX_REGISTRY_PATH,
            )
        )
        .expanduser()
        .resolve()
    )
    return resolved_domain_registry_path, resolved_prefix_registry_path


def claim_ursa_template_prefixes(
    templates: list[dict[str, Any]],
    *,
    domain_code: str = DEFAULT_TAPDB_DOMAIN_CODE,
    owner_repo_name: str = DEFAULT_TAPDB_OWNER_REPO,
    domain_registry_path: Path = DEFAULT_TAPDB_DOMAIN_REGISTRY_PATH,
    prefix_registry_path: Path = DEFAULT_TAPDB_PREFIX_REGISTRY_PATH,
) -> list[str]:
    """Claim Ursa-owned client template prefixes in the shared TapDB registry."""
    domain_payload = _load_json_object(domain_registry_path, required_key="domains")
    prefix_payload = _load_json_object(prefix_registry_path, required_key="ownership")
    domains = domain_payload["domains"]
    ownership = prefix_payload["ownership"]

    normalized_domain = str(domain_code or "").strip().upper()
    normalized_owner = str(owner_repo_name or "").strip()
    if normalized_domain not in domains:
        raise RuntimeError(
            f"Domain {normalized_domain!r} is not registered in {domain_registry_path}"
        )

    domain_claims = ownership.get(normalized_domain)
    if domain_claims is None:
        domain_claims = {}
        ownership[normalized_domain] = domain_claims
    if not isinstance(domain_claims, dict):
        raise RuntimeError(
            f"Prefix claims for domain {normalized_domain!r} must be an object in {prefix_registry_path}"
        )

    claimed_prefixes: list[str] = []
    updated = False
    for prefix in _claimable_template_prefixes(templates):
        claim = domain_claims.get(prefix)
        existing_owner = _owner_claim_value(claim)
        if existing_owner and existing_owner != normalized_owner:
            raise RuntimeError(
                f"Prefix {prefix!r} for domain {normalized_domain!r} is claimed by "
                f"{existing_owner!r}, not {normalized_owner!r}"
            )
        if (
            not existing_owner
            or not isinstance(claim, dict)
            or claim.get("issuer_app_code") != normalized_owner
        ):
            domain_claims[prefix] = {"issuer_app_code": normalized_owner}
            claimed_prefixes.append(prefix)
            updated = True

    if updated:
        ownership[normalized_domain] = domain_claims
        prefix_payload["ownership"] = ownership
        _write_json_object(prefix_registry_path, prefix_payload)

    return claimed_prefixes


def seed_ursa_templates(
    session,
    *,
    domain_registry_path: Path | str | None = None,
    prefix_registry_path: Path | str | None = None,
) -> None:
    """Load the canonical Ursa JSON template pack through TapDB."""
    resolved_domain_registry_path, resolved_prefix_registry_path = _resolve_registry_paths(
        domain_registry_path=domain_registry_path,
        prefix_registry_path=prefix_registry_path,
    )
    config_dirs = resolve_seed_config_dirs(template_config_root())
    templates, issues = validate_template_configs(config_dirs, strict=True)
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        joined = "; ".join(issue.message for issue in errors)
        raise RuntimeError(f"Ursa template pack validation failed: {joined}")
    claim_ursa_template_prefixes(
        templates,
        domain_code=DEFAULT_TAPDB_DOMAIN_CODE,
        owner_repo_name=DEFAULT_TAPDB_OWNER_REPO,
        domain_registry_path=resolved_domain_registry_path,
        prefix_registry_path=resolved_prefix_registry_path,
    )
    seed_templates(
        session,
        templates,
        overwrite=True,
        core_config_dir=find_tapdb_core_config_dir(),
        domain_code=DEFAULT_TAPDB_DOMAIN_CODE,
        owner_repo_name=DEFAULT_TAPDB_OWNER_REPO,
        domain_registry_path=resolved_domain_registry_path,
        prefix_registry_path=resolved_prefix_registry_path,
    )
