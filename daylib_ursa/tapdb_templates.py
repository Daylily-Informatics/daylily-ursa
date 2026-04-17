"""Ursa TapDB JSON template-pack helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from sqlalchemy import text
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
    seed_templates,
    validate_template_configs,
)
from daylib_ursa.integrations.tapdb_runtime import (
    DEFAULT_TAPDB_DOMAIN_CODE,
    DEFAULT_TAPDB_OWNER_REPO,
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

    resolved_domain_registry_path = _require_explicit_registry_path(
        domain_registry_path
        or os.environ.get("TAPDB_DOMAIN_REGISTRY_PATH")
        or getattr(settings, "tapdb_domain_registry_path", "")
        or "",
        field_name="tapdb_domain_registry_path",
    )
    resolved_prefix_registry_path = _require_explicit_registry_path(
        prefix_registry_path
        or os.environ.get("TAPDB_PREFIX_OWNERSHIP_REGISTRY_PATH")
        or getattr(settings, "tapdb_prefix_ownership_registry_path", "")
        or "",
        field_name="tapdb_prefix_ownership_registry_path",
    )
    return resolved_domain_registry_path, resolved_prefix_registry_path


def _require_explicit_registry_path(path_value: Path | str, *, field_name: str) -> Path:
    raw = str(path_value or "").strip()
    if not raw:
        raise RuntimeError(f"{field_name} is required and must be passed as a full path.")
    resolved = Path(raw)
    if not resolved.is_absolute():
        raise RuntimeError(f"{field_name} must be an absolute path, got: {raw}")
    return resolved


def claim_ursa_template_prefixes(
    templates: list[dict[str, Any]],
    *,
    domain_code: str = DEFAULT_TAPDB_DOMAIN_CODE,
    owner_repo_name: str = DEFAULT_TAPDB_OWNER_REPO,
    domain_registry_path: Path | str | None = None,
    prefix_registry_path: Path | str | None = None,
) -> list[str]:
    """Claim Ursa-owned client template prefixes in the shared TapDB registry."""
    resolved_domain_registry_path = _require_explicit_registry_path(
        domain_registry_path or "",
        field_name="tapdb_domain_registry_path",
    )
    resolved_prefix_registry_path = _require_explicit_registry_path(
        prefix_registry_path or "",
        field_name="tapdb_prefix_ownership_registry_path",
    )
    domain_payload = _load_json_object(resolved_domain_registry_path, required_key="domains")
    prefix_payload = _load_json_object(resolved_prefix_registry_path, required_key="ownership")
    domains = domain_payload["domains"]
    ownership = prefix_payload["ownership"]

    normalized_domain = str(domain_code or "").strip().upper()
    normalized_owner = str(owner_repo_name or "").strip()
    if normalized_domain not in domains:
        raise RuntimeError(
            f"Domain {normalized_domain!r} is not registered in {resolved_domain_registry_path}"
        )

    domain_claims = ownership.get(normalized_domain)
    if domain_claims is None:
        domain_claims = {}
        ownership[normalized_domain] = domain_claims
    if not isinstance(domain_claims, dict):
        raise RuntimeError(
            "Prefix claims for domain "
            f"{normalized_domain!r} must be an object in {resolved_prefix_registry_path}"
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
        _write_json_object(resolved_prefix_registry_path, prefix_payload)

    return claimed_prefixes


def _ensure_identity_prefix_config(
    session,
    *,
    entity: str,
    domain_code: str,
    owner_repo_name: str,
    prefix: str,
) -> None:
    normalized_entity = str(entity or "").strip()
    normalized_domain = str(domain_code or "").strip().upper()
    normalized_owner = str(owner_repo_name or "").strip()
    normalized_prefix = str(prefix or "").strip().upper()
    if not normalized_entity:
        raise ValueError("Ursa TapDB identity entity is required")
    if not normalized_prefix:
        raise ValueError(f"Ursa TapDB identity prefix is required for {normalized_entity!r}")

    params = {
        "entity": normalized_entity,
        "domain_code": normalized_domain,
        "owner_repo_name": normalized_owner,
        "prefix": normalized_prefix,
    }
    existing = session.execute(
        text(
            """
            SELECT prefix
            FROM tapdb_identity_prefix_config
            WHERE entity = :entity
              AND domain_code = :domain_code
              AND issuer_app_code = :owner_repo_name
            """
        ),
        params,
    ).scalar_one_or_none()
    if existing is not None:
        existing_prefix = str(existing or "").strip().upper()
        if existing_prefix != normalized_prefix:
            raise RuntimeError(
                f"Ursa identity prefix config for entity {normalized_entity!r} in domain "
                f"{normalized_domain!r} is already seeded with prefix {existing_prefix!r}, "
                f"not {normalized_prefix!r}"
            )
        return

    session.execute(
        text(
            """
            INSERT INTO tapdb_identity_prefix_config(
              entity, domain_code, issuer_app_code, prefix
            )
            VALUES (:entity, :domain_code, :owner_repo_name, :prefix)
            """
        ),
        params,
    )


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
    core_config_dir = find_tapdb_core_config_dir()
    core_templates, core_issues = validate_template_configs([core_config_dir], strict=True)
    client_templates, client_issues = validate_template_configs(
        [template_config_root()], strict=True
    )
    errors = [issue for issue in [*core_issues, *client_issues] if issue.level == "error"]
    if errors:
        joined = "; ".join(issue.message for issue in errors)
        raise RuntimeError(f"Ursa template pack validation failed: {joined}")
    claim_ursa_template_prefixes(
        client_templates,
        domain_code=DEFAULT_TAPDB_DOMAIN_CODE,
        owner_repo_name=DEFAULT_TAPDB_OWNER_REPO,
        domain_registry_path=resolved_domain_registry_path,
        prefix_registry_path=resolved_prefix_registry_path,
    )
    seed_templates(
        session,
        core_templates,
        overwrite=True,
        core_config_dir=core_config_dir,
        domain_code=DEFAULT_TAPDB_DOMAIN_CODE,
        owner_repo_name="daylily-tapdb",
        domain_registry_path=resolved_domain_registry_path,
        prefix_registry_path=resolved_prefix_registry_path,
    )
    _ensure_identity_prefix_config(
        session,
        entity="generic_template",
        domain_code=DEFAULT_TAPDB_DOMAIN_CODE,
        owner_repo_name=DEFAULT_TAPDB_OWNER_REPO,
        prefix="RGX",
    )
    _ensure_identity_prefix_config(
        session,
        entity="generic_instance_lineage",
        domain_code=DEFAULT_TAPDB_DOMAIN_CODE,
        owner_repo_name=DEFAULT_TAPDB_OWNER_REPO,
        prefix=GENERIC_INSTANCE_LINEAGE_PREFIX,
    )
    _ensure_identity_prefix_config(
        session,
        entity="audit_log",
        domain_code=DEFAULT_TAPDB_DOMAIN_CODE,
        owner_repo_name=DEFAULT_TAPDB_OWNER_REPO,
        prefix=AUDIT_LOG_PREFIX,
    )
    seed_templates(
        session,
        client_templates,
        overwrite=True,
        core_config_dir=core_config_dir,
        domain_code=DEFAULT_TAPDB_DOMAIN_CODE,
        owner_repo_name=DEFAULT_TAPDB_OWNER_REPO,
        domain_registry_path=resolved_domain_registry_path,
        prefix_registry_path=resolved_prefix_registry_path,
    )
