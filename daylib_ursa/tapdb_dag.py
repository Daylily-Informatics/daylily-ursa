"""Canonical TapDB DAG API integration for Ursa."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI

from daylily_tapdb.web import build_dag_capability_advertisement, create_tapdb_dag_router

from daylib_ursa.auth import get_observability_user
from daylib_ursa.config import Settings


def resolve_tapdb_dag_config_path(settings: Settings) -> str:
    """Return the explicit TapDB config path used by Ursa's DAG API."""

    raw_path = str(getattr(settings, "tapdb_config_path", "") or "").strip()
    if not raw_path:
        return ""
    resolved = Path(raw_path).expanduser()
    if not resolved.is_absolute():
        raise RuntimeError("Ursa TapDB DAG requires an explicit absolute tapdb_config_path.")
    return str(resolved)


def mount_tapdb_dag_api(app: FastAPI, settings: Settings) -> bool:
    """Mount the canonical `/api/dag/*` TapDB router when Ursa has explicit config."""

    config_path = resolve_tapdb_dag_config_path(settings)
    if not config_path:
        app.state.tapdb_dag_configured = False
        return False

    app.include_router(
        create_tapdb_dag_router(
            config_path=config_path,
            env_name=settings.tapdb_env,
            service_name="ursa",
        ),
        dependencies=[Depends(get_observability_user)],
    )
    app.state.tapdb_dag_configured = True
    app.state.tapdb_dag_config_path = config_path
    return True


def ursa_tapdb_dag_obs_services_fragment() -> dict[str, Any]:
    """Return Ursa-facing obs_services metadata for the TapDB DAG contract."""

    return build_dag_capability_advertisement(
        base_path="/api/dag",
        auth="operator_or_service_token",
    )
