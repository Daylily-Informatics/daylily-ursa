"""TapDB admin mount integration for the Ursa FastAPI app."""

from __future__ import annotations

import hmac
import importlib
import logging
import os
from typing import Callable

from fastapi import FastAPI
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from daylib_ursa.config import Settings

LOGGER = logging.getLogger("daylily.tapdb_mount")


def _force_mounted_auth_bypass_env() -> None:
    """Force TapDB into mounted mode without TapDB-local auth."""
    os.environ["TAPDB_ADMIN_DISABLE_AUTH"] = "true"
    os.environ["TAPDB_ADMIN_DISABLED_USER_ROLE"] = "admin"
    os.environ["TAPDB_ADMIN_DISABLED_USER_EMAIL"] = "ursa-mounted-tapdb-admin@localhost"
    os.environ["TAPDB_ADMIN_SHARED_AUTH"] = "false"


def _load_tapdb_admin_app() -> ASGIApp:
    """Load the TapDB admin FastAPI app lazily."""
    module = importlib.import_module("admin.main")
    tapdb_app = getattr(module, "app", None)
    if tapdb_app is None:
        raise RuntimeError("admin.main does not export an 'app'")
    return tapdb_app


class UrsaTapdbAdminGate:
    """ASGI guard that requires Ursa internal API key."""

    def __init__(self, app: ASGIApp, *, internal_api_key: str) -> None:
        self._app = app
        self._internal_api_key = str(internal_api_key or "")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self._app(scope, receive, send)
            return

        if not self._is_authenticated(scope):
            await self._deny_not_authenticated(scope, receive, send)
            return

        await self._app(scope, receive, send)

    def _is_authenticated(self, scope: Scope) -> bool:
        connection = HTTPConnection(scope)
        provided = str(connection.headers.get("x-api-key") or "")
        expected = str(self._internal_api_key or "")
        if not provided or not expected:
            return False
        return hmac.compare_digest(provided, expected)

    async def _deny_not_authenticated(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return
        response = JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
        await response(scope, receive, send)


def _normalize_mount_path(raw_path: str) -> str:
    path = str(raw_path or "").strip()
    if not path:
        raise RuntimeError("URSA_TAPDB_MOUNT_PATH must not be empty")
    if not path.startswith("/"):
        raise RuntimeError("URSA_TAPDB_MOUNT_PATH must start with '/'")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return path


def mount_tapdb_admin(
    app: FastAPI,
    settings: Settings,
    *,
    loader: Callable[[], ASGIApp] | None = None,
) -> None:
    """Mount TapDB admin under Ursa with API-key gating."""
    if not settings.ursa_tapdb_mount_enabled:
        LOGGER.info("TapDB mount disabled by URSA_TAPDB_MOUNT_ENABLED")
        return

    mount_path = _normalize_mount_path(settings.ursa_tapdb_mount_path)
    if any(getattr(route, "path", None) == mount_path for route in app.routes):
        raise RuntimeError(f"Cannot mount TapDB: path already in use: {mount_path}")

    _force_mounted_auth_bypass_env()
    resolved_loader = loader or _load_tapdb_admin_app
    try:
        tapdb_app = resolved_loader()
    except Exception as exc:
        raise RuntimeError(
            "Failed to import TapDB admin app for mounted mode. "
            "Install daylily-tapdb admin extras or disable URSA_TAPDB_MOUNT_ENABLED."
        ) from exc

    guarded_app = UrsaTapdbAdminGate(
        tapdb_app,
        internal_api_key=settings.ursa_internal_api_key,
    )
    app.mount(mount_path, guarded_app, name="ursa-tapdb-admin")
    LOGGER.info("Mounted TapDB admin at %s", mount_path)
