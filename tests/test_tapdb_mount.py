from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from daylib_ursa.config import Settings
from daylib_ursa.portal_auth import PORTAL_SESSION_COOKIE_NAME, encode_portal_session
from daylib_ursa.workset_api import create_app


class DummyStore:
    def ingest_analysis(self, **kwargs):
        raise RuntimeError("not used")

    def get_analysis(self, analysis_euid: str):
        return None

    def update_analysis_state(self, analysis_euid: str, **kwargs):
        raise KeyError("not used")

    def add_artifact(self, analysis_euid: str, **kwargs):
        raise KeyError("not used")

    def set_review_state(self, analysis_euid: str, **kwargs):
        raise KeyError("not used")

    def mark_returned(self, analysis_euid: str, **kwargs):
        raise KeyError("not used")


class DummyBloomClient:
    def resolve_run_assignment(self, run_euid: str, flowcell_id: str, lane: str, library_barcode: str):
        raise RuntimeError("not used")


def _fake_tapdb_app() -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def index():
        return {"tapdb": "ok"}

    return app


def _settings(*, mount_enabled: bool = True) -> Settings:
    return Settings(
        cors_origins="*",
        ursa_internal_api_key="test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_tapdb_mount_enabled=mount_enabled,
        ursa_tapdb_mount_path="/admin/tapdb",
        enable_auth=True,
    )


def _admin_cookie(secret: str, *, is_admin: bool) -> str:
    return encode_portal_session(
        secret,
        {
            "logged_in": True,
            "is_admin": is_admin,
            "user_email": "tapdb-test@lsmc.bio",
            "customer_id": "default-customer",
        },
    )


def test_mounted_route_exists_and_admin_can_access(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("daylib_ursa.workset_api.mount_portal", lambda app, settings: None)
    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", lambda: _fake_tapdb_app())

    settings = _settings(mount_enabled=True)
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=settings)

    assert any(getattr(route, "path", None) == "/admin/tapdb" for route in app.routes)

    with TestClient(app) as client:
        client.cookies.set(
            PORTAL_SESSION_COOKIE_NAME,
            _admin_cookie(settings.session_secret_key, is_admin=True),
        )
        response = client.get("/admin/tapdb/")

    assert response.status_code == 200
    assert response.json() == {"tapdb": "ok"}


def test_mounted_route_denies_unauthenticated_with_login_redirect(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("daylib_ursa.workset_api.mount_portal", lambda app, settings: None)
    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", lambda: _fake_tapdb_app())

    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings(mount_enabled=True))
    with TestClient(app) as client:
        response = client.get("/admin/tapdb/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/portal/login"


def test_mounted_route_denies_non_admin_with_403(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("daylib_ursa.workset_api.mount_portal", lambda app, settings: None)
    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", lambda: _fake_tapdb_app())

    settings = _settings(mount_enabled=True)
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=settings)

    with TestClient(app) as client:
        client.cookies.set(
            PORTAL_SESSION_COOKIE_NAME,
            _admin_cookie(settings.session_secret_key, is_admin=False),
        )
        response = client.get("/admin/tapdb/", follow_redirects=False)

    assert response.status_code == 403
    assert response.json() == {"detail": "Admin access required"}


def test_mounted_mode_forces_tapdb_local_auth_bypass(monkeypatch, tmp_path):
    captured: dict[str, str | None] = {}

    def _loader():
        captured["disable_auth"] = os.environ.get("TAPDB_ADMIN_DISABLE_AUTH")
        captured["disabled_role"] = os.environ.get("TAPDB_ADMIN_DISABLED_USER_ROLE")
        captured["shared_auth"] = os.environ.get("TAPDB_ADMIN_SHARED_AUTH")
        return _fake_tapdb_app()

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("daylib_ursa.workset_api.mount_portal", lambda app, settings: None)
    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", _loader)

    settings = _settings(mount_enabled=True)
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=settings)

    with TestClient(app) as client:
        client.cookies.set(
            PORTAL_SESSION_COOKIE_NAME,
            _admin_cookie(settings.session_secret_key, is_admin=True),
        )
        response = client.get("/admin/tapdb/")

    assert response.status_code == 200
    assert captured == {
        "disable_auth": "true",
        "disabled_role": "admin",
        "shared_auth": "false",
    }


def test_mount_enabled_fails_fast_when_tapdb_import_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("daylib_ursa.workset_api.mount_portal", lambda app, settings: None)

    def _boom():
        raise ModuleNotFoundError("admin.main")

    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", _boom)

    with pytest.raises(RuntimeError, match="Failed to import TapDB admin app"):
        create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings(mount_enabled=True))


def test_mount_disabled_skips_tapdb_import(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("daylib_ursa.workset_api.mount_portal", lambda app, settings: None)

    def _boom():
        raise ModuleNotFoundError("admin.main")

    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", _boom)
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings(mount_enabled=False))
    assert all(getattr(route, "path", None) != "/admin/tapdb" for route in app.routes)
