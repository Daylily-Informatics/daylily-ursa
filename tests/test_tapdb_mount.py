from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from daylib_ursa.config import Settings
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
        aws_profile=None,
        ursa_internal_api_key="test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_internal_output_bucket="ursa-internal",
        ursa_tapdb_mount_enabled=mount_enabled,
        ursa_tapdb_mount_path="/admin/tapdb",
        enable_auth=True,
    )


def test_mounted_route_exists_and_key_can_access(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", lambda: _fake_tapdb_app())

    settings = _settings(mount_enabled=True)
    with patch("daylib_ursa.workset_api.RegionAwareS3Client", return_value=object()):
        app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=settings)

    assert any(getattr(route, "path", None) == "/admin/tapdb" for route in app.routes)

    with TestClient(app) as client:
        response = client.get("/admin/tapdb/", headers={"X-API-Key": "test-key"})

    assert response.status_code == 200
    assert response.json() == {"tapdb": "ok"}


def test_mounted_route_denies_missing_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", lambda: _fake_tapdb_app())

    with patch("daylib_ursa.workset_api.RegionAwareS3Client", return_value=object()):
        app = create_app(
            DummyStore(),
            bloom_client=DummyBloomClient(),
            settings=_settings(mount_enabled=True),
        )

    with TestClient(app) as client:
        response = client.get("/admin/tapdb/")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_mounted_route_denies_wrong_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", lambda: _fake_tapdb_app())

    with patch("daylib_ursa.workset_api.RegionAwareS3Client", return_value=object()):
        app = create_app(
            DummyStore(),
            bloom_client=DummyBloomClient(),
            settings=_settings(mount_enabled=True),
        )

    with TestClient(app) as client:
        response = client.get("/admin/tapdb/", headers={"X-API-Key": "wrong-key"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_mounted_mode_forces_tapdb_local_auth_bypass(monkeypatch, tmp_path):
    captured: dict[str, str | None] = {}

    def _loader():
        captured["disable_auth"] = os.environ.get("TAPDB_ADMIN_DISABLE_AUTH")
        captured["disabled_role"] = os.environ.get("TAPDB_ADMIN_DISABLED_USER_ROLE")
        captured["shared_auth"] = os.environ.get("TAPDB_ADMIN_SHARED_AUTH")
        return _fake_tapdb_app()

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", _loader)

    with patch("daylib_ursa.workset_api.RegionAwareS3Client", return_value=object()):
        app = create_app(
            DummyStore(),
            bloom_client=DummyBloomClient(),
            settings=_settings(mount_enabled=True),
        )

    with TestClient(app) as client:
        response = client.get("/admin/tapdb/", headers={"X-API-Key": "test-key"})

    assert response.status_code == 200
    assert captured == {
        "disable_auth": "true",
        "disabled_role": "admin",
        "shared_auth": "false",
    }


def test_mount_enabled_fails_fast_when_tapdb_import_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    def _boom():
        raise ModuleNotFoundError("admin.main")

    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", _boom)

    with patch("daylib_ursa.workset_api.RegionAwareS3Client", return_value=object()):
        with pytest.raises(RuntimeError, match="Failed to import TapDB admin app"):
            create_app(
                DummyStore(),
                bloom_client=DummyBloomClient(),
                settings=_settings(mount_enabled=True),
            )


def test_mount_disabled_skips_tapdb_import(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    def _boom():
        raise ModuleNotFoundError("admin.main")

    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", _boom)
    with patch("daylib_ursa.workset_api.RegionAwareS3Client", return_value=object()):
        app = create_app(
            DummyStore(),
            bloom_client=DummyBloomClient(),
            settings=_settings(mount_enabled=False),
        )
    assert all(getattr(route, "path", None) != "/admin/tapdb" for route in app.routes)
