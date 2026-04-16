from __future__ import annotations

import os

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
    def resolve_run_assignment(
        self, run_euid: str, flowcell_id: str, lane: str, library_barcode: str
    ):
        raise RuntimeError("not used")


class DummyS3Client:
    pass


def _fake_tapdb_app() -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def index():
        return {"tapdb": "ok"}

    return app


def _settings(
    *, mount_enabled: bool = True, tapdb_config_path: str = "/tmp/ursa-tapdb-config.yaml"
) -> Settings:
    return Settings(
        cors_origins="*",
        aws_profile=None,
        ursa_internal_api_key="test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_internal_output_bucket="ursa-internal",
        cognito_domain="auth.example.com",
        cognito_app_client_id="client-1",
        cognito_callback_url="https://localhost:8913/auth/callback",
        cognito_logout_url="https://localhost:8913/login",
        tapdb_config_path=tapdb_config_path,
        ursa_tapdb_mount_enabled=mount_enabled,
        ursa_tapdb_mount_path="/admin/tapdb",
        enable_auth=True,
    )


def test_mounted_route_exists_and_key_can_access(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "daylib_ursa.tapdb_mount._load_tapdb_admin_app", lambda **_kwargs: _fake_tapdb_app()
    )

    settings = _settings(mount_enabled=True)
    app = create_app(
        DummyStore(),
        bloom_client=DummyBloomClient(),
        settings=settings,
        s3_client=DummyS3Client(),
    )

    assert any(getattr(route, "path", None) == "/admin/tapdb" for route in app.routes)

    with TestClient(app) as client:
        response = client.get("/admin/tapdb/", headers={"X-API-Key": "test-key"})

    assert response.status_code == 200
    assert response.json() == {"tapdb": "ok"}


def test_mounted_route_denies_missing_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "daylib_ursa.tapdb_mount._load_tapdb_admin_app", lambda **_kwargs: _fake_tapdb_app()
    )

    app = create_app(
        DummyStore(),
        bloom_client=DummyBloomClient(),
        settings=_settings(mount_enabled=True),
        s3_client=DummyS3Client(),
    )

    with TestClient(app) as client:
        response = client.get("/admin/tapdb/")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_mounted_route_denies_wrong_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "daylib_ursa.tapdb_mount._load_tapdb_admin_app", lambda **_kwargs: _fake_tapdb_app()
    )

    app = create_app(
        DummyStore(),
        bloom_client=DummyBloomClient(),
        settings=_settings(mount_enabled=True),
        s3_client=DummyS3Client(),
    )

    with TestClient(app) as client:
        response = client.get("/admin/tapdb/", headers={"X-API-Key": "wrong-key"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_mounted_mode_does_not_inject_tapdb_admin_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("TAPDB_ADMIN_DISABLE_AUTH", raising=False)
    monkeypatch.delenv("TAPDB_ADMIN_DISABLED_USER_ROLE", raising=False)
    monkeypatch.delenv("TAPDB_ADMIN_SHARED_AUTH", raising=False)
    monkeypatch.setattr(
        "daylib_ursa.tapdb_mount._load_tapdb_admin_app", lambda **_kwargs: _fake_tapdb_app()
    )

    app = create_app(
        DummyStore(),
        bloom_client=DummyBloomClient(),
        settings=_settings(mount_enabled=True),
        s3_client=DummyS3Client(),
    )

    with TestClient(app) as client:
        response = client.get("/admin/tapdb/", headers={"X-API-Key": "test-key"})

    assert response.status_code == 200
    assert "TAPDB_ADMIN_DISABLE_AUTH" not in os.environ
    assert "TAPDB_ADMIN_DISABLED_USER_ROLE" not in os.environ
    assert "TAPDB_ADMIN_SHARED_AUTH" not in os.environ


def test_mount_enabled_fails_fast_when_tapdb_import_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    def _boom(**_kwargs):
        raise ModuleNotFoundError("admin.main")

    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", _boom)

    with pytest.raises(RuntimeError, match="Failed to import TapDB admin app"):
        create_app(
            DummyStore(),
            bloom_client=DummyBloomClient(),
            settings=_settings(mount_enabled=True),
            s3_client=DummyS3Client(),
        )


def test_mount_disabled_skips_tapdb_import(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    def _boom(**_kwargs):
        raise ModuleNotFoundError("admin.main")

    monkeypatch.setattr("daylib_ursa.tapdb_mount._load_tapdb_admin_app", _boom)
    app = create_app(
        DummyStore(),
        bloom_client=DummyBloomClient(),
        settings=_settings(mount_enabled=False),
        s3_client=DummyS3Client(),
    )
    assert all(getattr(route, "path", None) != "/admin/tapdb" for route in app.routes)


def test_mount_uses_explicit_tapdb_context(monkeypatch):
    captured: dict[str, str] = {}
    app = FastAPI()

    def _loader(**kwargs):
        captured.update({key: str(value) for key, value in kwargs.items()})
        return _fake_tapdb_app()

    from daylib_ursa.tapdb_mount import mount_tapdb_admin

    mount_tapdb_admin(
        app,
        _settings(mount_enabled=True, tapdb_config_path="/tmp/ursa-tapdb-config.yaml"),
        loader=_loader,
    )

    assert captured == {
        "tapdb_env": "dev",
        "config_path": "/tmp/ursa-tapdb-config.yaml",
        "client_id": "local",
        "database_name": "ursa",
    }
