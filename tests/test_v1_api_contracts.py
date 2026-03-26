from __future__ import annotations

from daylib_ursa.config import Settings
from daylib_ursa.workset_api import create_app


class DummyStore:
    def get_analysis(self, analysis_euid: str):
        _ = analysis_euid
        return None


class DummyBloomClient:
    def resolve_run_assignment(self, *_args, **_kwargs):  # pragma: no cover - not used
        raise AssertionError("not used")


def _settings() -> Settings:
    return Settings(
        cors_origins="*",
        ursa_internal_api_key="ursa-test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_internal_output_bucket="ursa-internal",
        ursa_tapdb_mount_enabled=False,
    )


def test_public_routes_are_versioned_and_legacy_customer_routes_are_absent() -> None:
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings())
    paths = {
        getattr(route, "path", "")
        for route in app.routes
        if getattr(route, "path", "")
    }
    public_api_paths = {path for path in paths if path.startswith("/api")}
    assert public_api_paths
    assert all(path.startswith("/api/v1/") for path in public_api_paths)
    assert not any(path.startswith("/api/customers/") for path in public_api_paths)


def test_phase_one_route_families_exist() -> None:
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings())
    paths = {
        getattr(route, "path", "")
        for route in app.routes
        if getattr(route, "path", "")
    }
    expected = {
        "/api/v1/me",
        "/api/v1/analyses",
        "/api/v1/analyses/ingest",
        "/api/v1/worksets",
        "/api/v1/manifests",
        "/api/v1/buckets",
        "/api/v1/artifacts/import",
        "/api/v1/clusters",
        "/api/v1/clusters/jobs",
        "/api/v1/user-tokens",
        "/api/v1/admin/user-tokens",
        "/api/v1/admin/users",
        "/api/v1/admin/client-registrations",
        "/api/v1/admin/client-registrations/{client_registration_euid}",
        "/api/v1/admin/client-registrations/{client_registration_euid}/tokens",
    }
    assert expected.issubset(paths)
