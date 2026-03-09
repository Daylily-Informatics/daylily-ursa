from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from daylib_ursa.config import Settings
from daylib_ursa.workset_api import create_app


class DummyStore:
    pass


class DummyBloomClient:
    pass


class FakeClusterService:
    def __init__(self, clusters):
        self._clusters = clusters

    def get_all_clusters(self, force_refresh: bool = False):
        return list(self._clusters)

    def delete_cluster(self, cluster_name: str, region: str):
        return {"cluster_name": cluster_name, "region": region}


def _settings() -> Settings:
    return Settings(
        cors_origins="*",
        ursa_internal_api_key="ursa-test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_allowed_regions="us-west-2,us-east-1",
        ursa_cost_monitor_regions="us-west-2,us-east-1",
        ursa_cost_monitor_partitions="i192,g192",
    )


def _build_app(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("daylib_ursa.portal.get_cluster_service", lambda *args, **kwargs: FakeClusterService([]))
    monkeypatch.setattr(
        "daylib_ursa.portal._validate_cluster_create_identity",
        lambda *args, **kwargs: {"account_id": "123456789012", "arn": "arn:aws:iam::123456789012:user/test"},
    )
    monkeypatch.setattr("daylib_ursa.pricing_monitor.PricingMonitor.start", lambda self: None)
    return create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings())


def test_pricing_snapshot_api_returns_grouped_payload(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path)

    with TestClient(app) as client:
        store = app.state.portal_state
        run = store.create_pricing_run(trigger="manual", requested_by="admin")
        store.mark_pricing_run_running(run["run_id"])
        store.save_pricing_snapshot(
            run["run_id"],
            {
                "captured_at": "2026-03-08T12:00:00Z",
                "cluster_config_path": "/tmp/prod_cluster.yaml",
                "points": [
                    {
                        "captured_at": "2026-03-08T12:00:00Z",
                        "region": "us-west-2",
                        "availability_zone": "us-west-2a",
                        "partition": "i192",
                        "instance_type": "c7i.48xlarge",
                        "vcpu_count": 192,
                        "hourly_spot_price": 9.6,
                        "vcpu_cost_per_hour": 0.05,
                    },
                    {
                        "captured_at": "2026-03-08T12:00:00Z",
                        "region": "us-west-2",
                        "availability_zone": "us-west-2a",
                        "partition": "i192",
                        "instance_type": "m7i.48xlarge",
                        "vcpu_count": 192,
                        "hourly_spot_price": 10.56,
                        "vcpu_cost_per_hour": 0.055,
                    },
                ],
            },
        )

        response = client.get(
            "/api/pricing-snapshots?region=us-west-2&partitions=i192",
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshots"][0]["region"] == "us-west-2"
    assert payload["snapshots"][0]["partitions"][0]["partition"] == "i192"
    assert payload["snapshots"][0]["partitions"][0]["availability_zones"][0]["box"]["median"] == 0.055


def test_manual_pricing_run_endpoint_uses_monitor_queue(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path)

    with TestClient(app) as client:
        app.state.pricing_monitor.queue_capture = lambda **kwargs: {"run_id": 7, "status": "queued"}
        response = client.post(
            "/api/pricing-snapshots/run",
            headers={"X-Ursa-Admin": "true"},
        )

    assert response.status_code == 200
    assert response.json() == {"run_id": 7, "status": "queued"}


def test_cluster_create_enqueues_background_job(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "daylib_ursa.portal.start_create_job",
        lambda **kwargs: SimpleNamespace(
            job_id="job-123",
            cluster_name=kwargs["cluster_name"],
            region_az=kwargs["region_az"],
            status="running",
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/clusters/create",
            headers={"X-Ursa-Admin": "true"},
            json={
                "region_az": "us-east-1a",
                "cluster_name": "daylily-use1-smoke",
                "ssh_key_name": "key-east",
                "s3_bucket_name": "bucket-east",
                "pass_on_warn": True,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-123",
        "cluster_name": "daylily-use1-smoke",
        "region_az": "us-east-1a",
        "status": "running",
    }


def test_cluster_create_requires_admin_privilege(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/clusters/create",
            json={
                "region_az": "us-east-1a",
                "cluster_name": "daylily-use1-smoke",
                "ssh_key_name": "key-east",
                "s3_bucket_name": "bucket-east",
            },
        )

    assert response.status_code == 403
