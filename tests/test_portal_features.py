from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from daylib_ursa.cluster_service import ClusterInfo
from daylib_ursa.config import get_settings_for_testing
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


def _build_app(monkeypatch, tmp_path, clusters=None):
    monkeypatch.setenv("HOME", str(tmp_path))
    fake_service = FakeClusterService(clusters or [])
    monkeypatch.setattr("daylib_ursa.portal.get_cluster_service", lambda *args, **kwargs: fake_service)
    monkeypatch.setattr("daylib_ursa.portal.PricingMonitor.start", lambda self: None)
    monkeypatch.setattr(
        "daylib_ursa.portal._validate_cluster_create_identity",
        lambda settings, *, region_az: {"account_id": "000000000000", "arn": "arn:aws:iam::000000000000:user/test"},
    )
    settings = get_settings_for_testing(
        enable_auth=False,
        ursa_internal_api_key="test-key",
        ursa_tapdb_mount_enabled=False,
    )
    return create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=settings)


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

        response = client.get("/api/pricing-snapshots?region=us-west-2&partitions=i192")

    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshots"][0]["region"] == "us-west-2"
    assert payload["snapshots"][0]["partitions"][0]["partition"] == "i192"
    assert payload["snapshots"][0]["partitions"][0]["availability_zones"][0]["box"]["median"] == 0.055


def test_admin_workset_submit_without_cluster_enqueues_cluster_create(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path, clusters=[])
    monkeypatch.setattr(
        "daylib_ursa.portal.start_create_job",
        lambda **kwargs: SimpleNamespace(
            job_id="job-123",
            cluster_name=kwargs["cluster_name"],
            region_az=kwargs["region_az"],
            status="running",
        ),
    )

    payload = {
        "workset_name": "bootstrap-me",
        "pipeline_type": "germline_wgs_snv",
        "reference_genome": "GRCh38",
        "priority": "normal",
        "workset_type": "ruo",
        "enable_qc": True,
        "archive_results": True,
        "target_region": "us-east-1",
        "cluster_bootstrap": {
            "region": "us-east-1",
            "az_suffix": "a",
            "ssh_key_name": "key-east",
            "s3_bucket_name": "bucket-east",
        },
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/customers/default-customer/worksets",
            headers={"X-Ursa-Admin": "true"},
            json=payload,
        )
        worksets = client.get("/api/customers/default-customer/worksets").json()["worksets"]

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "pending_cluster_creation"
    assert body["cluster_create_job_id"] == "job-123"
    assert worksets[0]["state"] == "pending_cluster_creation"


def test_non_admin_workset_submit_without_cluster_is_blocked(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path, clusters=[])

    payload = {
        "workset_name": "blocked",
        "pipeline_type": "germline_wgs_snv",
        "reference_genome": "GRCh38",
        "target_region": "eu-central-1",
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/customers/default-customer/worksets",
            headers={"X-Ursa-Admin": "false"},
            json=payload,
        )

    assert response.status_code == 409
    assert "contact an admin" in response.json()["detail"]


def test_pending_workset_reconciles_when_cluster_appears(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path, clusters=[])
    monkeypatch.setattr(
        "daylib_ursa.portal.start_create_job",
        lambda **kwargs: SimpleNamespace(
            job_id="job-456",
            cluster_name=kwargs["cluster_name"],
            region_az=kwargs["region_az"],
            status="running",
        ),
    )

    payload = {
        "workset_name": "wait-for-cluster",
        "pipeline_type": "germline_wgs_snv",
        "reference_genome": "GRCh38",
        "target_region": "us-west-2",
        "cluster_bootstrap": {
            "region": "us-west-2",
            "az_suffix": "a",
            "ssh_key_name": "key-west",
            "s3_bucket_name": "bucket-west",
        },
    }

    with TestClient(app) as client:
        create_response = client.post(
            "/api/customers/default-customer/worksets",
            headers={"X-Ursa-Admin": "true"},
            json=payload,
        )
        workset_id = create_response.json()["workset_id"]
        fake_service = FakeClusterService(
            [
                ClusterInfo(
                    cluster_name="running-west",
                    region="us-west-2",
                    cluster_status="CREATE_COMPLETE",
                    compute_fleet_status="RUNNING",
                )
            ]
        )
        monkeypatch.setattr("daylib_ursa.portal.get_cluster_service", lambda *args, **kwargs: fake_service)
        detail_response = client.get(f"/api/customers/default-customer/worksets/{workset_id}")

    assert detail_response.status_code == 200
    body = detail_response.json()
    assert body["state"] == "ready"
    assert body["cluster_name"] == "running-west"


def test_manual_pricing_run_endpoint_uses_monitor_queue(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path)

    with TestClient(app) as client:
        app.state.pricing_monitor.queue_capture = lambda **kwargs: {"run_id": 7, "status": "queued"}
        response = client.post("/api/pricing-snapshots/run", headers={"X-Ursa-Admin": "true"})

    assert response.status_code == 200
    assert response.json() == {"run_id": 7, "status": "queued"}


def test_cluster_create_preflight_failure_returns_502(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path, clusters=[])

    def _raise_preflight(settings, *, region_az):
        raise RuntimeError("missing Arn")

    monkeypatch.setattr("daylib_ursa.portal._validate_cluster_create_identity", _raise_preflight)
    monkeypatch.setattr("daylib_ursa.portal.start_create_job", lambda **kwargs: (_ for _ in ()).throw(AssertionError))

    payload = {
        "region_az": "us-west-2d",
        "cluster_name": "daylily-1773055196",
        "ssh_key_name": "lsmc-omics-us-west-2",
        "s3_bucket_name": "lsmc-dayoa-omics-analysis-us-west-2",
    }

    with TestClient(app) as client:
        response = client.post("/api/clusters/create", headers={"X-Ursa-Admin": "true"}, json=payload)

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "AWS identity preflight failed" in detail
    assert "missing Arn" in detail


def test_cluster_create_daylily_preflight_failure_returns_502(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path, clusters=[])
    monkeypatch.setattr(
        "daylib_ursa.portal.start_create_job",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("daylily-ec preflight failed: missing Arn")),
    )

    payload = {
        "region_az": "us-west-2d",
        "cluster_name": "daylily-1773057991",
        "ssh_key_name": "lsmc-omics-us-west-2",
        "s3_bucket_name": "lsmc-dayoa-omics-analysis-us-west-2",
    }

    with TestClient(app) as client:
        response = client.post("/api/clusters/create", headers={"X-Ursa-Admin": "true"}, json=payload)

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "Cluster create preflight failed" in detail
    assert "daylily-ec preflight failed: missing Arn" in detail
