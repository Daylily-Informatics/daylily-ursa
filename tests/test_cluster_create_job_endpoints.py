"""Focused request-level tests for cluster-create API endpoints."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_test_client() -> TestClient:
    from daylib.routes.clusters import ClusterDependencies, create_clusters_router

    settings = MagicMock()
    settings.aws_profile = None
    settings.get_allowed_regions.return_value = ["us-west-2"]

    async def get_current_user():
        return {"is_admin": True, "email": "admin@example.com"}

    app = FastAPI()
    app.include_router(create_clusters_router(ClusterDependencies(settings=settings, get_current_user=get_current_user)))
    return TestClient(app)


def test_create_cluster_starts_background_job_and_returns_urls() -> None:
    fake_job = SimpleNamespace(
        job_id="ec_test_1",
        cluster_name="test-cluster",
        region_az="us-west-2a",
        aws_profile="lsmc",
        log_path=Path("/tmp/ec_test_1.log"),
    )

    class _FakeUrsaConfig:
        is_configured = True
        aws_profile = "lsmc"

        def get_allowed_regions(self):
            return ["us-west-2"]

    with patch("daylib.ursa_config.get_ursa_config", return_value=_FakeUrsaConfig()):
        with patch("daylib.ephemeral_cluster.runner.start_create_job", return_value=fake_job) as start_job:
            with _build_test_client() as client:
                response = client.post(
                    "/api/clusters/create",
                    json={
                        "region_az": "us-west-2a",
                        "cluster_name": "test-cluster",
                        "ssh_key_name": "kp-1",
                        "s3_bucket_name": "omics-analysis-test",
                        "config_path": None,
                        "pass_on_warn": True,
                        "debug": False,
                    },
                )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "ec_test_1"
    assert payload["job_status_url"] == "/api/clusters/create/jobs/ec_test_1"
    assert payload["job_logs_url"] == "/api/clusters/create/jobs/ec_test_1/logs"

    start_job.assert_called_once_with(
        region_az="us-west-2a",
        cluster_name="test-cluster",
        ssh_key_name="kp-1",
        s3_bucket_name="omics-analysis-test",
        aws_profile="lsmc",
        contact_email="admin@example.com",
        config_path_override=None,
        pass_on_warn=True,
        debug=False,
    )


def test_cluster_create_job_endpoints_return_job_data() -> None:
    with patch("daylib.ephemeral_cluster.runner.list_cluster_create_jobs", return_value=[{"job_id": "ec-1", "status": "running"}]):
        with patch("daylib.ephemeral_cluster.runner.read_cluster_create_job", return_value={"job_id": "ec-1", "status": "running"}):
            with patch("daylib.ephemeral_cluster.runner.tail_job_log", return_value="log text"):
                with _build_test_client() as client:
                    jobs_response = client.get("/api/clusters/create/jobs?limit=20")
                    job_response = client.get("/api/clusters/create/jobs/ec-1")
                    logs_response = client.get("/api/clusters/create/jobs/ec-1/logs?lines=25")

    assert jobs_response.status_code == 200
    assert jobs_response.json() == {"jobs": [{"job_id": "ec-1", "status": "running"}]}
    assert job_response.status_code == 200
    assert job_response.json() == {"job_id": "ec-1", "status": "running"}
    assert logs_response.status_code == 200
    assert logs_response.json() == {"job_id": "ec-1", "lines": 25, "log": "log text"}


def test_cluster_create_options_filters_keypairs_and_region_buckets() -> None:
    class _FakeUrsaConfig:
        is_configured = True
        aws_profile = "lsmc"

        def get_allowed_regions(self):
            return ["us-west-2"]

    mock_session = MagicMock()
    mock_ec2 = MagicMock()
    mock_s3 = MagicMock()
    mock_ec2.describe_key_pairs.return_value = {"KeyPairs": [{"KeyName": "kp-b"}, {"KeyName": "kp-a"}]}
    mock_s3.list_buckets.return_value = {
        "Buckets": [
            {"Name": "omics-analysis-west"},
            {"Name": "omics-analysis-east"},
            {"Name": "not-omics"},
        ]
    }
    mock_s3.get_bucket_location.side_effect = lambda Bucket: {
        "omics-analysis-west": {"LocationConstraint": "us-west-2"},
        "omics-analysis-east": {"LocationConstraint": "us-east-1"},
        "not-omics": {"LocationConstraint": "us-west-2"},
    }[Bucket]

    def _client(service_name: str, *, region_name: str):
        if service_name == "ec2":
            return mock_ec2
        if service_name == "s3":
            return mock_s3
        raise AssertionError(f"Unexpected service: {service_name}")

    mock_session.client.side_effect = _client

    with patch("daylib.ursa_config.get_ursa_config", return_value=_FakeUrsaConfig()):
        with patch("daylib.routes.clusters.boto3.Session", return_value=mock_session) as session_ctor:
            with _build_test_client() as client:
                response = client.get("/api/clusters/create/options?region=us-west-2")

    assert response.status_code == 200
    assert response.json() == {
        "aws_profile": "lsmc",
        "region": "us-west-2",
        "keypairs": ["kp-a", "kp-b"],
        "buckets": ["omics-analysis-west"],
    }
    session_ctor.assert_called_once_with(profile_name="lsmc")
