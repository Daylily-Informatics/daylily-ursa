"""Request-level coverage for cluster-create job and options endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_cluster_create_job_endpoints_have_request_level_coverage():
    from daylily_ursa.routes.clusters import ClusterDependencies, create_clusters_router

    settings = MagicMock()
    settings.aws_profile = None
    settings.get_allowed_regions.return_value = ["us-west-2"]

    async def get_current_user():
        return {"is_admin": True, "email": "admin@example.com"}

    app = FastAPI()
    app.include_router(
        create_clusters_router(
            ClusterDependencies(settings=settings, get_current_user=get_current_user)
        )
    )

    class _FakeUrsaConfig:
        is_configured = True
        aws_profile = "lsmc"

        def get_allowed_regions(self):
            return ["us-west-2"]

    mock_session = MagicMock()

    mock_ec2 = MagicMock()
    mock_ec2.describe_key_pairs.return_value = {"KeyPairs": [{"KeyName": "kp-1"}]}

    mock_s3 = MagicMock()
    mock_s3.list_buckets.return_value = {"Buckets": [{"Name": "omics-analysis-test"}]}
    mock_s3.get_bucket_location.return_value = {"LocationConstraint": "us-west-2"}

    def _client(service_name: str, *, region_name: str):
        if service_name == "ec2":
            return mock_ec2
        if service_name == "s3":
            return mock_s3
        raise AssertionError(f"Unexpected service: {service_name}")

    mock_session.client.side_effect = _client

    with patch("daylily_ursa.ursa_config.get_ursa_config", return_value=_FakeUrsaConfig()):
        with patch("daylily_ursa.routes.clusters.boto3.Session", return_value=mock_session):
            with patch(
                "daylily_ursa.ephemeral_cluster.runner.list_cluster_create_jobs",
                return_value=[{"job_id": "ec-1"}],
            ):
                with patch(
                    "daylily_ursa.ephemeral_cluster.runner.read_cluster_create_job",
                    return_value={"job_id": "ec-1"},
                ):
                    with patch(
                        "daylily_ursa.ephemeral_cluster.runner.tail_job_log", return_value="log"
                    ):
                        with TestClient(app, base_url="https://testserver") as client:
                            assert client.get("/api/v2/clusters/create/jobs").status_code != 404
                            assert (
                                client.get("/api/v2/clusters/create/jobs/ec-1").status_code != 404
                            )
                            assert (
                                client.get("/api/v2/clusters/create/jobs/ec-1/logs").status_code
                                != 404
                            )
                            assert (
                                client.get(
                                    "/api/v2/clusters/create/options?region=us-west-2"
                                ).status_code
                                != 404
                            )
