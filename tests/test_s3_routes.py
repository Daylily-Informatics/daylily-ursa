"""Request-level coverage for /api/s3/* routes without real AWS calls."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_s3_routes_have_request_level_coverage():
    from daylib.routes.s3 import S3Dependencies, create_s3_router
    from daylib.s3_bucket_validator import BucketValidationResult

    settings = MagicMock()
    settings.get_effective_region.return_value = "us-west-2"
    settings.aws_profile = None

    async def get_current_user():
        return None

    deps = S3Dependencies(settings=settings, get_current_user=get_current_user)
    app = FastAPI()
    app.state.settings = SimpleNamespace(get_effective_region=lambda: "us-west-2", aws_profile=None)
    app.include_router(create_s3_router(deps))

    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [{"Contents": []}]

    mock_s3_client = MagicMock()
    mock_s3_client.get_paginator.return_value = mock_paginator

    mock_boto_session = MagicMock()
    mock_boto_session.client.return_value = mock_s3_client

    mock_validator_instance = MagicMock()
    mock_validator_instance.validate_bucket.return_value = BucketValidationResult(
        bucket_name="example-bucket",
        exists=True,
        accessible=True,
        can_read=True,
        can_write=True,
        can_list=True,
        region="us-west-2",
    )
    mock_validator_instance.get_setup_instructions.return_value = None
    mock_validator_instance.generate_iam_policy_for_bucket.return_value = {"Statement": []}
    mock_validator_instance.generate_customer_bucket_policy.return_value = {"Statement": []}

    with patch("daylib.routes.s3.boto3.Session", return_value=mock_boto_session):
        with patch("daylib.s3_bucket_validator.S3BucketValidator", return_value=mock_validator_instance):
            with TestClient(app) as client:
                assert client.post("/api/s3/discover-samples", json={"bucket": "b", "prefix": ""}).status_code != 404
                assert client.post("/api/s3/validate-bucket", json={"bucket": "example-bucket"}).status_code != 404
                assert client.get("/api/s3/iam-policy/example-bucket").status_code != 404
                assert client.get("/api/s3/bucket-policy/example-bucket").status_code != 404

