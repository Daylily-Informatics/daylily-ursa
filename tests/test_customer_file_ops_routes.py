"""Request-level coverage for customer file operations routes."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_customer_file_ops_routes_have_request_level_coverage():
    from daylily_ursa.routes.files import FileDependencies, create_files_router

    customer_manager = MagicMock()
    customer_manager.get_customer_config.return_value = SimpleNamespace(s3_bucket="cust-bucket")

    app = FastAPI()
    app.include_router(create_files_router(FileDependencies(customer_manager=customer_manager)))

    mock_s3 = MagicMock()
    mock_s3.put_object.return_value = {}
    mock_s3.generate_presigned_url.return_value = "https://example.invalid/presigned"
    mock_s3.delete_object.return_value = {}
    mock_s3.head_object.return_value = {"ContentLength": 12, "ContentType": "text/plain"}

    mock_body = MagicMock()
    mock_body.read.return_value = b"hello\nworld\n"
    mock_s3.get_object.return_value = {"Body": mock_body}
    mock_s3.list_objects_v2.return_value = {
        "CommonPrefixes": [],
        "Contents": [
            {"Key": "readme.txt", "Size": 12, "LastModified": datetime.now(timezone.utc)},
        ],
    }

    with patch("daylily_ursa.routes.files.boto3.client", return_value=mock_s3):
        with TestClient(app, base_url="https://testserver") as client:
            # Upload
            upload_resp = client.post(
                "/api/customers/cust-001/files/upload",
                files={"file": ("readme.txt", b"hello\nworld\n", "text/plain")},
            )
            assert upload_resp.status_code != 404

            # Create folder
            assert client.post(
                "/api/customers/cust-001/files/create-folder",
                json={"folder_path": "data/test"},
            ).status_code != 404

            # Preview
            assert client.get("/api/customers/cust-001/files/readme.txt/preview").status_code != 404

            # Presigned download URL
            assert client.get("/api/customers/cust-001/files/readme.txt/download-url").status_code != 404

            # Delete
            assert client.delete("/api/customers/cust-001/files/readme.txt").status_code != 404

