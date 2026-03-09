from __future__ import annotations

import base64
import json
from dataclasses import replace

from fastapi.testclient import TestClient

from daylib_ursa.analysis_store import (
    AnalysisRecord,
    AnalysisState,
    ReviewState,
)
from daylib_ursa.config import Settings
from daylib_ursa.workset_api import create_app


class DummyStore:
    def __init__(self) -> None:
        self.record = AnalysisRecord(
            analysis_euid="AN-1",
            run_euid="RUN-1",
            flowcell_id="FLOW-1",
            lane="1",
            library_barcode="LIB-1",
            sequenced_library_assignment_euid="SQA-1",
            atlas_tenant_id="TEN-1",
            atlas_trf_euid="TRF-1",
            atlas_test_euid="TST-1",
            atlas_test_process_item_euid="TPC-1",
            analysis_type="beta-default",
            state=AnalysisState.INGESTED.value,
            review_state=ReviewState.PENDING.value,
            result_status="PENDING",
            run_folder="s3://analysis-bucket/RUN-1/",
            artifact_bucket="analysis-bucket",
            result_payload={},
            metadata={},
            created_at="2026-03-07T00:00:00Z",
            updated_at="2026-03-07T00:00:00Z",
            atlas_return={},
            artifacts=[],
        )

    def get_analysis(self, analysis_euid: str):
        return self.record if analysis_euid == self.record.analysis_euid else None

    def ingest_analysis(self, **kwargs):
        resolution = kwargs["resolution"]
        self.record = replace(
            self.record,
            run_euid=resolution.run_euid,
            flowcell_id=resolution.flowcell_id,
            lane=resolution.lane,
            library_barcode=resolution.library_barcode,
            analysis_type=kwargs["analysis_type"],
            artifact_bucket=kwargs["artifact_bucket"],
        )
        return self.record


class DummyBloomClient:
    def resolve_run_assignment(self, run_euid: str, flowcell_id: str, lane: str, library_barcode: str):
        raise RuntimeError("not used")


def _b64url(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _fake_id_token(claims: dict) -> str:
    return f"{_b64url({'alg': 'none', 'typ': 'JWT'})}.{_b64url(claims)}.x"


def _settings() -> Settings:
    return Settings(
        cors_origins="*",
        ursa_internal_api_key="test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        enable_auth=True,
        cognito_domain="daylily-ursa-5r8giqv5p.auth.us-west-2.amazoncognito.com",
        cognito_app_client_id="34g35v8tpurbe309a8e5t5ot7i",
        cognito_user_pool_id="us-west-2_5r8gIqV5P",
        cognito_region="us-west-2",
        ursa_tapdb_mount_enabled=False,
    )


def test_cognito_callback_sets_session_and_populates_dashboard(monkeypatch):
    async def _fake_exchange_code_for_tokens(**kwargs):
        return {
            "id_token": _fake_id_token(
                {
                    "sub": "sub-123",
                    "email": "jmajor@lsmc.bio",
                    "name": "Jordan Major",
                    "custom:customer_id": "lsmc-main",
                    "custom:customer_name": "LSMC Main",
                    "custom:s3_bucket": "lsmc-assigned-bucket",
                    "custom:cost_center": "RND-42",
                    "cognito:groups": ["admin"],
                }
            ),
            "access_token": "access-token",
        }

    async def _fake_fetch_userinfo(**kwargs):
        return {"email": "jmajor@lsmc.bio", "sub": "sub-123", "name": "Jordan Major"}

    monkeypatch.setattr("daylib_ursa.workset_api.exchange_code_for_tokens", _fake_exchange_code_for_tokens)
    monkeypatch.setattr("daylib_ursa.workset_api.fetch_userinfo", _fake_fetch_userinfo)
    monkeypatch.setattr(
        "daylib_ursa.workset_api.ensure_customer_onboarding",
        lambda identity, settings: {
            **identity,
            "s3_bucket": identity.get("s3_bucket") or "lsmc-assigned-bucket",
        },
    )
    monkeypatch.setattr("daylib_ursa.portal.PricingMonitor.start", lambda self: None)

    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings())

    with TestClient(app) as client:
        callback = client.get("/auth/callback?code=abc123", follow_redirects=False)
        dashboard = client.get("/portal")
        account = client.get("/portal/account")

    assert callback.status_code == 307
    assert callback.headers["location"] == "/portal"
    assert "ursa_portal_session=" in callback.headers.get("set-cookie", "")
    assert dashboard.status_code == 200
    assert "jmajor@lsmc.bio" in dashboard.text
    assert "LSMC Main" in dashboard.text
    assert "lsmc-assigned-bucket" in dashboard.text
    assert account.status_code == 200
    assert "jmajor@lsmc.bio" in account.text
    assert "s3://lsmc-assigned-bucket" in account.text
