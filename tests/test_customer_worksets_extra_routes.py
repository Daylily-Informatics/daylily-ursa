"""Request-level coverage for additional customer workset routes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_customer_worksets_extra_routes_have_request_level_coverage():
    from daylily_ursa.routes.customer_worksets import CustomerWorksetDependencies, create_customer_worksets_router
    from daylily_ursa.workset_state_db import WorksetStateDB

    state_db = MagicMock(spec=WorksetStateDB)

    # Workset owned by cust-001; omit execution_headnode_ip so logs endpoint avoids SSH.
    original = {
        "workset_id": "ws-123",
        "customer_id": "cust-001",
        "state": "error",
        "bucket": "cust-bucket",
        "prefix": "worksets/ws-123/",
        "priority": "normal",
        "workset_type": "ruo",
        "created_at": "2026-03-04T00:00:00Z",
        "updated_at": "2026-03-04T00:00:00Z",
        "state_history": [],
        "metadata": {"workset_name": "Test Workset"},
    }
    created_retry = {
        **original,
        "workset_id": "ws-124",
        "state": "ready",
        "metadata": {**original.get("metadata", {}), "retried_from": "ws-123"},
    }

    def _get_workset(workset_id: str):
        if workset_id == "ws-123":
            return original
        if workset_id == "ws-124":
            return created_retry
        return original

    state_db.get_workset.side_effect = _get_workset
    state_db.register_workset.return_value = True
    state_db.update_state.return_value = None
    state_db.get_performance_metrics.return_value = {"is_final": True, "metrics": {}}

    customer_manager = MagicMock()
    customer_manager.get_customer_config.return_value = SimpleNamespace(
        customer_id="cust-001",
        customer_name="Acme",
        email="acme@example.com",
        s3_bucket="cust-bucket",
        max_concurrent_worksets=10,
        max_storage_gb=500,
        billing_account_id=None,
        cost_center=None,
    )

    settings = MagicMock()
    settings.get_effective_region.return_value = "us-west-2"
    settings.aws_profile = None

    integration = MagicMock()
    integration.register_workset.return_value = True

    deps = CustomerWorksetDependencies(
        state_db=state_db,
        settings=settings,
        customer_manager=customer_manager,
        integration=integration,
        manifest_registry=None,
        get_current_user=None,
    )
    app = FastAPI()
    app.include_router(create_customer_worksets_router(deps))

    with TestClient(app, base_url="https://testserver") as client:
        assert client.post("/api/v2/customers/cust-001/worksets/ws-123/cancel").status_code != 404
        assert client.post("/api/v2/customers/cust-001/worksets/ws-123/retry").status_code != 404
        assert client.get("/api/v2/customers/cust-001/worksets/ws-123/logs").status_code != 404
        assert client.get("/api/v2/customers/cust-001/worksets/ws-123/performance-metrics").status_code != 404

        # Exercise the early validation path (prevents SSH/headnode access).
        resp = client.get("/api/v2/customers/cust-001/worksets/ws-123/snakemake-log/evil.txt")
        assert resp.status_code == 400
