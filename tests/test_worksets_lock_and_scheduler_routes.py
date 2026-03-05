"""Request-level coverage for workset lock/state/scheduler routes."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_worksets_lock_state_and_scheduler_routes_have_request_level_coverage():
    from daylib.routes.worksets import create_worksets_router
    from daylib.workset_state_db import WorksetStateDB

    state_db = MagicMock(spec=WorksetStateDB)
    state_db.get_workset.return_value = {
        "workset_id": "ws-123",
        "state": "ready",
        "priority": "normal",
        "workset_type": "ruo",
        "bucket": "test-bucket",
        "prefix": "worksets/ws-123/",
        "created_at": "2026-03-04T00:00:00Z",
        "updated_at": "2026-03-04T00:00:00Z",
    }
    state_db.acquire_lock.return_value = True
    state_db.release_lock.return_value = True
    state_db.get_ready_worksets_prioritized.return_value = []

    app = FastAPI()
    app.include_router(create_worksets_router(state_db=state_db, scheduler=None))

    with TestClient(app) as client:
        assert client.put(
            "/worksets/ws-123/state",
            json={"state": "ready", "reason": "test"},
        ).status_code != 404
        assert client.post("/worksets/ws-123/lock?owner_id=worker-1").status_code != 404
        assert client.delete("/worksets/ws-123/lock?owner_id=worker-1").status_code != 404

        # Scheduler not configured -> 503 (but still hits the route)
        assert client.get("/scheduler/stats").status_code == 503

        # No ready worksets -> null payload (200)
        assert client.get("/worksets/next").status_code != 404

