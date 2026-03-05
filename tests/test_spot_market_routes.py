from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _write_json(path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_spot_market_endpoints_require_admin(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    from daylily_ursa.routes.spot_market import SpotMarketDependencies, create_spot_market_router

    settings = MagicMock()
    settings.aws_profile = None
    settings.get_allowed_regions.return_value = ["us-west-2", "us-east-1", "eu-central-1"]

    def get_current_user():
        return {"is_admin": False}

    app = FastAPI()
    app.include_router(create_spot_market_router(SpotMarketDependencies(settings=settings, get_current_user=get_current_user)))

    with patch(
        "daylily_ursa.ursa_config.get_ursa_config",
        return_value=SimpleNamespace(is_configured=False, aws_profile=None),
    ):
        with TestClient(app, base_url="https://testserver") as client:
            assert client.get("/api/spot-market/status").status_code == 403


def test_spot_market_endpoints_have_request_level_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    from daylily_ursa.routes.spot_market import SpotMarketDependencies, create_spot_market_router

    settings = MagicMock()
    settings.aws_profile = None
    settings.get_allowed_regions.return_value = ["us-west-2", "us-east-1", "eu-central-1"]

    def get_current_user():
        return {"is_admin": True, "user_email": "admin@example.com"}

    app = FastAPI()
    app.include_router(create_spot_market_router(SpotMarketDependencies(settings=settings, get_current_user=get_current_user)))

    base = tmp_path / ".ursa" / "spot-market"

    # Seed one job + log
    job_id = "sm_test_0001"
    log_path = base / "logs" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("hello\n", encoding="utf-8")
    _write_json(
        base / "jobs" / f"{job_id}.json",
        {"job_id": job_id, "status": "completed", "log_path": str(log_path), "region": "us-west-2"},
    )

    # Seed one snapshot
    _write_json(
        base / "snapshots" / "us-west-2" / "20260305000000.json",
        {
            "timestamp": "2026-03-05T00:00:00Z",
            "region": "us-west-2",
            "best_zone": {"zone": "us-west-2a", "est_cost": 1.23, "instances": 4},
        },
    )

    with patch(
        "daylily_ursa.ursa_config.get_ursa_config",
        return_value=SimpleNamespace(is_configured=False, aws_profile=None),
    ):
        with TestClient(app, base_url="https://testserver") as client:
            assert client.get("/api/spot-market/status").status_code == 200
            assert (
                client.post(
                    "/api/spot-market/config",
                    json={"regions": ["us-west-2", "us-east-1"], "interval": "6h"},
                ).status_code
                == 200
            )

            assert client.get("/api/spot-market/jobs?limit=20").status_code == 200
            assert client.get(f"/api/spot-market/jobs/{job_id}").status_code == 200
            assert client.get(f"/api/spot-market/jobs/{job_id}/logs?lines=10").status_code == 200

            assert (
                client.get("/api/spot-market/snapshots?region=us-west-2&limit=50").status_code
                == 200
            )


def test_spot_market_poll_starts_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    from daylily_ursa.routes.spot_market import SpotMarketDependencies, create_spot_market_router

    settings = MagicMock()
    settings.aws_profile = "lsmc"
    settings.get_allowed_regions.return_value = ["us-west-2", "us-east-1", "eu-central-1"]

    def get_current_user():
        return {"is_admin": True, "user_email": "admin@example.com"}

    app = FastAPI()
    app.include_router(create_spot_market_router(SpotMarketDependencies(settings=settings, get_current_user=get_current_user)))

    with patch(
        "daylily_ursa.ursa_config.get_ursa_config",
        return_value=SimpleNamespace(is_configured=False, aws_profile="lsmc"),
    ):
        with patch(
            "daylily_ursa.spot_market.runner.start_poll_job",
            side_effect=[
                {"job_id": "sm_test_a", "region": "us-west-2"},
                {"job_id": "sm_test_b", "region": "us-east-1"},
            ],
        ):
            with TestClient(app, base_url="https://testserver") as client:
                resp = client.post(
                    "/api/spot-market/poll",
                    json={"regions": ["us-west-2", "us-east-1"], "mode": "now"},
                )
                assert resp.status_code == 200
                payload = resp.json()
                assert payload["job_ids"] == ["sm_test_a", "sm_test_b"]


def test_spot_market_poll_blocks_when_checker_running(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    from daylily_ursa.routes.spot_market import SpotMarketDependencies, create_spot_market_router

    settings = MagicMock()
    settings.aws_profile = "lsmc"
    settings.get_allowed_regions.return_value = ["us-west-2", "us-east-1", "eu-central-1"]

    def get_current_user():
        return {"is_admin": True, "user_email": "admin@example.com"}

    app = FastAPI()
    app.include_router(create_spot_market_router(SpotMarketDependencies(settings=settings, get_current_user=get_current_user)))

    with patch(
        "daylily_ursa.ursa_config.get_ursa_config",
        return_value=SimpleNamespace(is_configured=False, aws_profile="lsmc"),
    ):
        with patch(
            "daylily_ursa.spot_market.runner.list_running_jobs",
            return_value=[{"job_id": "sm_running_001", "status": "running", "region": "us-west-2"}],
        ):
            with patch("daylily_ursa.spot_market.runner.start_poll_job") as start_poll_job:
                with TestClient(app, base_url="https://testserver") as client:
                    resp = client.post(
                        "/api/spot-market/poll",
                        json={"regions": ["us-west-2"], "mode": "now"},
                    )
                assert resp.status_code == 409
                detail = resp.json().get("detail", "")
                assert "already running" in detail.lower()
                start_poll_job.assert_not_called()
