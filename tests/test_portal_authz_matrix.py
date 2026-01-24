"""Portal authz matrix tests.

This suite enumerates portal-sensitive endpoints and asserts behavior for:
- unauthenticated
- authenticated non-admin
- authenticated admin

We keep this compact and deterministic by stubbing ~/.ursa (Path.home) and
cluster service dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from daylib.workset_api import create_app
from daylib.workset_state_db import WorksetStateDB


def _make_authenticated_client(mock_state_db: MagicMock, *, customer_id: str, is_admin: bool) -> TestClient:
  mock_customer_manager = MagicMock()
  mock_customer = MagicMock()
  mock_customer.customer_id = customer_id
  mock_customer.is_admin = is_admin
  mock_customer_manager.get_customer_by_email.return_value = mock_customer

  app = create_app(
    state_db=mock_state_db,
    enable_auth=False,
    customer_manager=mock_customer_manager,
  )
  client = TestClient(app)
  client.post("/portal/login", data={"email": "user@example.com", "password": "testpass"})
  return client


@dataclass(frozen=True)
class _AuthzCase:
  name: str
  path: str
  unauth_status: int
  non_admin_status: int
  admin_status: int
  assert_non_admin: Optional[Callable[[object], None]] = None
  assert_admin: Optional[Callable[[object], None]] = None


def _assert_redirect_to_login(response) -> None:
  assert response.status_code == 302
  assert "/portal/login" in response.headers.get("location", "")


def _assert_admin_required_403(response) -> None:
  assert response.status_code == 403
  assert response.json()["detail"] == "Admin access required"


def _assert_clusters_non_admin(response) -> None:
  assert b"AWS Budget" not in response.content
  assert b"Slurm Job Queue" not in response.content
  assert b"Public IP" not in response.content


def _assert_clusters_admin(response) -> None:
  assert b"AWS Budget" in response.content
  assert b"Slurm Job Queue" in response.content
  assert b"Public IP" in response.content


@pytest.fixture(autouse=True)
def _stable_home_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  # Portal monitor endpoints consult ~/.ursa for pid/logs.
  monkeypatch.setattr(Path, "home", lambda: tmp_path)


@pytest.fixture(autouse=True)
def _stub_cluster_services(monkeypatch: pytest.MonkeyPatch) -> None:
  class _FakeUrsaConfig:
    is_configured = True
    aws_profile = None

    def get_allowed_regions(self):
      return ["us-west-2"]

  monkeypatch.setattr("daylib.ursa_config.get_ursa_config", lambda: _FakeUrsaConfig())

  class _FakeCluster:
    def to_dict(self, *, include_sensitive: bool = True):
      base = {
        "cluster_name": "test-cluster",
        "region": "us-west-2",
        "cluster_status": "CREATE_COMPLETE",
        "compute_fleet_status": "RUNNING",
        "head_node": {
          "instance_type": "t3.medium",
          "public_ip": "1.2.3.4",
          "private_ip": "10.0.0.1",
          "state": "running",
        },
        "budget_info": None,
        "job_queue": None,
      }
      if include_sensitive:
        base["budget_info"] = {"total_budget": 100.0, "used_budget": 1.0, "percent_used": 1.0}
        base["job_queue"] = {"total_jobs": 1, "running_jobs": 1, "pending_jobs": 0, "configuring_jobs": 0, "total_cpus": 4, "jobs": []}
      return base

  service = MagicMock()
  service.get_all_clusters_with_status.return_value = [_FakeCluster()]
  monkeypatch.setattr("daylib.cluster_service.get_cluster_service", lambda **kwargs: service)


@pytest.fixture
def mock_state_db() -> MagicMock:
  mock_db = MagicMock(spec=WorksetStateDB)
  mock_db.list_worksets_by_state.return_value = []

  worksets = {
    "ws-owned-by-cust-user": {
      "workset_id": "ws-owned-by-cust-user",
      "state": "ready",
      "priority": "normal",
      "bucket": "test-bucket",
      "prefix": "worksets/test/",
      "customer_id": "cust-user",
      "created_at": "2026-01-24T00:00:00Z",
      "updated_at": "2026-01-24T00:00:00Z",
    },
    "ws-owned-by-cust-admin": {
      "workset_id": "ws-owned-by-cust-admin",
      "state": "ready",
      "priority": "normal",
      "bucket": "test-bucket",
      "prefix": "worksets/test/",
      "customer_id": "cust-admin",
      "created_at": "2026-01-24T00:00:00Z",
      "updated_at": "2026-01-24T00:00:00Z",
    },
    "ws-owned-by-cust-other": {
      "workset_id": "ws-owned-by-cust-other",
      "state": "ready",
      "priority": "normal",
      "bucket": "test-bucket",
      "prefix": "worksets/test/",
      "customer_id": "cust-other",
      "created_at": "2026-01-24T00:00:00Z",
      "updated_at": "2026-01-24T00:00:00Z",
    },
  }

  def _get_workset(workset_id: str):
    return worksets.get(workset_id)

  mock_db.get_workset.side_effect = _get_workset
  return mock_db


@pytest.fixture
def unauthenticated_client(mock_state_db: MagicMock) -> TestClient:
  app = create_app(state_db=mock_state_db, enable_auth=False)
  return TestClient(app)


@pytest.fixture
def authenticated_non_admin_client(mock_state_db: MagicMock) -> TestClient:
  return _make_authenticated_client(mock_state_db, customer_id="cust-user", is_admin=False)


@pytest.fixture
def authenticated_admin_client(mock_state_db: MagicMock) -> TestClient:
  return _make_authenticated_client(mock_state_db, customer_id="cust-admin", is_admin=True)


_CASES = [
  _AuthzCase("portal_dashboard", "/portal", 302, 200, 200),
  _AuthzCase("worksets_list", "/portal/worksets", 302, 200, 200),
  _AuthzCase("clusters", "/portal/clusters", 302, 200, 200, assert_non_admin=_assert_clusters_non_admin, assert_admin=_assert_clusters_admin),
  _AuthzCase("monitor_page", "/portal/monitor", 302, 403, 200),
  _AuthzCase("monitor_status", "/api/monitor/status", 302, 403, 200),
  _AuthzCase("monitor_logs", "/api/monitor/logs", 302, 403, 200),
  _AuthzCase("workset_detail_owned_user", "/portal/worksets/ws-owned-by-cust-user", 302, 200, 404),
  _AuthzCase("workset_detail_owned_admin", "/portal/worksets/ws-owned-by-cust-admin", 302, 404, 200),
  _AuthzCase("workset_detail_not_owned", "/portal/worksets/ws-owned-by-cust-other", 302, 404, 404),
]


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_portal_authz_matrix(
  case: _AuthzCase,
  unauthenticated_client: TestClient,
  authenticated_non_admin_client: TestClient,
  authenticated_admin_client: TestClient,
):
  unauth = unauthenticated_client.get(case.path, follow_redirects=False)
  assert unauth.status_code == case.unauth_status
  if unauth.status_code == 302:
    _assert_redirect_to_login(unauth)

  non_admin = authenticated_non_admin_client.get(case.path, follow_redirects=False)
  assert non_admin.status_code == case.non_admin_status
  if non_admin.status_code == 403:
    _assert_admin_required_403(non_admin)
  if case.assert_non_admin and non_admin.status_code == 200:
    case.assert_non_admin(non_admin)

  admin = authenticated_admin_client.get(case.path, follow_redirects=False)
  assert admin.status_code == case.admin_status
  if case.assert_admin and admin.status_code == 200:
    case.assert_admin(admin)

  if case.path == "/api/monitor/status" and admin.status_code == 200:
    data = admin.json()
    assert "running" in data
    assert "pid" in data
    assert "stats" in data

  if case.path == "/api/monitor/logs" and admin.status_code == 200:
    data = admin.json()
    assert "lines" in data
    assert "error" in data
    assert "log_file" in data

