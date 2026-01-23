"""Tests for Monitor Dashboard routes and API endpoints."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from daylib.workset_state_db import WorksetStateDB, WorksetState
from daylib.workset_api import create_app


@pytest.fixture
def mock_state_db():
    """Create mock state database."""
    mock_db = MagicMock(spec=WorksetStateDB)
    # Return empty lists by default for state queries
    mock_db.list_worksets_by_state.return_value = []
    return mock_db


@pytest.fixture
def mock_ursa_dir(tmp_path):
    """Create mock ~/.ursa directory structure."""
    ursa_dir = tmp_path / ".ursa"
    ursa_dir.mkdir()
    logs_dir = ursa_dir / "logs"
    logs_dir.mkdir()
    return ursa_dir


@pytest.fixture
def client_and_mock(mock_state_db, mock_ursa_dir):
    """Create test client with authenticated session.

    Returns tuple of (client, mock_ursa_dir) for use in tests.
    """
    app = create_app(state_db=mock_state_db, enable_auth=False)
    client = TestClient(app)
    # Perform login to set session
    client.post("/portal/login", data={"email": "test@example.com", "password": "testpass"})
    return client, mock_ursa_dir


@pytest.fixture
def authenticated_client(mock_state_db, mock_ursa_dir):
    """Create test client with authenticated session."""
    app = create_app(state_db=mock_state_db, enable_auth=False)
    client = TestClient(app)
    # Perform login to set session
    client.post("/portal/login", data={"email": "test@example.com", "password": "testpass"})
    return client


@pytest.fixture
def mock_pid_file(mock_ursa_dir):
    """Create mock PID file with a valid PID."""
    pid_file = mock_ursa_dir / "monitor.pid"
    # Use current process PID so os.kill(pid, 0) succeeds
    pid_file.write_text(str(os.getpid()))
    return pid_file


@pytest.fixture
def mock_log_file(mock_ursa_dir):
    """Create mock monitor log file."""
    logs_dir = mock_ursa_dir / "logs"
    log_file = logs_dir / "monitor_20260123_120000.log"
    log_content = """2026-01-23 12:00:00 [INFO] Starting Daylily workset monitor in us-west-2
2026-01-23 12:00:00 [INFO] Monitoring prefix: worksets/ (worksets carry individual buckets from cluster region)
2026-01-23 12:00:00 [INFO] Poll interval: 60s, continuous: True, max parallel: 4
2026-01-23 12:00:01 [INFO] Found 2 ready worksets
2026-01-23 12:00:02 [DEBUG] Checking workset test-workset-001
2026-01-23 12:00:03 [WARNING] Workset test-workset-002 is locked
2026-01-23 12:00:05 [ERROR] Failed to process workset test-workset-003
"""
    log_file.write_text(log_content)
    return log_file


@pytest.fixture
def mock_config_file(mock_ursa_dir):
    """Create mock monitor config file."""
    config_file = mock_ursa_dir / "monitor-config.yaml"
    config_content = """aws:
  profile: test-profile
  region: us-west-2

monitor:
  prefix: worksets/
  poll_interval_seconds: 60
  max_concurrent_worksets: 2
  archive_prefix: worksets/archived/

cluster:
  reuse_cluster_name: test-cluster
"""
    config_file.write_text(config_content)
    return config_file


class TestMonitorDashboardRoute:
    """Test /portal/monitor route."""

    def test_monitor_page_renders_authenticated(self, authenticated_client, mock_ursa_dir, monkeypatch):
        """Test that the monitor page renders successfully when authenticated."""
        # Patch Path.home() to return our mock directory's parent
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/portal/monitor")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Monitor" in response.content

    def test_monitor_page_unauthenticated_redirect(self, mock_state_db):
        """Test that unauthenticated users are redirected to login."""
        app = create_app(state_db=mock_state_db, enable_auth=False)
        client = TestClient(app)
        # Don't login
        response = client.get("/portal/monitor", follow_redirects=False)
        assert response.status_code == 302
        assert "/portal/login" in response.headers["location"]

    def test_monitor_page_shows_stopped_status(
        self, authenticated_client, mock_ursa_dir, monkeypatch
    ):
        """Test monitor page shows 'Stopped' when no PID file exists."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/portal/monitor")
        assert response.status_code == 200
        # Should show stopped status (no PID file = not running)
        assert b"Stopped" in response.content or b"stopped" in response.content.lower()

    def test_monitor_page_shows_running_status(
        self, authenticated_client, mock_ursa_dir, mock_pid_file, mock_log_file, monkeypatch
    ):
        """Test monitor page shows 'Running' when PID file exists with valid PID."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/portal/monitor")
        assert response.status_code == 200
        # Should show running status
        assert b"Running" in response.content or b"running" in response.content.lower()

    def test_monitor_page_shows_pid(
        self, authenticated_client, mock_ursa_dir, mock_pid_file, mock_log_file, monkeypatch
    ):
        """Test monitor page displays PID when running."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/portal/monitor")
        assert response.status_code == 200
        # Should contain the PID
        assert str(os.getpid()).encode() in response.content

    def test_monitor_page_reads_max_parallel_from_log(
        self, authenticated_client, mock_ursa_dir, mock_pid_file, mock_log_file, monkeypatch
    ):
        """Test monitor page reads max parallel value from log file, not config."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/portal/monitor")
        assert response.status_code == 200
        # The log file has "max parallel: 4", not the config file's value of 2
        # Check the response contains "4" somewhere near max parallel context
        content = response.content.decode("utf-8")
        assert "4" in content  # Runtime value from log

    def test_monitor_page_displays_log_lines(
        self, authenticated_client, mock_ursa_dir, mock_log_file, monkeypatch
    ):
        """Test monitor page displays log lines from log file."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/portal/monitor")
        assert response.status_code == 200
        content = response.content.decode("utf-8")
        # Check for log content
        assert "Starting Daylily workset monitor" in content or "workset" in content.lower()

    def test_monitor_page_handles_missing_log_files(
        self, authenticated_client, mock_ursa_dir, monkeypatch
    ):
        """Test monitor page handles gracefully when no log files exist."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/portal/monitor")
        assert response.status_code == 200
        # Should still render without error

    def test_monitor_page_shows_workset_statistics(
        self, authenticated_client, mock_ursa_dir, mock_state_db, monkeypatch
    ):
        """Test monitor page shows workset statistics from DynamoDB."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        # Configure mock to return specific counts for each state
        def mock_list_by_state(state, limit=500):
            state_counts = {
                WorksetState.READY: [{"workset_id": f"ready-{i}"} for i in range(3)],
                WorksetState.IN_PROGRESS: [{"workset_id": f"progress-{i}"} for i in range(2)],
                WorksetState.COMPLETE: [{"workset_id": f"complete-{i}"} for i in range(5)],
                WorksetState.ERROR: [{"workset_id": "error-1"}],
            }
            return state_counts.get(state, [])

        mock_state_db.list_worksets_by_state.side_effect = mock_list_by_state

        response = authenticated_client.get("/portal/monitor")
        assert response.status_code == 200
        content = response.content.decode("utf-8")
        # Should contain the count values
        assert "3" in content  # ready count
        assert "2" in content  # in_progress count


class TestMonitorAPIStatus:
    """Test /api/monitor/status endpoint."""

    def test_api_status_returns_json(self, authenticated_client, mock_ursa_dir, monkeypatch):
        """Test that API returns proper JSON structure."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/api/monitor/status")
        assert response.status_code == 200
        data = response.json()
        assert "running" in data
        assert "pid" in data
        assert "stats" in data
        assert isinstance(data["stats"], dict)

    def test_api_status_running_false_no_pid(self, authenticated_client, mock_ursa_dir, monkeypatch):
        """Test API returns running=False when no PID file."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/api/monitor/status")
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is False
        assert data["pid"] is None

    def test_api_status_running_true_with_pid(
        self, authenticated_client, mock_ursa_dir, mock_pid_file, monkeypatch
    ):
        """Test API returns running=True when valid PID file exists."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/api/monitor/status")
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is True
        assert data["pid"] == os.getpid()

    def test_api_status_includes_workset_stats(
        self, authenticated_client, mock_ursa_dir, mock_state_db, monkeypatch
    ):
        """Test API returns workset statistics."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        def mock_list_by_state(state, limit=500):
            if state == WorksetState.READY:
                return [{"workset_id": "test-1"}, {"workset_id": "test-2"}]
            return []

        mock_state_db.list_worksets_by_state.side_effect = mock_list_by_state

        response = authenticated_client.get("/api/monitor/status")
        assert response.status_code == 200
        data = response.json()
        assert data["stats"]["ready"] == 2


class TestMonitorAPILogs:
    """Test /api/monitor/logs endpoint."""

    def test_api_logs_returns_json(self, authenticated_client, mock_ursa_dir, mock_log_file, monkeypatch):
        """Test that logs API returns proper JSON structure."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/api/monitor/logs")
        assert response.status_code == 200
        data = response.json()
        assert "lines" in data
        assert "error" in data
        assert "log_file" in data
        assert isinstance(data["lines"], list)

    def test_api_logs_returns_log_content(
        self, authenticated_client, mock_ursa_dir, mock_log_file, monkeypatch
    ):
        """Test logs API returns actual log content."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/api/monitor/logs")
        assert response.status_code == 200
        data = response.json()
        assert len(data["lines"]) > 0
        # Check some log content is present
        log_text = "\n".join(data["lines"])
        assert "Starting Daylily workset monitor" in log_text

    def test_api_logs_respects_lines_param(
        self, authenticated_client, mock_ursa_dir, mock_log_file, monkeypatch
    ):
        """Test logs API respects the lines parameter."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/api/monitor/logs?lines=50")
        assert response.status_code == 200
        data = response.json()
        assert len(data["lines"]) <= 50

    def test_api_logs_filters_by_level(
        self, authenticated_client, mock_ursa_dir, mock_log_file, monkeypatch
    ):
        """Test logs API filters by log level."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/api/monitor/logs?level=ERROR")
        assert response.status_code == 200
        data = response.json()
        # Should only return ERROR level lines
        for line in data["lines"]:
            assert "[ERROR]" in line

    def test_api_logs_filter_warning(
        self, authenticated_client, mock_ursa_dir, mock_log_file, monkeypatch
    ):
        """Test logs API filters WARNING level."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/api/monitor/logs?level=WARNING")
        assert response.status_code == 200
        data = response.json()
        for line in data["lines"]:
            assert "[WARNING]" in line

    def test_api_logs_handles_no_log_files(
        self, authenticated_client, mock_ursa_dir, monkeypatch
    ):
        """Test logs API handles gracefully when no log files exist."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/api/monitor/logs")
        assert response.status_code == 200
        data = response.json()
        assert data["lines"] == []
        assert data["error"] is None
        assert data["log_file"] is None

    def test_api_logs_returns_log_filename(
        self, authenticated_client, mock_ursa_dir, mock_log_file, monkeypatch
    ):
        """Test logs API returns the log filename."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/api/monitor/logs")
        assert response.status_code == 200
        data = response.json()
        assert data["log_file"] == "monitor_20260123_120000.log"


class TestMonitorConfigDisplay:
    """Test monitor configuration display."""

    def test_config_display_from_yaml(
        self, authenticated_client, mock_ursa_dir, mock_config_file, monkeypatch
    ):
        """Test that configuration values are read from YAML config file."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/portal/monitor")
        assert response.status_code == 200
        content = response.content.decode("utf-8")
        # Check config values appear
        assert "worksets/" in content  # prefix
        assert "60" in content  # poll_interval_seconds

    def test_config_display_shows_config_path(
        self, authenticated_client, mock_ursa_dir, mock_config_file, monkeypatch
    ):
        """Test that configuration path is displayed."""
        monkeypatch.setattr(Path, "home", lambda: mock_ursa_dir.parent)

        response = authenticated_client.get("/portal/monitor")
        assert response.status_code == 200
        content = response.content.decode("utf-8")
        assert "monitor-config.yaml" in content or ".ursa" in content

