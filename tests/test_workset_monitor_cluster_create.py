"""Focused tests for the WorksetMonitor cluster-create path.

These are unit tests (no AWS) and ensure the monitor uses the shared
daylily-ephemeral-cluster runner integration.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_monitor_create_cluster_uses_ephemeral_cluster_runner(monkeypatch) -> None:
    from daylib.workset_monitor import WorksetMonitor

    monitor = WorksetMonitor.__new__(WorksetMonitor)
    monitor.debug = False
    monitor.config = SimpleNamespace(
        aws=SimpleNamespace(profile="lsmc", region="us-west-2"),
        cluster=SimpleNamespace(
            template_path="/tmp/daylily-ec.yaml",
            preferred_availability_zone="us-west-2b",
            contact_email="ops@example.com",
        ),
    )

    proc = subprocess.CompletedProcess(args=["daylily-ec"], returncode=0, stdout="ok", stderr="")
    run_mock = MagicMock(return_value=proc)
    monkeypatch.setattr("daylib.ephemeral_cluster.runner.run_create_sync", run_mock)

    cluster_name = monitor._create_cluster({"cluster_name": "test-cluster"})
    assert cluster_name == "test-cluster"

    run_mock.assert_called_once()
    kwargs = run_mock.call_args.kwargs
    assert kwargs["region_az"] == "us-west-2b"
    assert kwargs["aws_profile"] == "lsmc"
    assert kwargs["config_path"] == "/tmp/daylily-ec.yaml"
    assert kwargs["pass_on_warn"] is False
    assert kwargs["debug"] is False
    assert kwargs["contact_email"] == "ops@example.com"
