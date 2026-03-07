"""Focused tests for the WorksetMonitor cluster-create path."""

from __future__ import annotations

import subprocess
from pathlib import Path
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

    run_mock.assert_called_once_with(
        region_az="us-west-2b",
        aws_profile="lsmc",
        config_path="/tmp/daylily-ec.yaml",
        pass_on_warn=False,
        debug=False,
        contact_email="ops@example.com",
    )


def test_monitor_create_cluster_generates_config_when_template_missing(monkeypatch, tmp_path: Path) -> None:
    from daylib.workset_monitor import WorksetMonitor

    monitor = WorksetMonitor.__new__(WorksetMonitor)
    monitor.debug = True
    monitor.config = SimpleNamespace(
        aws=SimpleNamespace(profile="lsmc", region="us-west-2"),
        cluster=SimpleNamespace(
            template_path=None,
            preferred_availability_zone=None,
            contact_email="ops@example.com",
        ),
    )

    generated_path = tmp_path / "generated.yaml"
    write_mock = MagicMock(return_value=generated_path)
    proc = subprocess.CompletedProcess(args=["daylily-ec"], returncode=0, stdout="ok", stderr="")
    run_mock = MagicMock(return_value=proc)

    monkeypatch.setattr("daylib.ephemeral_cluster.runner.write_generated_ec_config", write_mock)
    monkeypatch.setattr("daylib.ephemeral_cluster.runner.run_create_sync", run_mock)

    cluster_name = monitor._create_cluster(
        {
            "cluster_name": "generated-cluster",
            "ssh_key_name": "kp-1",
            "s3_bucket_name": "omics-analysis-west",
        }
    )

    assert cluster_name == "generated-cluster"
    write_mock.assert_called_once()
    write_kwargs = write_mock.call_args.kwargs
    assert write_kwargs["cluster_name"] == "generated-cluster"
    assert write_kwargs["ssh_key_name"] == "kp-1"
    assert write_kwargs["s3_bucket_name"] == "omics-analysis-west"
    assert write_kwargs["contact_email"] == "ops@example.com"

    run_mock.assert_called_once_with(
        region_az="us-west-2a",
        aws_profile="lsmc",
        config_path=str(generated_path),
        pass_on_warn=False,
        debug=True,
        contact_email="ops@example.com",
    )