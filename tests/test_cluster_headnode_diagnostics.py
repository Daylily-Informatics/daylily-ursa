from __future__ import annotations

import time
from types import SimpleNamespace

from daylily_ec.aws import ssm as ssm_module

from daylib_ursa.cluster_service import ClusterService
from daylib_ursa.ephemeral_cluster.runner import REQUIRED_DAYLILY_EC_VERSION


class FakeDaylilyEcClient:
    def __init__(self, *, instance_id: str | None = "i-0123456789abcdef0") -> None:
        self.instance_id = instance_id

    def cluster_describe(self, *, cluster_name: str, region: str):
        head_node = {
            "instanceType": "c7i.large",
            "state": "running",
        }
        if self.instance_id:
            head_node["instanceId"] = self.instance_id
        return {
            "clusterName": cluster_name,
            "clusterStatus": "CREATE_COMPLETE",
            "computeFleetStatus": "RUNNING",
            "headNode": head_node,
            "scheduler": {"type": "slurm"},
        }

    def cluster_list(self, *, region: str, details: bool = True):
        return {
            "clusters": [
                {
                    "name": "cluster-1",
                    "status": "CREATE_COMPLETE",
                    "details": self.cluster_describe(cluster_name="cluster-1", region=region)
                    if details
                    else {},
                }
            ]
        }


def _service(*, instance_id: str | None = "i-0123456789abcdef0") -> ClusterService:
    return ClusterService(
        regions=["us-west-2"],
        aws_profile="lsmc",
        client=FakeDaylilyEcClient(instance_id=instance_id),
    )


def test_cluster_payload_includes_pinned_version_console_url_and_cached_probe() -> None:
    service = _service()
    cluster = service.describe_cluster("cluster-1", "us-west-2")

    payload = cluster.to_dict()

    assert payload["daylily_ec_pinned_version"] == REQUIRED_DAYLILY_EC_VERSION
    assert payload["aws_console_url"] == (
        "https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2"
        "#InstanceDetails:instanceId=i-0123456789abcdef0"
    )


def test_static_probe_uses_ssm_and_caches_until_ttl(monkeypatch) -> None:
    calls: list[str] = []

    def fake_wait_for_ssm_online(*_args, **_kwargs) -> None:
        return None

    def fake_run_shell(_instance_id, _region, script, **_kwargs):
        calls.append(script)
        return SimpleNamespace(
            stdout=(
                "__DAYLILY_EC_VERSION_BEGIN__\n"
                '{"app": "daylily-ec", "version": "2.1.11", "git_hash": "abc123"}\n'
                "__DAYLILY_EC_VERSION_END__\n"
                "__DAY_CLONE_HELP_BEGIN__\n"
                "Usage: day-clone [OPTIONS]\n"
                "__DAY_CLONE_HELP_END__\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(ssm_module, "wait_for_ssm_online", fake_wait_for_ssm_online)
    monkeypatch.setattr(ssm_module, "run_shell", fake_run_shell)
    service = _service()
    service.get_all_clusters()

    first = service.fetch_headnode_static_probe(cluster_name="cluster-1", region="us-west-2")
    second = service.fetch_headnode_static_probe(cluster_name="cluster-1", region="us-west-2")
    listed = service.get_all_clusters()

    assert len(calls) == 1
    assert first["cached"] is False
    assert second["cached"] is True
    assert listed[0].headnode_probes["static"]["cached"] is True
    assert second["data"]["remote_daylily_ec_version"] == "2.1.11"
    assert second["data"]["remote_git_hash"] == "abc123"
    assert second["data"]["day_clone_available"] is True
    assert "day-clone --help" in calls[0]

    key = ("static", "us-west-2", "cluster-1")
    _expires_at, result = service._probe_cache[key]
    service._probe_cache[key] = (time.time() - 1, result)
    refreshed = service.fetch_headnode_static_probe(
        cluster_name="cluster-1",
        region="us-west-2",
    )

    assert len(calls) == 2
    assert refreshed["cached"] is False


def test_scheduler_and_fsx_probes_return_raw_outputs(monkeypatch) -> None:
    scripts: list[str] = []

    def fake_wait_for_ssm_online(*_args, **_kwargs) -> None:
        return None

    def fake_run_shell(_instance_id, _region, script, **_kwargs):
        scripts.append(script)
        if "squeue" in script:
            stdout = (
                "__SQUEUE_BEGIN__\n"
                "JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)\n"
                "42 compute test ubuntu R 0:12 1 compute-1\n"
                "__SQUEUE_END__\n"
                "__SINFO_BEGIN__\n"
                "PARTITION AVAIL TIMELIMIT NODES STATE NODELIST\n"
                "compute* up infinite 1 idle compute-1\n"
                "__SINFO_END__\n"
            )
        else:
            stdout = (
                "__DF_FSX_BEGIN__\n"
                "Filesystem Size Used Avail Use% Mounted on\n"
                "fsx 1.2T 200G 1.0T 17% /fsx\n"
                "__DF_FSX_END__\n"
            )
        return SimpleNamespace(stdout=stdout, stderr="")

    monkeypatch.setattr(ssm_module, "wait_for_ssm_online", fake_wait_for_ssm_online)
    monkeypatch.setattr(ssm_module, "run_shell", fake_run_shell)
    service = _service()

    scheduler = service.fetch_headnode_scheduler_probe(
        cluster_name="cluster-1",
        region="us-west-2",
    )
    fsx = service.fetch_headnode_fsx_probe(cluster_name="cluster-1", region="us-west-2")

    assert "JOBID PARTITION" in scheduler["data"]["squeue_output"]
    assert "compute* up" in scheduler["data"]["sinfo_output"]
    assert "fsx 1.2T" in fsx["data"]["df_output"]
    assert any("sinfo" in script for script in scripts)
    assert any("df -h /fsx" in script for script in scripts)


def test_missing_instance_id_returns_probe_error(monkeypatch) -> None:
    called = False

    def fake_run_shell(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("run_shell should not run without an instance id")

    monkeypatch.setattr(ssm_module, "run_shell", fake_run_shell)
    service = _service(instance_id=None)

    result = service.fetch_headnode_fsx_probe(cluster_name="cluster-1", region="us-west-2")

    assert called is False
    assert result["error"] == "Headnode EC2 instance id is unavailable for cluster-1"
    assert result["instance_id"] is None


def test_nonzero_ssm_command_is_reported_inline(monkeypatch) -> None:
    def fake_wait_for_ssm_online(*_args, **_kwargs) -> None:
        return None

    def fake_run_shell(*_args, **_kwargs):
        command_result = ssm_module.SsmCommandResult(
            command_id="cmd-1",
            instance_id="i-0123456789abcdef0",
            status="Failed",
            response_code=127,
            stdout="",
            stderr="squeue: command not found",
        )
        raise ssm_module.SsmCommandFailedError("command failed", command_result)

    monkeypatch.setattr(ssm_module, "wait_for_ssm_online", fake_wait_for_ssm_online)
    monkeypatch.setattr(ssm_module, "run_shell", fake_run_shell)
    service = _service()

    result = service.fetch_headnode_scheduler_probe(
        cluster_name="cluster-1",
        region="us-west-2",
    )

    assert result["error"] == "squeue: command not found"
    assert result["data"]["response_code"] == 127
    assert result["data"]["status"] == "Failed"
