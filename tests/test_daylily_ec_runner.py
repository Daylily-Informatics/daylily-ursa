from __future__ import annotations

import subprocess

import pytest

from daylib_ursa.ephemeral_cluster import runner


def test_run_json_raises_when_daylily_ec_returns_empty_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "require_daylily_ec_version", lambda: "2.1.3")

    def fake_run(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["python", "-m", "daylily_ec.cli", "--json", "cluster", "list"],
            returncode=0,
            stdout="",
            stderr="AWS profile is required. Set AWS_PROFILE or use --profile.\n",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    client = runner.DaylilyEcClient()

    with pytest.raises(RuntimeError, match="AWS profile is required"):
        client.run_json(["cluster", "list", "--region", "us-west-2"])


def test_run_json_parses_object_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "require_daylily_ec_version", lambda: "2.1.3")
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs) -> subprocess.CompletedProcess[str]:  # noqa: ANN001
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args=["python", "-m", "daylily_ec.cli", "--json", "cluster", "list"],
            returncode=0,
            stdout='{"clusters":[],"regions":["us-west-2"]}\n',
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    client = runner.DaylilyEcClient()

    assert client.run_json(["cluster", "list", "--region", "us-west-2"]) == {
        "clusters": [],
        "regions": ["us-west-2"],
    }
    assert captured["command"][:4] == [
        runner.sys.executable,
        "-m",
        "daylily_ec.cli",
        "--json",
    ]


def test_stage_samples_builds_daylily_ec_213_argv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(runner, "require_daylily_ec_version", lambda: "2.1.3")

    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):  # noqa: ANN001
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    client = runner.DaylilyEcClient(aws_profile="profile-a")
    result = client.stage_samples(
        analysis_samples=tmp_path / "analysis_samples.tsv",
        reference_bucket="s3://bucket/ref",
        config_dir=tmp_path,
        region="us-west-2",
        stage_target="/data/staged_sample_data",
        debug=True,
    )

    assert result.returncode == 0
    assert captured["command"] == [
        runner.sys.executable,
        "-m",
        "daylily_ec.cli",
        "samples",
        "stage",
        str(tmp_path / "analysis_samples.tsv"),
        "--reference-bucket",
        "s3://bucket/ref",
        "--config-dir",
        str(tmp_path),
        "--region",
        "us-west-2",
        "--stage-target",
        "/data/staged_sample_data",
        "--profile",
        "profile-a",
        "--debug",
    ]
