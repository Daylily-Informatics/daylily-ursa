from __future__ import annotations

import json

from cli_core_yo.runtime_checks import evaluate_prereq
from typer.testing import CliRunner

from daylib_ursa.cli import app, spec

runner = CliRunner()


def _runtime_prereq(key: str):
    assert spec.runtime is not None
    for prereq in spec.runtime.prereqs:
        if prereq.key == key:
            return prereq
    raise AssertionError(f"missing prereq {key}")


def test_cli_spec_uses_platform_v2_runtime() -> None:
    assert spec.policy.profile == "platform-v2"
    assert spec.runtime is not None
    assert spec.runtime.default_backend == "ursa-conda"
    assert spec.runtime.allow_skip_check is False
    assert {prereq.key for prereq in spec.runtime.prereqs} == {
        "ursa-conda-active-env",
        "ursa-conda-env-name",
        "ursa-daylily-tapdb",
        "ursa-daylily-auth-cognito",
    }


def test_cli_runtime_requires_active_conda_env() -> None:
    result = evaluate_prereq(
        _runtime_prereq("ursa-conda-active-env"),
        env={"CONDA_DEFAULT_ENV": ""},
    )

    assert result.status == "fail"
    assert "active deployment-scoped conda environment" in result.summary


def test_cli_runtime_requires_hyphenated_conda_env_name() -> None:
    result = evaluate_prereq(
        _runtime_prereq("ursa-conda-env-name"),
        env={"CONDA_DEFAULT_ENV": "URSA"},
    )

    assert result.status == "fail"
    assert "deployment-scoped conda environment name with '-'" in result.summary


def test_cli_registry_exposes_v2_command_tree_and_policies() -> None:
    registry = app._cli_core_yo_registry

    assert registry.resolve_command_args(["version"]) is not None
    assert registry.resolve_command_args(["server", "status"]) is not None
    assert registry.resolve_command_args(["db", "reset"]) is not None
    assert registry.resolve_command_args(["test", "run"]) is not None
    assert registry.resolve_command_args(["quality", "check"]) is not None
    assert registry.resolve_command_args(["monitor", "start"]) is not None
    assert registry.resolve_command_args(["integrations", "dewey", "get-artifact"]) is not None

    version_cmd = registry.get_command(("version",))
    server_status_cmd = registry.get_command(("server", "status"))
    db_reset_cmd = registry.get_command(("db", "reset"))
    env_validate_cmd = registry.get_command(("env", "validate"))
    monitor_start_cmd = registry.get_command(("monitor", "start"))
    import_artifact_cmd = registry.get_command(("integrations", "dewey", "import-artifact"))

    assert version_cmd is not None
    assert version_cmd.policy.runtime_guard == "exempt"

    assert server_status_cmd is not None
    assert server_status_cmd.policy.prereq_tags == {"ursa-runtime"}

    assert db_reset_cmd is not None
    assert db_reset_cmd.policy.mutates_state is True
    assert db_reset_cmd.policy.interactive is True

    assert env_validate_cmd is not None
    assert env_validate_cmd.policy.runtime_guard == "exempt"

    assert monitor_start_cmd is not None
    assert monitor_start_cmd.policy.long_running is True
    assert monitor_start_cmd.policy.mutates_state is True

    assert import_artifact_cmd is not None
    assert import_artifact_cmd.policy.supports_json is True
    assert import_artifact_cmd.policy.mutates_state is True


def test_root_json_is_global_for_version() -> None:
    result = runner.invoke(app, ["--json", "version"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["app"] == "Ursa"


def test_json_rejected_for_non_json_command() -> None:
    result = runner.invoke(app, ["--json", "server", "status"])

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "contract_violation"
    assert payload["error"]["details"]["command"] == "server/status"


def test_runtime_exempt_command_bypasses_runtime_guard(monkeypatch) -> None:
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)

    result = runner.invoke(app, ["--json", "version"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["app"] == "Ursa"


def test_runtime_required_command_fails_without_active_env(monkeypatch) -> None:
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)

    result = runner.invoke(app, ["server", "status"])

    assert result.exit_code == 3
    assert "Runtime validation failed." in result.stderr
    assert "ursa-conda-active-env" in result.stderr
    assert "source ./activate <deploy-name>" in result.stderr
