from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner


def _read_project_scripts(pyproject: Path) -> dict[str, str]:
    """Parse the [project.scripts] table from pyproject.toml.

    Avoids adding a TOML parser dependency; this project keeps that section
    simple (string literals with no inline tables).
    """

    scripts: dict[str, str] = {}
    in_section = False
    for raw_line in pyproject.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_section = line == "[project.scripts]"
            continue
        if not in_section:
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        scripts[key] = value
    return scripts


def test_console_script_entrypoints_are_importable_and_callable():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = _read_project_scripts(pyproject)
    assert scripts, "No [project.scripts] entries found in pyproject.toml"

    for name, target in scripts.items():
        assert ":" in target, f"Console script {name!r} has unexpected target: {target!r}"
        module_name, func_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        assert callable(func), f"Console script {name!r} target is not callable: {target!r}"


def test_ursa_server_start_uses_packaged_entrypoint(
    monkeypatch,
    tmp_path: Path,
):
    from daylib_ursa.cli import server as server_mod
    import daylib_ursa.ursa_config as ursa_config_mod

    class DummyUrsaConfig:
        aws_profile = "test-profile"
        is_configured = True
        cognito_user_pool_id = "us-west-2_testpool"
        cognito_app_client_id = "test-app-client"
        cognito_region = "us-west-2"
        cognito_domain = "ursa-auth"
        cognito_callback_url = "https://localhost:8913/auth/callback"
        cognito_logout_url = "https://localhost:8913/login"

        def get_allowed_regions(self):
            return ["us-west-2"]

    monkeypatch.setenv("AWS_PROFILE", "test-profile")
    monkeypatch.setattr(ursa_config_mod, "get_ursa_config", lambda reload=False: DummyUrsaConfig())
    monkeypatch.setattr(server_mod, "_ensure_dir", lambda: None)
    monkeypatch.setattr(server_mod, "_get_pid", lambda: None)
    monkeypatch.setattr(server_mod, "_require_auth_dependencies", lambda: None)
    monkeypatch.setattr(server_mod, "_run_cognito_uri_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(server_mod, "_write_runtime_meta", lambda **_kwargs: None)
    monkeypatch.setattr(server_mod, "_clear_runtime_meta", lambda: None)
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_text("cert", encoding="utf-8")
    key_path.write_text("key", encoding="utf-8")
    monkeypatch.setattr(
        server_mod,
        "get_settings",
        lambda: SimpleNamespace(
            database_backend="tapdb",
            database_target="local",
            tapdb_client_id="local",
            tapdb_database_name="ursa",
            tapdb_env="dev",
            api_host="0.0.0.0",
            api_port=8913,
        ),
    )
    monkeypatch.setattr(
        server_mod, "export_database_url_for_target", lambda **_kwargs: "postgresql://test-db"
    )

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd=None, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["kwargs"] = kwargs

        class _DummyCompletedProcess:
            def __init__(self, returncode: int):
                self.returncode = returncode

        return _DummyCompletedProcess(0)

    monkeypatch.setattr(server_mod.subprocess, "run", _fake_run)

    server_mod.start(
        port=1234,
        host="127.0.0.1",
        ssl=True,
        cert=str(cert_path),
        key=str(key_path),
        reload=False,
        background=False,
    )

    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    assert "--ssl" in cmd
    assert "--cert" in cmd
    assert "--key" in cmd
    assert str(cert_path) in cmd
    assert str(key_path) in cmd
    assert "--profile" in cmd
    assert "test-profile" in cmd

    kwargs = captured.get("kwargs")
    assert isinstance(kwargs, dict)
    env = kwargs.get("env")
    assert isinstance(env, dict)
    assert env["DATABASE_BACKEND"] == "tapdb"
    assert env["DATABASE_TARGET"] == "local"
    assert env["DATABASE_URL"] == "postgresql://test-db"
    assert env["MERIDIAN_DOMAIN_CODE"] == "Z"
    assert env["TAPDB_APP_CODE"] == "R"
    assert "TAPDB_CLIENT_ID" not in env
    assert "TAPDB_DATABASE_NAME" not in env
    assert "TAPDB_ENV" not in env


def test_ursa_server_start_command_uses_module_entrypoint_and_profile(
    monkeypatch,
    tmp_path: Path,
):
    from daylib_ursa.cli import server as server_mod
    import daylib_ursa.ursa_config as ursa_config_mod

    class DummyUrsaConfig:
        aws_profile = "test-profile"
        is_configured = True
        cognito_user_pool_id = "us-west-2_testpool"
        cognito_app_client_id = "test-app-client"
        cognito_region = "us-west-2"
        cognito_domain = "ursa-auth"
        cognito_callback_url = "https://localhost:8913/auth/callback"
        cognito_logout_url = "https://localhost:8913/login"

        def get_allowed_regions(self):
            return ["us-west-2"]

    monkeypatch.setenv("AWS_PROFILE", "test-profile")
    monkeypatch.setattr(ursa_config_mod, "get_ursa_config", lambda reload=False: DummyUrsaConfig())
    monkeypatch.setattr(server_mod, "_ensure_dir", lambda: None)
    monkeypatch.setattr(server_mod, "_get_pid", lambda: None)
    monkeypatch.setattr(server_mod, "_require_auth_dependencies", lambda: None)
    monkeypatch.setattr(server_mod, "_run_cognito_uri_check", lambda *args, **kwargs: None)
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_text("cert", encoding="utf-8")
    key_path.write_text("key", encoding="utf-8")
    monkeypatch.setattr(
        server_mod,
        "get_settings",
        lambda: SimpleNamespace(
            database_backend="tapdb",
            database_target="local",
            tapdb_client_id="local",
            tapdb_database_name="ursa",
            tapdb_env="dev",
            api_host="0.0.0.0",
            api_port=8913,
        ),
    )
    monkeypatch.setattr(
        server_mod, "export_database_url_for_target", lambda **_kwargs: "postgresql://test-db"
    )

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd=None, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["kwargs"] = kwargs

        class _DummyCompletedProcess:
            def __init__(self, returncode: int):
                self.returncode = returncode

        return _DummyCompletedProcess(0)

    monkeypatch.setattr(server_mod.subprocess, "run", _fake_run)

    server_mod.start(
        port=1234,
        host="127.0.0.1",
        ssl=True,
        cert=str(cert_path),
        key=str(key_path),
        reload=False,
        background=False,
    )

    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    assert cmd[:3] == [sys.executable, "-m", "daylib_ursa.workset_api_cli"]
    assert not any("bin/daylily-workset-api" in str(part) for part in cmd)
    assert "--ssl" in cmd
    assert "--cert" in cmd
    assert "--key" in cmd
    assert str(cert_path) in cmd
    assert str(key_path) in cmd


def test_ursa_config_template_bytes_are_fresh() -> None:
    from daylib_ursa.config import build_default_config_template

    first = build_default_config_template()
    second = build_default_config_template()

    assert first != second
    text = first.decode("utf-8")
    assert "session_secret_key:" in text
    assert "default_tenant_id: 00000000-0000-0000-0000-000000000000" in text
    assert "auto_provision_allowed_domains:" in text
    assert "whitelist_domains: lsmc.com,lsmc.bio,lsmc.life,daylilyinformatics.com" in text
    assert "api_port: 8913" in text
    assert "cognito_callback_url: https://localhost:8913/auth/callback" in text
    assert "cognito_logout_url: https://localhost:8913/login" in text


def test_ursa_server_start_allows_ambient_credentials(monkeypatch):
    from daylib_ursa.cli import server as server_mod
    import daylib_ursa.ursa_config as ursa_config_mod

    class DummyUrsaConfig:
        aws_profile = None
        is_configured = True
        cognito_user_pool_id = "us-west-2_testpool"
        cognito_app_client_id = "test-app-client"
        cognito_region = "us-west-2"
        cognito_domain = "ursa-auth"
        cognito_callback_url = "https://localhost:8913/auth/callback"
        cognito_logout_url = "https://localhost:8913/login"

        def get_allowed_regions(self):
            return ["us-west-2"]

    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setattr(ursa_config_mod, "get_ursa_config", lambda reload=False: DummyUrsaConfig())
    monkeypatch.setattr(server_mod, "_ensure_dir", lambda: None)
    monkeypatch.setattr(server_mod, "_get_pid", lambda: None)
    monkeypatch.setattr(server_mod, "_require_auth_dependencies", lambda: None)
    monkeypatch.setattr(server_mod, "_run_cognito_uri_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(server_mod, "_write_runtime_meta", lambda **_kwargs: None)
    monkeypatch.setattr(server_mod, "_clear_runtime_meta", lambda: None)
    monkeypatch.setattr(
        server_mod,
        "get_settings",
        lambda: SimpleNamespace(
            database_backend="tapdb",
            database_target="local",
            tapdb_client_id="local",
            tapdb_database_name="ursa",
            tapdb_env="dev",
            api_host="0.0.0.0",
            api_port=8913,
        ),
    )
    monkeypatch.setattr(
        server_mod, "export_database_url_for_target", lambda **_kwargs: "postgresql://test-db"
    )

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd=None, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["kwargs"] = kwargs

        class _DummyCompletedProcess:
            def __init__(self, returncode: int):
                self.returncode = returncode

        return _DummyCompletedProcess(0)

    monkeypatch.setattr(server_mod.subprocess, "run", _fake_run)

    server_mod.start(
        port=1234,
        host="127.0.0.1",
        ssl=False,
        reload=False,
        background=False,
    )

    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    assert cmd[:3] == [sys.executable, "-m", "daylib_ursa.workset_api_cli"]
    assert "--profile" not in cmd
    assert "--no-ssl" in cmd
    assert "--cert" not in cmd
    assert "--key" not in cmd

    kwargs = captured.get("kwargs")
    assert isinstance(kwargs, dict)
    env = kwargs.get("env")
    assert isinstance(env, dict)
    assert "AWS_PROFILE" not in env
    assert env["DATABASE_URL"] == "postgresql://test-db"
    assert env["MERIDIAN_DOMAIN_CODE"] == "Z"
    assert env["TAPDB_APP_CODE"] == "R"
    assert "TAPDB_CLIENT_ID" not in env
    assert "TAPDB_DATABASE_NAME" not in env
    assert "TAPDB_ENV" not in env


def test_ursa_server_restart_forwards_tls_flags(monkeypatch):
    from daylib_ursa.cli import server as server_mod

    called: dict[str, object] = {}

    monkeypatch.setattr(server_mod, "stop", lambda: called.setdefault("stop", True))

    def fake_start(**kwargs):
        called["start_kwargs"] = kwargs

    monkeypatch.setattr(server_mod, "start", fake_start)

    server_mod.restart(port=1234, host="127.0.0.1", ssl=False, cert=None, key=None)

    assert called["stop"] is True
    assert called["start_kwargs"] == {
        "port": 1234,
        "host": "127.0.0.1",
        "ssl": False,
        "cert": None,
        "key": None,
        "reload": False,
        "background": True,
    }


def test_ursa_cli_exposes_standardized_groups():
    from daylib_ursa.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "version" in result.output
    assert "info" in result.output
    assert "server" in result.output
    assert "config" in result.output
    assert "env" in result.output
    assert "quality" in result.output
    assert "integrations" in result.output
    assert "monitor" in result.output
    assert "doctor" not in result.output
    assert "logs" not in result.output


def test_ursa_cli_exposes_dewey_integration_group():
    from daylib_ursa.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["integrations", "dewey", "--help"])

    assert result.exit_code == 0
    assert "resolve-artifact" in result.output
    assert "resolve-artifact-set" in result.output
    assert "import-artifact" in result.output


# ---------------------------------------------------------------------------
# §2.3 — Strict config validation: dict-format regions must be rejected
# ---------------------------------------------------------------------------


class TestStrictConfigValidation:
    """Verify that deprecated config formats are rejected, not silently tolerated."""

    def test_validate_config_file_rejects_dict_regions(self, tmp_path):
        import yaml

        from daylib_ursa.ursa_config import validate_config_file

        config_file = tmp_path / "ursa-config.yaml"
        config_file.write_text(yaml.dump({"regions": {"us-west-2": "bucket"}}))
        valid, errors, warnings = validate_config_file(config_file)
        assert not valid
        assert any("must be a list, not a dict" in e for e in errors), (
            f"Expected dict-format rejection error, got errors={errors}"
        )

    def test_load_raises_on_dict_regions(self, tmp_path):
        import yaml

        from daylib_ursa.ursa_config import UrsaConfig

        config_file = tmp_path / "ursa-config.yaml"
        config_file.write_text(yaml.dump({"regions": {"us-west-2": "bucket"}}))

        with pytest.raises(ValueError, match="must be a list, not a dict"):
            UrsaConfig.load(config_file)

    def test_validate_config_file_accepts_list_regions(self, tmp_path):
        import yaml

        from daylib_ursa.ursa_config import validate_config_file

        config_file = tmp_path / "ursa-config.yaml"
        config_file.write_text(yaml.dump({"regions": ["us-west-2", "eu-central-1"]}))
        valid, errors, warnings = validate_config_file(config_file)
        assert valid, f"Expected valid config, got errors={errors}"
