from __future__ import annotations

import importlib
import sys
from pathlib import Path


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


def test_ursa_server_start_uses_packaged_entrypoint(monkeypatch):
    from daylib.cli import server as server_mod
    import daylib.ursa_config as ursa_config_mod

    class DummyUrsaConfig:
        aws_profile = "test-profile"
        is_configured = True
        cognito_user_pool_id = None
        cognito_app_client_id = None
        cognito_region = None

        def get_allowed_regions(self):
            return ["us-west-2"]

    monkeypatch.setenv("AWS_PROFILE", "test-profile")
    monkeypatch.setattr(ursa_config_mod, "get_ursa_config", lambda reload=False: DummyUrsaConfig())
    monkeypatch.setattr(server_mod, "_ensure_dir", lambda: None)
    monkeypatch.setattr(server_mod, "_get_pid", lambda: None)
    monkeypatch.setattr(server_mod, "_source_env_file", lambda: False)

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd=None, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(server_mod.subprocess, "run", _fake_run)

    server_mod.start(
        port=1234,
        host="127.0.0.1",
        auth=False,
        reload=False,
        background=False,
    )

    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    assert cmd[:3] == [sys.executable, "-m", "daylib.workset_api_cli"]
    assert not any("bin/daylily-workset-api" in str(part) for part in cmd)
