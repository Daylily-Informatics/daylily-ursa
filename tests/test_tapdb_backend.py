"""Tests for the Ursa TapDB composition adapter and runtime helpers."""

from __future__ import annotations

import inspect
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from daylib_ursa.integrations import tapdb_runtime
from daylib_ursa.tapdb_graph import backend as backend_module
from daylib_ursa.tapdb_graph.backend import (
    TEMPLATE_DEFINITIONS,
    URSA_TEMPLATE_DEFINITIONS,
    TapDBBackend,
    TemplateSpec,
    from_json_addl,
    to_action_history_entry,
    utc_now_iso,
)
from daylib_ursa.tapdb_templates import claim_ursa_template_prefixes


def _tapdb_dependency_spec() -> str:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    for dependency in pyproject["project"]["dependencies"]:
        if dependency.startswith("daylily-tapdb"):
            return dependency
    raise AssertionError("daylily-tapdb dependency missing from pyproject.toml")


def test_backend_adapter_reexports_tapdb_surface_without_legacy_inheritance() -> None:
    params = inspect.signature(TapDBBackend).parameters
    assert TapDBBackend.__mro__ == (TapDBBackend, object)
    assert "bundle" in params
    assert "app_username" in params
    assert backend_module.TEMPLATE_DEFINITIONS is TEMPLATE_DEFINITIONS
    assert callable(backend_module.from_json_addl)
    assert callable(backend_module.to_action_history_entry)
    assert callable(backend_module.utc_now_iso)


def test_template_definitions_cover_phase_one_objects() -> None:
    assert len(TEMPLATE_DEFINITIONS) >= 16
    codes = {spec.template_code for spec in TEMPLATE_DEFINITIONS}
    assert "RGX/analysis/run-linked/1.0/" in codes
    assert "RGX/workset/gui-ready/1.0/" in codes
    assert "RGX/manifest/dewey-bound/1.0/" in codes
    assert "RGX/auth/user-token/1.0/" in codes
    assert "RGX/auth/client-registration/1.0/" in codes


def test_template_definitions_are_template_spec_instances() -> None:
    for spec in URSA_TEMPLATE_DEFINITIONS:
        assert isinstance(spec, TemplateSpec)
        assert spec.template_code.endswith("/")


def test_template_definitions_use_frozen_prefix_remaps() -> None:
    template_pack = yaml.safe_load(
        Path("config/tapdb_templates/ursa/templates.json").read_text(encoding="utf-8")
    )
    prefixes = {template["instance_prefix"] for template in template_pack["templates"]}
    assert prefixes == {"RGX"}


def test_from_json_addl_extracts_dict() -> None:
    class _FakeInstance:
        json_addl = {"foo": "bar", "n": 42}

    result = from_json_addl(_FakeInstance())
    assert result == {"foo": "bar", "n": 42}
    assert result is not _FakeInstance.json_addl


def test_from_json_addl_handles_none() -> None:
    class _FakeInstance:
        json_addl = None

    assert from_json_addl(_FakeInstance()) == {}


def test_to_action_history_entry_structure() -> None:
    entry = to_action_history_entry("a", "b", key="val")
    assert entry == {"args": ["a", "b"], "kwargs": {"key": "val"}}


def test_utc_now_iso_format() -> None:
    ts = utc_now_iso()
    assert ts.endswith("Z") or "+00:00" in ts
    assert "T" in ts


def test_adapter_module_has_no_legacy_repo_reference() -> None:
    source = Path("daylib_ursa/tapdb_graph/backend.py").read_text(encoding="utf-8")
    assert "UrsaTapdbRepository" not in source
    assert "TapdbClientBundle" in source
    assert "sys.path.insert" not in source


def test_tapdb_env_for_target_uses_explicit_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        tapdb_runtime,
        "_detect_tapdb_env_for_target",
        lambda _target, **_kwargs: None,
    )

    assert tapdb_runtime.tapdb_env_for_target("local") == "dev"
    assert tapdb_runtime.tapdb_env_for_target("aurora") == "prod"


def test_export_database_url_for_target_sets_runtime_environment(monkeypatch) -> None:
    monkeypatch.setattr(
        tapdb_runtime, "ensure_tapdb_version", lambda *_args, **_kwargs: _tapdb_dependency_spec()
    )
    monkeypatch.setattr(
        tapdb_runtime,
        "_get_tapdb_db_config_for_env",
        lambda _env, **_kwargs: {
            "user": "ursa_user",
            "password": "secret",
            "host": "db.example.test",
            "port": "5432",
            "database": "daylily_ursa",
        },
    )
    monkeypatch.setattr(
        tapdb_runtime,
        "_resolve_tapdb_config_path",
        lambda **_kwargs: "/tmp/ursa-tapdb.yaml",
    )

    db_url = tapdb_runtime.export_database_url_for_target(
        target="local",
        client_id="local",
        profile="test-profile",
        region="us-west-2",
        namespace="ursa",
        tapdb_env="dev",
    )

    assert db_url == "postgresql+psycopg2://ursa_user:secret@db.example.test:5432/daylily_ursa"
    assert "DATABASE_URL" not in tapdb_runtime.os.environ


def test_run_tapdb_cli_exports_explicit_identity_env(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(tapdb_runtime, "ensure_tapdb_version", lambda: _tapdb_dependency_spec())
    monkeypatch.setattr(
        tapdb_runtime,
        "_resolve_tapdb_config_path",
        lambda **_kwargs: "/tmp/ursa-tapdb.yaml",
    )

    def fake_run(cmd, *, cwd=None, env=None, capture_output, text):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tapdb_runtime.subprocess, "run", fake_run)

    result = tapdb_runtime.run_tapdb_cli(
        ["bootstrap", "local", "--no-gui"],
        target="local",
        client_id="local",
        profile="lsmc",
        region="us-west-2",
        namespace="ursa",
    )

    assert result.returncode == 0
    assert captured["cmd"][:5] == [
        sys.executable,
        "-m",
        "daylily_tapdb.cli",
        "--config",
        "/tmp/ursa-tapdb.yaml",
    ]
    assert captured["cmd"][5:7] == ["--env", "dev"]
    assert captured["env"]["MERIDIAN_DOMAIN_CODE"] == "Z"
    assert captured["env"]["TAPDB_OWNER_REPO"] == "ursa"


def test_ensure_local_tapdb_namespace_config_initializes_namespaced_config(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {"cmds": []}
    config_path = tmp_path / "tapdb" / "tapdb-config.yaml"

    monkeypatch.setattr(tapdb_runtime, "ensure_tapdb_version", lambda: _tapdb_dependency_spec())
    monkeypatch.setattr(
        tapdb_runtime,
        "_resolve_runtime_env",
        lambda **_kwargs: {
            "aws_profile": "lsmc",
            "aws_region": "us-west-2",
            "client_id": "local",
            "database_name": "ursa",
            "tapdb_env": "dev",
            "config_path": str(config_path),
            "domain_code": "Z",
            "owner_repo_name": "ursa",
            "domain_registry_path": str(tmp_path / "domain.json"),
            "prefix_registry_path": str(tmp_path / "prefix.json"),
        },
    )

    def fake_run(cmd, *, env=None, capture_output, text):
        captured["cmds"].append(cmd)
        captured["env"] = env
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(tapdb_runtime.subprocess, "run", fake_run)

    result = tapdb_runtime.ensure_local_tapdb_namespace_config(config_path=str(config_path))

    assert result.returncode == 0
    assert config_path.parent.is_dir()
    assert captured["cmds"] == [
        [
            sys.executable,
            "-m",
            "daylily_tapdb.cli",
            "--config",
            str(config_path),
            "db-config",
            "init",
            "--client-id",
            "local",
            "--database-name",
            "ursa",
            "--owner-repo-name",
            "ursa",
            "--domain-code",
            "dev=Z",
            "--domain-registry-path",
            str(tmp_path / "domain.json"),
            "--prefix-ownership-registry-path",
            str(tmp_path / "prefix.json"),
            "--env",
            "dev",
            "--db-port",
            "dev=5588",
            "--ui-port",
            "dev=8918",
        ],
        [
            sys.executable,
            "-m",
            "daylily_tapdb.cli",
            "--config",
            str(config_path),
            "--client-id",
            "local",
            "--database-name",
            "ursa",
            "db-config",
            "update",
            "--env",
            "dev",
            "--owner-repo-name",
            "ursa",
            "--domain-code",
            "Z",
            "--domain-registry-path",
            str(tmp_path / "domain.json"),
            "--prefix-ownership-registry-path",
            str(tmp_path / "prefix.json"),
            "--engine-type",
            "local",
            "--host",
            "localhost",
            "--port",
            "5588",
            "--ui-port",
            "8918",
            "--database",
            "ursa",
        ],
    ]
    assert captured["env"]["MERIDIAN_DOMAIN_CODE"] == "Z"
    assert captured["env"]["TAPDB_OWNER_REPO"] == "ursa"


def test_ensure_local_tapdb_namespace_config_requires_explicit_config_path(monkeypatch) -> None:
    monkeypatch.setattr(
        tapdb_runtime,
        "_resolve_runtime_env",
        lambda **_kwargs: {
            "aws_profile": "lsmc",
            "aws_region": "us-west-2",
            "client_id": "local",
            "database_name": "ursa",
            "tapdb_env": "dev",
            "config_path": "",
            "domain_code": "Z",
            "owner_repo_name": "ursa",
            "domain_registry_path": "/tmp/domain.json",
            "prefix_registry_path": "/tmp/prefix.json",
        },
    )

    with pytest.raises(tapdb_runtime.TapDBRuntimeError, match="TapDB config path is required"):
        tapdb_runtime.ensure_local_tapdb_namespace_config(config_path="")


@pytest.mark.parametrize("missing_key", ["domain_registry_path", "prefix_registry_path"])
def test_ensure_local_tapdb_namespace_config_requires_explicit_registry_paths(
    monkeypatch, missing_key
) -> None:
    runtime_env = {
        "aws_profile": "lsmc",
        "aws_region": "us-west-2",
        "client_id": "local",
        "database_name": "ursa",
        "tapdb_env": "dev",
        "config_path": "/tmp/ursa-tapdb.yaml",
        "domain_code": "Z",
        "owner_repo_name": "ursa",
        "domain_registry_path": "/tmp/domain.json",
        "prefix_registry_path": "/tmp/prefix.json",
    }
    runtime_env[missing_key] = ""
    monkeypatch.setattr(tapdb_runtime, "_resolve_runtime_env", lambda **_kwargs: runtime_env)

    with pytest.raises(
        tapdb_runtime.TapDBRuntimeError,
        match="domain-registry-path|prefix-ownership-registry-path",
    ):
        tapdb_runtime.ensure_local_tapdb_namespace_config(config_path="/tmp/ursa-tapdb.yaml")


def test_resolve_tapdb_config_path_requires_explicit_path() -> None:
    resolved = tapdb_runtime._resolve_tapdb_config_path(namespace="ursa", client_id="local")

    assert resolved is None


def test_resolve_tapdb_config_path_returns_explicit_path() -> None:
    resolved = tapdb_runtime._resolve_tapdb_config_path(
        namespace="ursa",
        client_id="local",
        config_path="/tmp/ursa-tapdb.yaml",
    )

    assert resolved == "/tmp/ursa-tapdb.yaml"


def test_resolved_default_identity_uses_settings_config_and_registry_paths(monkeypatch) -> None:
    monkeypatch.setenv("TAPDB_CONFIG_PATH", "/tmp/from-env.yaml")
    monkeypatch.setenv("TAPDB_CLIENT_ID", "env-client")
    monkeypatch.setenv("TAPDB_DATABASE_NAME", "env-db")
    monkeypatch.setenv("TAPDB_ENV", "prod")

    monkeypatch.setitem(
        sys.modules,
        "daylib_ursa.config",
            SimpleNamespace(
                get_settings=lambda: SimpleNamespace(
                    tapdb_client_id="yaml-client",
                    tapdb_database_name="yaml-db",
                    tapdb_env="dev",
                    tapdb_config_path="/tmp/from-yaml.yaml",
                    tapdb_domain_registry_path="/tmp/domain_code_registry.json",
                    tapdb_prefix_ownership_registry_path="/tmp/prefix_ownership_registry.json",
                )
            ),
    )
    try:
        assert tapdb_runtime._resolved_default_identity() == (
            "env-client",
            "env-db",
            "prod",
            "/tmp/from-yaml.yaml",
            "/tmp/domain_code_registry.json",
            "/tmp/prefix_ownership_registry.json",
        )
    finally:
        sys.modules.pop("daylib_ursa.config", None)


def test_repo_ships_tapdb_config_template() -> None:
    template_path = Path("config/tapdb-config-ursa.yaml")
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))

    assert template_path.is_file()
    assert payload["meta"]["config_version"] == 3
    assert payload["meta"]["client_id"] == "local"
    assert payload["meta"]["database_name"] == "ursa"
    assert payload["meta"]["owner_repo_name"] == "ursa"
    assert payload["meta"]["domain_code"] == "Z"
    assert payload["meta"]["domain_registry_path"] == "/absolute/path/to/domain_code_registry.json"
    assert (
        payload["meta"]["prefix_ownership_registry_path"]
        == "/absolute/path/to/prefix_ownership_registry.json"
    )
    assert payload["environments"]["dev"]["domain_code"] == "Z"
    assert payload["environments"]["dev"]["port"] == "5588"
    assert payload["environments"]["dev"]["database"] == "tapdb_ursa_dev"
    assert payload["environments"]["dev"]["audit_log_euid_prefix"] == "RGX"
    assert payload["environments"]["prod"]["domain_code"] == "Z"


def test_claim_ursa_template_prefixes_initializes_missing_registry(tmp_path: Path) -> None:
    domain_registry = tmp_path / "domain_code_registry.json"
    prefix_registry = tmp_path / "prefix_ownership_registry.json"
    domain_registry.write_text(
        json.dumps({"version": "0.4.0", "domains": {"Z": {"name": "localhost"}}}) + "\n",
        encoding="utf-8",
    )

    claimed = claim_ursa_template_prefixes(
        [{"instance_prefix": "RGX"}],
        domain_code="Z",
        owner_repo_name="ursa",
        domain_registry_path=domain_registry,
        prefix_registry_path=prefix_registry,
    )

    assert claimed == ["RGX"]
    payload = json.loads(prefix_registry.read_text(encoding="utf-8"))
    assert payload["ownership"]["Z"]["RGX"]["issuer_app_code"] == "ursa"


def test_claim_ursa_template_prefixes_rejects_conflict(tmp_path: Path) -> None:
    domain_registry = tmp_path / "domain_code_registry.json"
    prefix_registry = tmp_path / "prefix_ownership_registry.json"
    domain_registry.write_text(
        json.dumps({"version": "0.4.0", "domains": {"Z": {"name": "localhost"}}}) + "\n",
        encoding="utf-8",
    )
    prefix_registry.write_text(
        json.dumps(
            {
                "version": "0.4.0",
                "ownership": {"Z": {"RGX": {"issuer_app_code": "other-repo"}}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="claimed by 'other-repo'"):
        claim_ursa_template_prefixes(
            [{"instance_prefix": "RGX"}],
            domain_code="Z",
            owner_repo_name="ursa",
            domain_registry_path=domain_registry,
            prefix_registry_path=prefix_registry,
        )


def test_backend_scopes_template_lookups_to_bundle_domain_code() -> None:
    template_calls: list[tuple[str, dict[str, str]]] = []

    class _FakeTemplateManager:
        def get_template(self, _session, template_code: str, **kwargs):
            template_calls.append((template_code, kwargs))
            return None

    backend = TapDBBackend(
        bundle=SimpleNamespace(
            connection=SimpleNamespace(domain_code="z"),
            template_manager=_FakeTemplateManager(),
            instance_factory=SimpleNamespace(),
        ),
        app_username="ursa",
        client_id="local",
        namespace="ursa",
        tapdb_env="dev",
    )

    with pytest.raises(RuntimeError, match="Missing Ursa templates"):
        backend.ensure_templates(object())

    assert template_calls
    assert {kwargs["domain_code"] for _, kwargs in template_calls} == {"Z"}


def test_get_tapdb_bundle_scopes_instance_factory_to_runtime_domain(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(tapdb_runtime, "ensure_tapdb_version", lambda: "6.0.4")
    monkeypatch.setattr(
        tapdb_runtime,
        "_resolve_runtime_env",
        lambda **_kwargs: {
            "tapdb_env": "dev",
            "client_id": "local",
            "database_name": "ursa",
            "aws_region": "us-west-2",
            "domain_code": "Z",
            "owner_repo_name": "ursa",
        },
    )
    monkeypatch.setattr(
        tapdb_runtime, "_require_config_path", lambda _runtime_env: "/tmp/ursa.yaml"
    )
    monkeypatch.setattr(
        tapdb_runtime,
        "_get_tapdb_db_config_for_env",
        lambda *_args, **_kwargs: {
            "host": "localhost",
            "port": "5432",
            "user": "ursa",
            "password": "secret",
            "database": "tapdb_ursa_dev",
            "engine_type": "postgres",
        },
    )

    class _FakeConnection:
        def __init__(self, **kwargs):
            captured["connection_kwargs"] = kwargs

    class _FakeTemplateManager:
        def __init__(self, config_path):
            captured["template_config_path"] = str(config_path)

    class _FakeInstanceFactory:
        def __init__(self, template_manager, *, domain_code=None):
            captured["instance_factory_template_manager"] = template_manager
            captured["instance_factory_domain_code"] = domain_code

    monkeypatch.setattr(tapdb_runtime, "TAPDBConnection", _FakeConnection)
    monkeypatch.setattr(tapdb_runtime, "TemplateManager", _FakeTemplateManager)
    monkeypatch.setattr(tapdb_runtime, "InstanceFactory", _FakeInstanceFactory)

    bundle = tapdb_runtime.get_tapdb_bundle()

    assert captured["connection_kwargs"]["domain_code"] == "Z"
    assert captured["template_config_path"] == "/tmp/ursa.yaml"
    assert captured["instance_factory_domain_code"] == "Z"
    assert bundle.connection is not None
