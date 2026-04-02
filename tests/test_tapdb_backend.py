"""Tests for the Ursa TapDB composition adapter and runtime helpers."""

from __future__ import annotations

import inspect
from pathlib import Path

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
    assert "workflow/analysis/run-linked/1.0/" in codes
    assert "workflow/workset/gui-ready/1.0/" in codes
    assert "data/manifest/dewey-bound/1.0/" in codes
    assert "integration/auth/user-token/1.0/" in codes
    assert "integration/auth/client-registration/1.0/" in codes


def test_template_definitions_are_template_spec_instances() -> None:
    for spec in URSA_TEMPLATE_DEFINITIONS:
        assert isinstance(spec, TemplateSpec)
        assert spec.template_code.endswith("/")


def test_template_definitions_use_frozen_prefix_remaps() -> None:
    template_pack = yaml.safe_load(Path("config/tapdb_templates/ursa/templates.json").read_text(encoding="utf-8"))
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
    monkeypatch.setattr(tapdb_runtime, "ensure_tapdb_version", lambda *_args, **_kwargs: "3.0.9")
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


def test_resolve_tapdb_config_path_prefers_deployment_scoped_user_config(monkeypatch, tmp_path) -> None:
    scoped = (
        tmp_path
        / ".config"
        / "tapdb"
        / "local"
        / "ursa-local2"
        / "tapdb-config.yaml"
    )
    scoped.parent.mkdir(parents=True, exist_ok=True)
    scoped.write_text("meta: {}\n", encoding="utf-8")

    monkeypatch.delenv("TAPDB_CONFIG_PATH", raising=False)
    monkeypatch.delenv("URSA_DEPLOYMENT_CODE", raising=False)
    monkeypatch.delenv("LSMC_DEPLOYMENT_CODE", raising=False)
    monkeypatch.setenv("DEPLOYMENT_CODE", "local2")
    monkeypatch.setattr(tapdb_runtime.Path, "home", classmethod(lambda cls: tmp_path))

    resolved = tapdb_runtime._resolve_tapdb_config_path(namespace="ursa", client_id="local")

    assert resolved == str(scoped)


def test_repo_ships_tapdb_config_template() -> None:
    template_path = Path("config/tapdb-config-ursa.yaml")
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))

    assert template_path.is_file()
    assert payload["meta"]["config_version"] == 3
    assert payload["meta"]["client_id"] == "local"
    assert payload["meta"]["database_name"] == "ursa"
    assert payload["environments"]["dev"]["port"] == "5588"
    assert payload["environments"]["dev"]["database"] == "tapdb_ursa_dev"
    assert payload["environments"]["dev"]["audit_log_euid_prefix"] == "RGX"
