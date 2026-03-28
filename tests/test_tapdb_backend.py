"""Tests for the Ursa TapDB composition adapter and runtime helpers."""

from __future__ import annotations

import inspect
from pathlib import Path

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
    monkeypatch.delenv("URSA_TAPDB_ENV_LOCAL", raising=False)
    monkeypatch.delenv("URSA_TAPDB_ENV_AURORA", raising=False)
    monkeypatch.setattr(tapdb_runtime, "_detect_tapdb_env_for_target", lambda _target: None)

    assert tapdb_runtime.tapdb_env_for_target("local") == "dev"
    assert tapdb_runtime.tapdb_env_for_target("aurora") == "prod"


def test_export_database_url_for_target_sets_runtime_environment(monkeypatch) -> None:
    monkeypatch.setattr(tapdb_runtime, "ensure_tapdb_version", lambda *_args, **_kwargs: "3.0.6")
    monkeypatch.setattr(
        tapdb_runtime,
        "_get_tapdb_db_config_for_env",
        lambda _env: {
            "user": "ursa_user",
            "password": "secret",
            "host": "db.example.test",
            "port": "5432",
            "database": "daylily_ursa",
        },
    )
    monkeypatch.setattr(tapdb_runtime, "_resolve_tapdb_config_path", lambda **_kwargs: None)

    db_url = tapdb_runtime.export_database_url_for_target(
        target="local",
        client_id="local",
        profile="test-profile",
        region="us-west-2",
        namespace="ursa",
        tapdb_env="dev",
    )

    assert db_url == "postgresql+psycopg2://ursa_user:secret@db.example.test:5432/daylily_ursa"
    assert tapdb_runtime.os.environ["AWS_PROFILE"] == "test-profile"
    assert tapdb_runtime.os.environ["AWS_REGION"] == "us-west-2"
    assert tapdb_runtime.os.environ["TAPDB_CLIENT_ID"] == "local"
    assert tapdb_runtime.os.environ["TAPDB_DATABASE_NAME"] == "ursa"
    assert tapdb_runtime.os.environ["TAPDB_ENV"] == "dev"
    assert tapdb_runtime.os.environ["DATABASE_URL"] == db_url
