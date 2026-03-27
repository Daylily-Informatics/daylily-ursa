"""Tests for the thin Ursa TapDB adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_backend_adapter_reexports_tapdb_ursa_surface() -> None:
    assert TapDBBackend.__mro__[1].__name__ == "UrsaTapdbRepository"
    assert backend_module.TEMPLATE_DEFINITIONS is TEMPLATE_DEFINITIONS
    assert callable(backend_module.from_json_addl)
    assert callable(backend_module.to_action_history_entry)
    assert callable(backend_module.utc_now_iso)


def test_template_definitions_cover_phase_one_objects() -> None:
    """Templates are always locally defined — never empty."""
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
    # Must return a copy, not the original dict
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


def test_adapter_module_has_no_direct_sqlalchemy_import() -> None:
    source = Path("daylib_ursa/tapdb_graph/backend.py").read_text(encoding="utf-8")
    # backend.py must not import sqlalchemy directly — it uses TapDB models
    assert "import sqlalchemy" not in source
    assert "from sqlalchemy" not in source
    assert "sys.path.insert" not in source
