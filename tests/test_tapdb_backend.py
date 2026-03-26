"""Tests for the thin Ursa TapDB adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from daylib_ursa.tapdb_graph import backend as backend_module
from daylib_ursa.tapdb_graph.backend import TEMPLATE_DEFINITIONS, TapDBBackend


def test_backend_adapter_reexports_tapdb_ursa_surface() -> None:
    assert TapDBBackend.__mro__[1].__name__ == "UrsaTapdbRepository"
    assert backend_module.TEMPLATE_DEFINITIONS is TEMPLATE_DEFINITIONS
    assert callable(backend_module.from_json_addl)
    assert callable(backend_module.to_action_history_entry)
    assert callable(backend_module.utc_now_iso)


def test_template_definitions_cover_phase_one_objects_when_available() -> None:
    if not TEMPLATE_DEFINITIONS:
        pytest.skip("TapDB package in this environment does not expose URSA templates")

    codes = {spec.template_code for spec in TEMPLATE_DEFINITIONS}
    assert "workflow/analysis/run-linked/1.0/" in codes
    assert "workflow/workset/gui-ready/1.0/" in codes
    assert "data/manifest/dewey-bound/1.0/" in codes
    assert "integration/auth/user-token/1.0/" in codes
    assert "integration/auth/client-registration/1.0/" in codes


def test_adapter_module_fallback_is_explicit_when_templates_are_unavailable() -> None:
    if TEMPLATE_DEFINITIONS:
        pytest.skip("URSA templates are available in this environment")

    with pytest.raises(RuntimeError, match="TapDB backend is unavailable"):
        TapDBBackend()


def test_adapter_module_has_no_sqlalchemy_dependency() -> None:
    source = Path("daylib_ursa/tapdb_graph/backend.py").read_text(encoding="utf-8")
    assert "sqlalchemy" not in source
    assert "session.execute" not in source
    assert "sys.path.insert" not in source
