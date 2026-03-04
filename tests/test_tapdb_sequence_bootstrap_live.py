"""Optional live TapDB sequence readiness smoke test."""

from __future__ import annotations

import os

import pytest

from daylib.tapdb_graph.backend import TapDBBackend


@pytest.mark.skipif(
    os.environ.get("URSA_RUN_TAPDB_LIVE_TESTS") != "1",
    reason="Set URSA_RUN_TAPDB_LIVE_TESTS=1 to enable live TapDB checks.",
)
def test_live_bootstrap_ensures_required_instance_sequences():
    os.environ.setdefault("TAPDB_STRICT_NAMESPACE", "1")
    os.environ.setdefault("TAPDB_CLIENT_ID", "local")
    os.environ.setdefault("TAPDB_DATABASE_NAME", "ursa")
    os.environ.setdefault("TAPDB_ENV", "dev")

    backend = TapDBBackend(app_username="ursa-live-sequence-test")
    with backend.session_scope(commit=True) as session:
        backend.ensure_templates(session)
        missing = backend.get_missing_instance_sequences(session)

    assert missing == []

