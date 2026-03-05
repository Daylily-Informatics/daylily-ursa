"""Unit tests for TapDB backend readiness hardening."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from daylib.tapdb_graph.backend import TEMPLATE_DEFINITIONS, TapDBBackend, expected_ursa_database_name


def test_ensure_templates_calls_sequence_readiness():
    backend = TapDBBackend.__new__(TapDBBackend)
    backend._ensure_template = MagicMock()
    backend.ensure_instance_sequences = MagicMock()
    session = MagicMock()

    backend.ensure_templates(session)

    assert backend._ensure_template.call_count == len(TEMPLATE_DEFINITIONS)
    backend.ensure_instance_sequences.assert_called_once_with(session)


def test_ensure_instance_sequences_ensures_each_required_prefix_once():
    backend = TapDBBackend.__new__(TapDBBackend)
    backend._required_instance_prefixes = MagicMock(return_value=["CT", "WS", "CT"])
    session = MagicMock()

    with patch("daylib.tapdb_graph.backend.ensure_instance_prefix_sequence") as ensure_seq:
        backend.ensure_instance_sequences(session)

    ensure_seq.assert_has_calls(
        [
            call(session, "CT"),
            call(session, "WS"),
        ]
    )
    assert ensure_seq.call_count == 2


def test_create_instance_ensures_template_and_sequence():
    backend = TapDBBackend.__new__(TapDBBackend)
    backend.templates = MagicMock()
    backend.factory = MagicMock()
    backend.ensure_templates = MagicMock()
    backend._normalize_prefix = TapDBBackend._normalize_prefix

    template = SimpleNamespace(instance_prefix="ct")
    backend.templates.get_template.side_effect = [None, template]

    row = SimpleNamespace(json_addl={}, bstatus="active", is_singleton=False)
    backend.factory.create_instance.return_value = row
    session = MagicMock()

    with patch("daylib.tapdb_graph.backend.ensure_instance_prefix_sequence") as ensure_seq:
        created = backend.create_instance(
            session=session,
            template_code="actor/customer/account/1.0/",
            name="Customer",
            json_addl={"customer_id": "cust-001"},
            bstatus="created",
            singleton=True,
        )

    backend.ensure_templates.assert_called_once_with(session)
    ensure_seq.assert_called_once_with(session, "CT")
    assert created.json_addl["customer_id"] == "cust-001"
    assert created.bstatus == "created"
    assert created.is_singleton is True


def test_get_missing_instance_sequences_reports_gaps():
    backend = TapDBBackend.__new__(TapDBBackend)
    backend._required_instance_sequence_names = MagicMock(
        return_value=["ct_instance_seq", "ws_instance_seq"]
    )
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = [
        ("ws_instance_seq",),
        ("other_instance_seq",),
    ]

    missing = backend.get_missing_instance_sequences(session)

    assert missing == ["ct_instance_seq"]


def test_expected_ursa_database_name_uses_env_suffix():
    assert expected_ursa_database_name("dev") == "daylily-ursa-dev"
    assert expected_ursa_database_name("prod") == "daylily-ursa-prod"


def test_init_rejects_non_ursa_database_name(monkeypatch):
    monkeypatch.setenv("TAPDB_ENV", "dev")

    cfg = {
        "host": "db.example",
        "port": "5432",
        "database": "tapdb_dev",
        "user": "tapdb_admin",
        "password": "secret",
        "engine_type": "aurora",
        "region": "us-west-2",
        "iam_auth": "true",
    }

    with (
        patch("daylib.tapdb_graph.backend.resolve_context"),
        patch("daylib.tapdb_graph.backend.get_db_config_for_env", return_value=cfg),
        pytest.raises(RuntimeError) as excinfo,
    ):
        TapDBBackend(app_username="test")

    message = str(excinfo.value)
    assert "daylily-ursa-dev" in message
    assert "tapdb_dev" in message


def test_init_accepts_ursa_database_name(monkeypatch):
    monkeypatch.setenv("TAPDB_ENV", "dev")

    cfg = {
        "host": "db.example",
        "port": "5432",
        "database": "daylily-ursa-dev",
        "user": "tapdb_admin",
        "password": "secret",
        "engine_type": "aurora",
        "region": "us-west-2",
        "iam_auth": "true",
    }

    with (
        patch("daylib.tapdb_graph.backend.resolve_context"),
        patch("daylib.tapdb_graph.backend.get_db_config_for_env", return_value=cfg),
        patch("daylib.tapdb_graph.backend.TAPDBConnection") as mock_conn_cls,
    ):
        TapDBBackend(app_username="test")

    mock_conn_cls.assert_called_once()
    assert mock_conn_cls.call_args.kwargs["db_name"] == "daylily-ursa-dev"
