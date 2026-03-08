"""Unit tests for TapDB backend readiness hardening."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from daylib_ursa.tapdb_graph.backend import TEMPLATE_DEFINITIONS, TapDBBackend


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
    backend._required_instance_prefixes = MagicMock(return_value=["AN", "AF", "AN"])
    session = MagicMock()

    with patch("daylib_ursa.tapdb_graph.backend.ensure_instance_prefix_sequence") as ensure_seq:
        backend.ensure_instance_sequences(session)

    ensure_seq.assert_has_calls(
        [
            call(session, "AF"),
            call(session, "AN"),
        ]
    )
    assert ensure_seq.call_count == 2


def test_create_instance_ensures_template_and_sequence():
    backend = TapDBBackend.__new__(TapDBBackend)
    backend.templates = MagicMock()
    backend.factory = MagicMock()
    backend.ensure_templates = MagicMock()
    backend._normalize_prefix = TapDBBackend._normalize_prefix

    template = SimpleNamespace(instance_prefix="an")
    backend.templates.get_template.side_effect = [None, template]

    row = SimpleNamespace(json_addl={}, bstatus="active", is_singleton=False)
    backend.factory.create_instance.return_value = row
    session = MagicMock()

    with patch("daylib_ursa.tapdb_graph.backend.ensure_instance_prefix_sequence") as ensure_seq:
        created = backend.create_instance(
            session=session,
            template_code="workflow/analysis/run-linked/1.0/",
            name="Analysis",
            json_addl={"analysis_type": "beta-default"},
            bstatus="created",
            singleton=True,
        )

    backend.ensure_templates.assert_called_once_with(session)
    ensure_seq.assert_called_once_with(session, "AN")
    assert created.json_addl["analysis_type"] == "beta-default"
    assert created.bstatus == "created"
    assert created.is_singleton is True


def test_get_missing_instance_sequences_reports_gaps():
    backend = TapDBBackend.__new__(TapDBBackend)
    backend._required_instance_sequence_names = MagicMock(
        return_value=["af_instance_seq", "an_instance_seq"]
    )
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = [
        ("an_instance_seq",),
        ("other_instance_seq",),
    ]

    missing = backend.get_missing_instance_sequences(session)

    assert missing == ["af_instance_seq"]
