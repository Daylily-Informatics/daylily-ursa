"""Unit tests for daylily_ursa.manifest_registry using TapDB graph semantics."""

from unittest.mock import MagicMock, patch

import pytest

from daylily_ursa.manifest_registry import (
    ManifestRegistry,
    ManifestTooLargeError,
    SavedManifest,
    _estimate_sample_count,
    _gzip_b64_decode,
    _gzip_b64_encode,
    _sha256_hex,
)


class _SessionCtx:
    def __init__(self, session: MagicMock):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def manifest_registry() -> ManifestRegistry:
    registry = ManifestRegistry.__new__(ManifestRegistry)
    registry.backend = MagicMock()
    registry._session = MagicMock()
    registry.backend.session_scope.return_value = _SessionCtx(registry._session)
    return registry


class TestSaveManifestEncoding:
    def test_save_manifest_encodes_tsv_and_stores_metadata(self, manifest_registry: ManifestRegistry):
        tsv_content = "RUN_ID\tSAMPLE_ID\nR0\tHG002\n"

        customer = MagicMock(uuid=101, euid="cust-euid")
        manifest = MagicMock(euid="manifest-euid")
        manifest_registry.backend.find_instance_by_external_id.return_value = None
        manifest_registry.backend.create_instance.side_effect = [customer, manifest]

        saved = manifest_registry.save_manifest(
            customer_id="cust-001",
            tsv_content=tsv_content,
            name="Run 1",
            description="Test manifest",
        )

        assert isinstance(saved, SavedManifest)
        assert saved.customer_id == "cust-001"
        assert saved.manifest_id.startswith("m-")
        assert saved.sample_count == _estimate_sample_count(tsv_content)
        assert saved.tsv_sha256 == _sha256_hex(tsv_content)
        assert _gzip_b64_decode(saved.tsv_gzip_b64) == tsv_content
        assert saved.manifest_euid == "manifest-euid"

        manifest_payload = manifest_registry.backend.create_instance.call_args_list[1].kwargs["json_addl"]
        assert manifest_payload["customer_id"] == "cust-001"
        assert manifest_payload["sample_count"] == saved.sample_count
        assert manifest_payload["tsv_sha256"] == saved.tsv_sha256
        assert _gzip_b64_decode(manifest_payload["tsv_gzip_b64"]) == tsv_content

        manifest_registry.backend.create_lineage.assert_called_once()

    def test_save_manifest_rejects_oversized_payload(self, manifest_registry: ManifestRegistry):
        with patch("daylily_ursa.manifest_registry._gzip_b64_encode", return_value="x" * 340001):
            with pytest.raises(ManifestTooLargeError):
                manifest_registry.save_manifest(
                    customer_id="cust-001",
                    tsv_content="RUN_ID\tSAMPLE_ID\nR0\tHG002\n",
                    name="too-big",
                )


class TestListCustomerManifests:
    def test_list_customer_manifests_normalizes_fields(self, manifest_registry: ManifestRegistry):
        customer = MagicMock(uuid=101)
        row = MagicMock(
            euid="manifest-euid",
            name="Run 1",
            created_dt=None,
            modified_dt=None,
            bstatus="active",
            json_addl={
                "manifest_id": "m-1",
                "customer_id": "cust-001",
                "name": "Run 1",
                "description": "",
                "created_at": "2026-01-01T00:00:00Z",
                "sample_count": "3",
                "tsv_sha256": "abc",
            },
        )
        manifest_registry.backend.find_instance_by_external_id.return_value = customer
        manifest_registry.backend.get_customer_owned.return_value = [row]

        manifests = manifest_registry.list_customer_manifests("cust-001")

        assert len(manifests) == 1
        m = manifests[0]
        assert m["manifest_id"] == "m-1"
        assert m["customer_id"] == "cust-001"
        assert m["name"] == "Run 1"
        assert m["description"] == ""
        assert m["sample_count"] == 3
        assert m["manifest_euid"] == "manifest-euid"

    def test_list_customer_manifests_empty_when_customer_missing(self, manifest_registry: ManifestRegistry):
        manifest_registry.backend.find_instance_by_external_id.return_value = None
        assert manifest_registry.list_customer_manifests("cust-404") == []


class TestGetManifestAndTsv:
    def test_get_manifest_and_tsv_round_trip(self, manifest_registry: ManifestRegistry):
        tsv_content = "RUN_ID\tSAMPLE_ID\tR1_FQ\tR2_FQ\nrun1\tsample1\tr1.fq\tr2.fq\n"
        encoded = _gzip_b64_encode(tsv_content)

        row = MagicMock(
            euid="manifest-euid",
            name="Run 1",
            created_dt=None,
            modified_dt=None,
            bstatus="active",
            json_addl={
                "manifest_id": "m-1",
                "customer_id": "cust-001",
                "name": "Run 1",
                "description": "",
                "created_at": "2026-01-01T00:00:00Z",
                "sample_count": 1,
                "tsv_sha256": _sha256_hex(tsv_content),
                "tsv_gzip_b64": encoded,
            },
        )
        manifest_registry.backend.find_instance_by_external_id.side_effect = [row, row]

        saved = manifest_registry.get_manifest(customer_id="cust-001", manifest_id="m-1")
        assert isinstance(saved, SavedManifest)
        assert saved.manifest_id == "m-1"
        assert saved.customer_id == "cust-001"
        assert saved.sample_count == 1
        assert saved.tsv_sha256 == _sha256_hex(tsv_content)

        round_tripped = manifest_registry.get_manifest_tsv(customer_id="cust-001", manifest_id="m-1")
        assert round_tripped == tsv_content

    def test_get_manifest_returns_none_when_not_found(self, manifest_registry: ManifestRegistry):
        manifest_registry.backend.find_instance_by_external_id.return_value = None
        result = manifest_registry.get_manifest(customer_id="cust-001", manifest_id="m-missing")
        assert result is None

    def test_get_manifest_returns_none_for_customer_mismatch(self, manifest_registry: ManifestRegistry):
        row = MagicMock(
            euid="manifest-euid",
            name="Run 1",
            created_dt=None,
            modified_dt=None,
            bstatus="active",
            json_addl={
                "manifest_id": "m-1",
                "customer_id": "other-customer",
                "name": "Run 1",
                "tsv_gzip_b64": _gzip_b64_encode("RUN_ID\tSAMPLE_ID\nR0\tHG002\n"),
            },
        )
        manifest_registry.backend.find_instance_by_external_id.return_value = row
        assert manifest_registry.get_manifest(customer_id="cust-001", manifest_id="m-1") is None


class TestBootstrap:
    def test_bootstrap_ensures_templates(self, manifest_registry: ManifestRegistry):
        manifest_registry.bootstrap()
        manifest_registry.backend.ensure_templates.assert_called_once_with(manifest_registry._session)
