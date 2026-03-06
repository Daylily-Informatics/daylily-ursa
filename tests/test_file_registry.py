"""Graph-native tests for daylily_ursa.file_registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from daylily_ursa.file_registry import (
    BiosampleMetadata,
    FileMetadata,
    FileRegistration,
    FileRegistry,
    FileSet,
    FileWorksetUsage,
    SequencingMetadata,
    detect_file_format,
    generate_file_id,
)


class _SessionCtx:
    def __init__(self, session: MagicMock):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, exc_type, exc, tb):
        return False


def _instance(payload: dict, *, euid: str = "euid-1", template_uuid: int = 1, bstatus: str = "active"):
    row = MagicMock()
    row.json_addl = dict(payload)
    row.euid = euid
    row.name = payload.get("file_id") or payload.get("fileset_id") or "node"
    row.created_dt = None
    row.modified_dt = None
    row.bstatus = bstatus
    row.template_uuid = template_uuid
    row.uuid = hash(euid) & 0xFFFFFFFF
    row.is_deleted = False
    return row


@pytest.fixture
def file_registry() -> FileRegistry:
    reg = FileRegistry.__new__(FileRegistry)

    reg.backend = MagicMock()
    reg._session = MagicMock()
    reg.backend.session_scope.return_value = _SessionCtx(reg._session)
    reg.backend.templates.get_template.return_value = MagicMock(uuid=1)
    return reg


@pytest.fixture
def sample_registration() -> FileRegistration:
    return FileRegistration(
        file_id="file-001",
        customer_id="cust-001",
        file_metadata=FileMetadata(
            file_id="file-001",
            s3_uri="s3://bucket/sample_R1.fastq.gz",
            file_size_bytes=1024,
        ),
        sequencing_metadata=SequencingMetadata(run_id="run-001"),
        biosample_metadata=BiosampleMetadata(biosample_id="bio-001", subject_id="subj-001"),
        tags=["wgs"],
    )


def test_detect_file_format_and_id_generation():
    assert detect_file_format("s3://bucket/sample.fastq.gz") == "fastq"
    assert detect_file_format("s3://bucket/sample.unknown") == "unknown"

    fid1 = generate_file_id("s3://bucket/a/sample.fastq.gz", "cust-001")
    fid2 = generate_file_id("s3://bucket/a/sample.fastq.gz", "cust-001")
    fid3 = generate_file_id("s3://bucket/b/sample.fastq.gz", "cust-001")

    assert fid1 == fid2
    assert fid1 != fid3
    assert fid1.startswith("file-")


def test_register_file_creates_customer_and_file(file_registry: FileRegistry, sample_registration: FileRegistration):
    customer_row = _instance({"customer_id": "cust-001"}, euid="cust-euid")
    file_row = _instance({"file_id": "file-001", "customer_id": "cust-001"}, euid="file-euid")

    file_registry.backend.find_instance_by_external_id.side_effect = [None, None]
    file_registry.backend.create_instance.side_effect = [customer_row, file_row]

    euid = file_registry.register_file(sample_registration)

    assert euid == "file-euid"
    assert file_registry.backend.create_instance.call_count == 2
    payload = file_registry.backend.create_instance.call_args_list[1].kwargs["json_addl"]
    assert payload["file_id"] == "file-001"
    assert payload["customer_id"] == "cust-001"
    assert payload["file_metadata"]["s3_uri"] == "s3://bucket/sample_R1.fastq.gz"
    file_registry.backend.create_lineage.assert_called_once()


def test_register_file_updates_existing(file_registry: FileRegistry, sample_registration: FileRegistration):
    existing = _instance({"file_id": "file-001"}, euid="file-old")
    file_registry.backend.find_instance_by_external_id.return_value = existing

    euid = file_registry.register_file(sample_registration)

    assert euid == "file-old"
    file_registry.backend.update_instance_json.assert_called_once()
    file_registry.backend.create_instance.assert_not_called()


def test_get_file_returns_registration(file_registry: FileRegistry):
    payload = {
        "file_id": "file-001",
        "customer_id": "cust-001",
        "file_metadata": {
            "file_id": "file-001",
            "s3_uri": "s3://bucket/sample_R1.fastq.gz",
            "file_size_bytes": 1024,
            "file_format": "fastq",
            "md5_checksum": None,
            "created_at": "2026-01-01T00:00:00Z",
        },
        "sequencing_metadata": {
            "platform": "ILLUMINA_NOVASEQ_X",
            "vendor": "ILMN",
            "run_id": "run-001",
            "lane": 1,
            "barcode_id": "S1",
            "flowcell_id": None,
            "run_date": None,
        },
        "biosample_metadata": {
            "biosample_id": "bio-001",
            "subject_id": "subj-001",
            "sample_type": "blood",
            "tissue_type": None,
            "collection_date": None,
            "preservation_method": None,
            "tumor_fraction": None,
        },
        "tags": ["wgs"],
    }
    file_registry.backend.find_instance_by_external_id.return_value = _instance(payload, euid="file-euid")

    reg = file_registry.get_file("file-001")

    assert reg is not None
    assert reg.file_id == "file-001"
    assert reg.file_euid == "file-euid"
    assert reg.customer_id == "cust-001"
    assert reg.file_metadata.s3_uri == "s3://bucket/sample_R1.fastq.gz"
    assert reg.tags == ["wgs"]


def test_list_customer_files(file_registry: FileRegistry):
    customer = _instance({"customer_id": "cust-001"}, euid="cust-euid")
    row = _instance(
        {
            "file_id": "file-001",
            "customer_id": "cust-001",
            "file_metadata": {
                "file_id": "file-001",
                "s3_uri": "s3://bucket/sample_R1.fastq.gz",
                "file_size_bytes": 1,
                "file_format": "fastq",
                "created_at": "2026-01-01T00:00:00Z",
            },
            "sequencing_metadata": {
                "platform": "ILLUMINA_NOVASEQ_X",
                "vendor": "ILMN",
                "run_id": "run-001",
                "lane": 1,
                "barcode_id": "S1",
                "flowcell_id": None,
                "run_date": None,
            },
            "biosample_metadata": {
                "biosample_id": "bio-001",
                "subject_id": "subj-001",
                "sample_type": "blood",
                "tissue_type": None,
                "collection_date": None,
                "preservation_method": None,
                "tumor_fraction": None,
            },
            "tags": [],
        },
        euid="file-euid",
    )
    file_registry.backend.find_instance_by_external_id.return_value = customer
    file_registry.backend.get_customer_owned.return_value = [row]

    files = file_registry.list_customer_files("cust-001")

    assert len(files) == 1
    assert files[0].file_id == "file-001"


def test_update_file_tags(file_registry: FileRegistry):
    row = _instance({"file_id": "file-001", "customer_id": "cust-001", "tags": []}, euid="file-euid")
    file_registry.backend.find_instance_by_external_id.return_value = row

    ok = file_registry.update_file_tags("file-001", ["tumor", "wgs"])

    assert ok is True
    updated_payload = file_registry.backend.update_instance_json.call_args.args[2]
    assert updated_payload["tags"] == ["tumor", "wgs"]


def test_create_fileset_links_customer_and_files(file_registry: FileRegistry):
    customer = _instance({"customer_id": "cust-001"}, euid="cust-euid")
    fileset_row = _instance({"fileset_id": "fs-001", "customer_id": "cust-001"}, euid="fs-euid")
    file_row_1 = _instance({"file_id": "file-001"}, euid="file-001")
    file_row_2 = _instance({"file_id": "file-002"}, euid="file-002")

    file_registry.backend.find_instance_by_external_id.side_effect = [
        None,   # existing fileset
        None,   # customer
        file_row_1,
        file_row_2,
    ]
    file_registry.backend.create_instance.side_effect = [customer, fileset_row]

    fileset = FileSet(
        fileset_id="fs-001",
        customer_id="cust-001",
        name="Tumor set",
        file_ids=["file-001", "file-002"],
    )

    euid = file_registry.create_fileset(fileset)

    assert euid == "fs-euid"
    assert file_registry.backend.create_lineage.call_count == 3


def test_get_file_by_euid(file_registry: FileRegistry):
    payload = {
        "file_id": "file-001",
        "customer_id": "cust-001",
        "file_metadata": {
            "file_id": "file-001",
            "s3_uri": "s3://bucket/sample_R1.fastq.gz",
            "file_size_bytes": 1024,
            "file_format": "fastq",
            "md5_checksum": None,
            "created_at": "2026-01-01T00:00:00Z",
        },
        "sequencing_metadata": {
            "platform": "ILLUMINA_NOVASEQ_X", "vendor": "ILMN", "run_id": "run-001",
            "lane": 1, "barcode_id": "S1", "flowcell_id": None, "run_date": None,
        },
        "biosample_metadata": {
            "biosample_id": "bio-001", "subject_id": "subj-001", "sample_type": "blood",
            "tissue_type": None, "collection_date": None, "preservation_method": None, "tumor_fraction": None,
        },
        "tags": [],
    }
    row = _instance(payload, euid="file-euid-123")
    file_registry.backend.find_instance_by_euid.return_value = row

    reg = file_registry.get_file_by_euid("file-euid-123")

    assert reg is not None
    assert reg.file_euid == "file-euid-123"
    assert reg.file_id == "file-001"
    file_registry.backend.find_instance_by_euid.assert_called_once()


def test_get_file_by_euid_not_found(file_registry: FileRegistry):
    file_registry.backend.find_instance_by_euid.return_value = None
    assert file_registry.get_file_by_euid("no-such-euid") is None


def test_get_fileset_by_euid(file_registry: FileRegistry):
    payload = {
        "fileset_id": "fs-001",
        "customer_id": "cust-001",
        "name": "Test set",
        "file_ids": ["file-001"],
        "tags": [],
    }
    row = _instance(payload, euid="fs-euid-456")
    file_registry.backend.find_instance_by_euid.return_value = row

    fs = file_registry.get_fileset_by_euid("fs-euid-456")

    assert fs is not None
    assert fs.fileset_euid == "fs-euid-456"
    assert fs.fileset_id == "fs-001"


def test_get_fileset_by_euid_not_found(file_registry: FileRegistry):
    file_registry.backend.find_instance_by_euid.return_value = None
    assert file_registry.get_fileset_by_euid("no-such-euid") is None


def test_register_file_workset_usage_updates_history(file_registry: FileRegistry):
    file_row = _instance(
        {
            "file_id": "file-001",
            "customer_id": "cust-001",
            "workset_usage": [],
            "file_metadata": {
                "file_id": "file-001",
                "s3_uri": "s3://bucket/sample_R1.fastq.gz",
                "file_size_bytes": 1,
                "file_format": "fastq",
                "created_at": "2026-01-01T00:00:00Z",
            },
            "sequencing_metadata": {
                "platform": "ILLUMINA_NOVASEQ_X",
                "vendor": "ILMN",
                "run_id": "run-001",
                "lane": 1,
                "barcode_id": "S1",
                "flowcell_id": None,
                "run_date": None,
            },
            "biosample_metadata": {
                "biosample_id": "bio-001",
                "subject_id": "subj-001",
                "sample_type": "blood",
                "tissue_type": None,
                "collection_date": None,
                "preservation_method": None,
                "tumor_fraction": None,
            },
        },
        euid="file-euid",
    )
    workset_row = _instance({"workset_id": "ws-001"}, euid="ws-euid")

    file_registry.backend.find_instance_by_external_id.side_effect = [file_row, workset_row]

    ok = file_registry.register_file_workset_usage(
        file_id="file-001",
        workset_id="ws-001",
        customer_id="cust-001",
        usage_type="input",
        workset_state="ready",
    )

    assert ok is True
    payload = file_registry.backend.update_instance_json.call_args.args[2]
    assert len(payload["workset_usage"]) == 1
    assert payload["workset_usage"][0]["workset_id"] == "ws-001"
    file_registry.backend.create_lineage.assert_called_once()


def test_get_file_workset_history(file_registry: FileRegistry):
    row = _instance(
        {
            "file_id": "file-001",
            "customer_id": "cust-001",
            "file_metadata": {
                "file_id": "file-001",
                "s3_uri": "s3://bucket/sample_R1.fastq.gz",
                "file_size_bytes": 1,
                "file_format": "fastq",
                "created_at": "2026-01-01T00:00:00Z",
            },
            "sequencing_metadata": {
                "platform": "ILLUMINA_NOVASEQ_X",
                "vendor": "ILMN",
                "run_id": "run-001",
                "lane": 1,
                "barcode_id": "S1",
                "flowcell_id": None,
                "run_date": None,
            },
            "biosample_metadata": {
                "biosample_id": "bio-001",
                "subject_id": "subj-001",
                "sample_type": "blood",
                "tissue_type": None,
                "collection_date": None,
                "preservation_method": None,
                "tumor_fraction": None,
            },
            "workset_usage": [
                {
                    "file_id": "file-001",
                    "workset_id": "ws-001",
                    "customer_id": "cust-001",
                    "usage_type": "input",
                    "added_at": "2026-01-01T00:00:00Z",
                    "workset_state": "ready",
                    "notes": None,
                }
            ],
        },
        euid="file-euid",
    )
    file_registry.backend.find_instance_by_external_id.return_value = row

    history = file_registry.get_file_workset_history("file-001")

    assert len(history) == 1
    assert isinstance(history[0], FileWorksetUsage)
    assert history[0].workset_id == "ws-001"


def test_get_files_for_workset_recreation_deduplicates(file_registry: FileRegistry):
    usage = FileWorksetUsage(
        file_id="file-001",
        workset_id="ws-001",
        customer_id="cust-001",
        usage_type="input",
    )
    out_reg = FileRegistration(
        file_id="file-001",
        customer_id="cust-001",
        file_metadata=FileMetadata(file_id="file-001", s3_uri="s3://bucket/f1.fastq.gz", file_size_bytes=1),
        sequencing_metadata=SequencingMetadata(run_id="run"),
        biosample_metadata=BiosampleMetadata(biosample_id="bio-001", subject_id="subj-001"),
    )
    file_registry.get_workset_files = MagicMock(return_value=[usage, usage])
    file_registry.get_file = MagicMock(return_value=out_reg)

    files = file_registry.get_files_for_workset_recreation("ws-001")

    assert len(files) == 1
    assert files[0].file_id == "file-001"
