"""Graph-native tests for daylily_ursa.biospecimen."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from daylily_ursa.biospecimen import (
    Biosample,
    Biospecimen,
    BiospecimenRegistry,
    Library,
    Subject,
    generate_biosample_id,
    generate_biospecimen_id,
    generate_library_id,
    generate_subject_id,
)


class _SessionCtx:
    def __init__(self, session: MagicMock):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, exc_type, exc, tb):
        return False


def _instance(
    payload: dict, *, euid: str = "euid-1", template_uuid: int = 1, bstatus: str = "active"
):
    row = MagicMock()
    row.json_addl = dict(payload)
    row.euid = euid
    row.name = (
        payload.get("subject_id")
        or payload.get("biospecimen_id")
        or payload.get("biosample_id")
        or payload.get("library_id")
        or "node"
    )
    row.created_dt = None
    row.modified_dt = None
    row.bstatus = bstatus
    row.template_uuid = template_uuid
    row.uuid = hash(euid) & 0xFFFFFFFF
    row.is_deleted = False
    return row


@pytest.fixture
def registry() -> BiospecimenRegistry:
    reg = BiospecimenRegistry.__new__(BiospecimenRegistry)

    reg.backend = MagicMock()
    reg._session = MagicMock()
    reg.backend.session_scope.return_value = _SessionCtx(reg._session)
    return reg


@pytest.fixture
def sample_subject() -> Subject:
    return Subject(
        subject_id=generate_subject_id("customer-123", "patient-001"),
        customer_id="customer-123",
        display_name="John Doe",
        sex="male",
        tags=["tumor"],
    )


@pytest.fixture
def sample_biospecimen(sample_subject: Subject) -> Biospecimen:
    return Biospecimen(
        biospecimen_id=generate_biospecimen_id("customer-123", "biospecimen-001"),
        customer_id="customer-123",
        subject_id=sample_subject.subject_id,
        biospecimen_type="tissue",
        is_tumor=True,
    )


@pytest.fixture
def sample_biosample(sample_subject: Subject, sample_biospecimen: Biospecimen) -> Biosample:
    return Biosample(
        biosample_id=generate_biosample_id("customer-123", "biosample-001"),
        customer_id="customer-123",
        biospecimen_id=sample_biospecimen.biospecimen_id,
        subject_id=sample_subject.subject_id,
    )


@pytest.fixture
def sample_library(sample_biosample: Biosample) -> Library:
    return Library(
        library_id=generate_library_id("customer-123", "library-001"),
        customer_id="customer-123",
        biosample_id=sample_biosample.biosample_id,
    )


def test_id_generation_is_stable_and_prefixed():
    sid1 = generate_subject_id("cust", "subj")
    sid2 = generate_subject_id("cust", "subj")
    assert sid1 == sid2 and sid1.startswith("subj-")

    assert generate_biospecimen_id("cust", "bspec").startswith("bspec-")
    assert generate_biosample_id("cust", "bio").startswith("bio-")
    assert generate_library_id("cust", "lib").startswith("lib-")


def test_create_subject_creates_customer_lineage(
    registry: BiospecimenRegistry, sample_subject: Subject
):
    customer = _instance({"customer_id": sample_subject.customer_id}, euid="cust-euid")
    subject_row = _instance({"subject_id": sample_subject.subject_id}, euid="subj-euid")

    registry.backend.find_instance_by_external_id.side_effect = [None, None]
    registry.backend.create_instance.side_effect = [customer, subject_row]

    ok = registry.create_subject(sample_subject)

    assert ok is True
    assert registry.backend.create_instance.call_count == 2
    registry.backend.create_lineage.assert_called_once()


def test_get_subject_returns_dataclass(registry: BiospecimenRegistry, sample_subject: Subject):
    registry.backend.find_instance_by_external_id.return_value = _instance(
        sample_subject.__dict__, euid="subj-euid"
    )

    subject = registry.get_subject(sample_subject.subject_id)

    assert subject is not None
    assert subject.subject_id == sample_subject.subject_id
    assert subject.customer_id == sample_subject.customer_id


def test_create_biospecimen_requires_subject(
    registry: BiospecimenRegistry, sample_biospecimen: Biospecimen
):
    registry.backend.find_instance_by_external_id.return_value = None
    assert registry.create_biospecimen(sample_biospecimen) is False


def test_create_biospecimen_success(registry: BiospecimenRegistry, sample_biospecimen: Biospecimen):
    subject_row = _instance({"subject_id": sample_biospecimen.subject_id}, euid="subj-euid")
    bspec_row = _instance({"biospecimen_id": sample_biospecimen.biospecimen_id}, euid="bspec-euid")

    registry.backend.find_instance_by_external_id.side_effect = [subject_row, None]
    registry.backend.create_instance.return_value = bspec_row

    ok = registry.create_biospecimen(sample_biospecimen)

    assert ok is True
    registry.backend.create_lineage.assert_called_once()


def test_create_biosample_and_library(
    registry: BiospecimenRegistry, sample_biosample: Biosample, sample_library: Library
):
    bspec_row = _instance({"biospecimen_id": sample_biosample.biospecimen_id}, euid="bspec-euid")
    biosample_row = _instance({"biosample_id": sample_biosample.biosample_id}, euid="bio-euid")
    library_row = _instance({"library_id": sample_library.library_id}, euid="lib-euid")

    # create_biosample: find biospecimen, then find biosample(existing)
    registry.backend.find_instance_by_external_id.side_effect = [bspec_row, None]
    registry.backend.create_instance.return_value = biosample_row
    assert registry.create_biosample(sample_biosample) is True

    # create_library: find biosample, then find library(existing)
    registry.backend.find_instance_by_external_id.side_effect = [biosample_row, None]
    registry.backend.create_instance.return_value = library_row
    assert registry.create_library(sample_library) is True


def test_list_biospecimens_for_subject_filters_children(
    registry: BiospecimenRegistry, sample_subject: Subject, sample_biospecimen: Biospecimen
):
    subject_row = _instance({"subject_id": sample_subject.subject_id}, euid="subj-euid")
    valid = _instance(sample_biospecimen.__dict__, euid="bspec-euid")
    invalid = _instance({"other": "value"}, euid="x-euid")

    registry.backend.find_instance_by_external_id.return_value = subject_row
    registry.backend.list_children.return_value = [valid, invalid]

    biospecimens = registry.list_biospecimens_for_subject(sample_subject.subject_id)

    assert len(biospecimens) == 1
    assert biospecimens[0].biospecimen_id == sample_biospecimen.biospecimen_id


def test_get_subject_hierarchy_and_stats(
    registry: BiospecimenRegistry,
    sample_subject: Subject,
    sample_biospecimen: Biospecimen,
    sample_biosample: Biosample,
    sample_library: Library,
):
    registry.get_subject = MagicMock(return_value=sample_subject)
    registry.list_biospecimens_for_subject = MagicMock(return_value=[sample_biospecimen])
    registry.list_biosamples_for_biospecimen = MagicMock(return_value=[sample_biosample])
    registry.list_libraries_for_biosample = MagicMock(return_value=[sample_library])

    hierarchy = registry.get_subject_hierarchy(sample_subject.subject_id)

    assert hierarchy["subject"]["subject_id"] == sample_subject.subject_id
    assert len(hierarchy["biospecimens"]) == 1
    assert (
        hierarchy["biospecimens"][0]["biospecimen"]["biospecimen_id"]
        == sample_biospecimen.biospecimen_id
    )

    registry.list_subjects = MagicMock(return_value=[sample_subject])
    registry.list_biospecimens = MagicMock(return_value=[sample_biospecimen])
    registry.list_biosamples = MagicMock(return_value=[sample_biosample])
    registry.list_libraries = MagicMock(return_value=[sample_library])

    stats = registry.get_statistics(sample_subject.customer_id)
    assert stats == {"subjects": 1, "biospecimens": 1, "biosamples": 1, "libraries": 1}
