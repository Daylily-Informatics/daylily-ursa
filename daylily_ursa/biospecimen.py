"""Biospecimen hierarchy backed by TapDB graph objects."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from daylily_ursa.tapdb_graph import TapDBBackend, from_json_addl, utc_now_iso


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class Subject:
    subject_id: str
    customer_id: str
    display_name: Optional[str] = None
    sex: Optional[str] = None
    date_of_birth: Optional[str] = None
    species: str = "Homo sapiens"
    cohort: Optional[str] = None
    external_ids: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)


@dataclass
class Biospecimen:
    biospecimen_id: str
    customer_id: str
    subject_id: str
    biospecimen_type: str = "tissue"
    collection_date: Optional[str] = None
    collection_method: Optional[str] = None
    preservation_method: Optional[str] = None
    tissue_type: Optional[str] = None
    anatomical_site: Optional[str] = None
    tumor_fraction: Optional[float] = None
    is_tumor: bool = False
    produced_date: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)


@dataclass
class Biosample:
    biosample_id: str
    customer_id: str
    biospecimen_id: str
    subject_id: str
    sample_type: str = "blood"
    tissue_type: Optional[str] = None
    anatomical_site: Optional[str] = None
    collection_date: Optional[str] = None
    collection_method: Optional[str] = None
    preservation_method: Optional[str] = None
    tumor_fraction: Optional[float] = None
    tumor_grade: Optional[str] = None
    tumor_stage: Optional[str] = None
    is_tumor: bool = False
    matched_normal_id: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)


@dataclass
class Library:
    library_id: str
    customer_id: str
    biosample_id: str
    library_prep: str = "pcr_free_wgs"
    library_kit: Optional[str] = None
    target_insert_size: Optional[int] = None
    capture_kit: Optional[str] = None
    target_regions_bed: Optional[str] = None
    target_coverage: Optional[float] = None
    target_read_count: Optional[int] = None
    protocol_id: Optional[str] = None
    lab_id: Optional[str] = None
    prep_date: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)


def generate_subject_id(customer_id: str, identifier: str) -> str:
    return f"subj-{hashlib.sha256(f'{customer_id}:subject:{identifier}'.encode()).hexdigest()[:16]}"


def generate_biospecimen_id(customer_id: str, identifier: str) -> str:
    return f"bspec-{hashlib.sha256(f'{customer_id}:biospecimen:{identifier}'.encode()).hexdigest()[:12]}"


def generate_biosample_id(customer_id: str, identifier: str) -> str:
    return f"bio-{hashlib.sha256(f'{customer_id}:biosample:{identifier}'.encode()).hexdigest()[:16]}"


def generate_library_id(customer_id: str, identifier: str) -> str:
    return f"lib-{hashlib.sha256(f'{customer_id}:library:{identifier}'.encode()).hexdigest()[:16]}"


class BiospecimenRegistry:
    """TapDB-backed biospecimen hierarchy registry."""

    CUSTOMER_TEMPLATE = "actor/customer/account/1.0/"
    SUBJECT_TEMPLATE = "subject/person/participant/1.0/"
    BIOSPECIMEN_TEMPLATE = "content/biospecimen/entity/1.0/"
    BIOSAMPLE_TEMPLATE = "content/biosample/entity/1.0/"
    LIBRARY_TEMPLATE = "content/library/entity/1.0/"

    def __init__(
        self,
    ):
        self.backend = TapDBBackend(app_username="ursa-biospecimen")

    def bootstrap(self) -> None:
        with self.backend.session_scope(commit=True) as session:
            self.backend.ensure_templates(session)

    def _ensure_customer(self, session, customer_id: str):
        customer = self.backend.find_instance_by_external_id(
            session,
            template_code=self.CUSTOMER_TEMPLATE,
            key="customer_id",
            value=customer_id,
        )
        if customer is None:
            customer = self.backend.create_instance(
                session,
                template_code=self.CUSTOMER_TEMPLATE,
                name=customer_id,
                json_addl={
                    "customer_id": customer_id,
                    "customer_name": customer_id,
                    "email": "",
                    "s3_bucket": "",
                    "created_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                },
            )
        return customer

    def _create_or_update(self, session, *, template_code: str, key: str, value: str, name: str, payload: Dict[str, Any]):
        row = self.backend.find_instance_by_external_id(
            session,
            template_code=template_code,
            key=key,
            value=value,
        )
        if row is None:
            row = self.backend.create_instance(
                session,
                template_code=template_code,
                name=name,
                json_addl=payload,
                bstatus="active",
            )
        else:
            self.backend.update_instance_json(session, row, payload)
        return row

    @staticmethod
    def _to_subject(payload: Dict[str, Any]) -> Subject:
        return Subject(**{k: payload.get(k) for k in Subject.__dataclass_fields__.keys()})

    @staticmethod
    def _to_biospecimen(payload: Dict[str, Any]) -> Biospecimen:
        return Biospecimen(**{k: payload.get(k) for k in Biospecimen.__dataclass_fields__.keys()})

    @staticmethod
    def _to_biosample(payload: Dict[str, Any]) -> Biosample:
        return Biosample(**{k: payload.get(k) for k in Biosample.__dataclass_fields__.keys()})

    @staticmethod
    def _to_library(payload: Dict[str, Any]) -> Library:
        return Library(**{k: payload.get(k) for k in Library.__dataclass_fields__.keys()})

    def create_subject(self, subject: Subject) -> bool:
        payload = asdict(subject)
        payload["updated_at"] = utc_now_iso()
        with self.backend.session_scope(commit=True) as session:
            customer = self._ensure_customer(session, subject.customer_id)
            row = self._create_or_update(
                session,
                template_code=self.SUBJECT_TEMPLATE,
                key="subject_id",
                value=subject.subject_id,
                name=subject.display_name or subject.subject_id,
                payload=payload,
            )
            self.backend.create_lineage(session, parent=customer, child=row, relationship_type="owns")
            return True

    def get_subject(self, subject_id: str) -> Optional[Subject]:
        with self.backend.session_scope(commit=False) as session:
            row = self.backend.find_instance_by_external_id(session, template_code=self.SUBJECT_TEMPLATE, key="subject_id", value=subject_id)
            if row is None:
                return None
            return self._to_subject(from_json_addl(row))

    def update_subject(self, subject: Subject) -> bool:
        return self.create_subject(subject)

    def list_subjects(self, customer_id: str, limit: int = 100) -> List[Subject]:
        with self.backend.session_scope(commit=False) as session:
            customer = self.backend.find_instance_by_external_id(session, template_code=self.CUSTOMER_TEMPLATE, key="customer_id", value=customer_id)
            if customer is None:
                return []
            rows = self.backend.get_customer_owned(session, customer=customer, template_code=self.SUBJECT_TEMPLATE, relationship_type="owns", limit=limit)
            return [self._to_subject(from_json_addl(row)) for row in rows]

    def create_biospecimen(self, biospecimen: Biospecimen) -> bool:
        payload = asdict(biospecimen)
        payload["updated_at"] = utc_now_iso()
        with self.backend.session_scope(commit=True) as session:
            subject = self.backend.find_instance_by_external_id(session, template_code=self.SUBJECT_TEMPLATE, key="subject_id", value=biospecimen.subject_id)
            if subject is None:
                return False
            row = self._create_or_update(
                session,
                template_code=self.BIOSPECIMEN_TEMPLATE,
                key="biospecimen_id",
                value=biospecimen.biospecimen_id,
                name=biospecimen.biospecimen_id,
                payload=payload,
            )
            self.backend.create_lineage(session, parent=subject, child=row, relationship_type="has_biospecimen")
            return True

    def get_biospecimen(self, biospecimen_id: str) -> Optional[Biospecimen]:
        with self.backend.session_scope(commit=False) as session:
            row = self.backend.find_instance_by_external_id(session, template_code=self.BIOSPECIMEN_TEMPLATE, key="biospecimen_id", value=biospecimen_id)
            if row is None:
                return None
            return self._to_biospecimen(from_json_addl(row))

    def update_biospecimen(self, biospecimen: Biospecimen) -> bool:
        return self.create_biospecimen(biospecimen)

    def list_biospecimens_for_subject(self, subject_id: str) -> List[Biospecimen]:
        with self.backend.session_scope(commit=False) as session:
            subject = self.backend.find_instance_by_external_id(session, template_code=self.SUBJECT_TEMPLATE, key="subject_id", value=subject_id)
            if subject is None:
                return []
            children = self.backend.list_children(session, parent=subject, relationship_type="has_biospecimen")
            out = []
            for row in children:
                payload = from_json_addl(row)
                if payload.get("biospecimen_id"):
                    out.append(self._to_biospecimen(payload))
            return out

    def list_biospecimens(self, customer_id: str, limit: int = 100) -> List[Biospecimen]:
        out: List[Biospecimen] = []
        for subj in self.list_subjects(customer_id, limit=10000):
            out.extend(self.list_biospecimens_for_subject(subj.subject_id))
            if len(out) >= limit:
                break
        return out[:limit]

    def create_biosample(self, biosample: Biosample) -> bool:
        payload = asdict(biosample)
        payload["updated_at"] = utc_now_iso()
        with self.backend.session_scope(commit=True) as session:
            parent = self.backend.find_instance_by_external_id(session, template_code=self.BIOSPECIMEN_TEMPLATE, key="biospecimen_id", value=biosample.biospecimen_id)
            if parent is None:
                return False
            row = self._create_or_update(
                session,
                template_code=self.BIOSAMPLE_TEMPLATE,
                key="biosample_id",
                value=biosample.biosample_id,
                name=biosample.biosample_id,
                payload=payload,
            )
            self.backend.create_lineage(session, parent=parent, child=row, relationship_type="has_biosample")
            return True

    def get_biosample(self, biosample_id: str) -> Optional[Biosample]:
        with self.backend.session_scope(commit=False) as session:
            row = self.backend.find_instance_by_external_id(session, template_code=self.BIOSAMPLE_TEMPLATE, key="biosample_id", value=biosample_id)
            if row is None:
                return None
            return self._to_biosample(from_json_addl(row))

    def update_biosample(self, biosample: Biosample) -> bool:
        return self.create_biosample(biosample)

    def list_biosamples(self, customer_id: str, limit: int = 100) -> List[Biosample]:
        out: List[Biosample] = []
        for bspec in self.list_biospecimens(customer_id, limit=10000):
            out.extend(self.list_biosamples_for_biospecimen(bspec.biospecimen_id))
            if len(out) >= limit:
                break
        return out[:limit]

    def list_biosamples_for_subject(self, subject_id: str) -> List[Biosample]:
        out: List[Biosample] = []
        for bspec in self.list_biospecimens_for_subject(subject_id):
            out.extend(self.list_biosamples_for_biospecimen(bspec.biospecimen_id))
        return out

    def list_biosamples_for_biospecimen(self, biospecimen_id: str) -> List[Biosample]:
        with self.backend.session_scope(commit=False) as session:
            parent = self.backend.find_instance_by_external_id(session, template_code=self.BIOSPECIMEN_TEMPLATE, key="biospecimen_id", value=biospecimen_id)
            if parent is None:
                return []
            children = self.backend.list_children(session, parent=parent, relationship_type="has_biosample")
            out = []
            for row in children:
                payload = from_json_addl(row)
                if payload.get("biosample_id"):
                    out.append(self._to_biosample(payload))
            return out

    def create_library(self, library: Library) -> bool:
        payload = asdict(library)
        payload["updated_at"] = utc_now_iso()
        with self.backend.session_scope(commit=True) as session:
            parent = self.backend.find_instance_by_external_id(session, template_code=self.BIOSAMPLE_TEMPLATE, key="biosample_id", value=library.biosample_id)
            if parent is None:
                return False
            row = self._create_or_update(
                session,
                template_code=self.LIBRARY_TEMPLATE,
                key="library_id",
                value=library.library_id,
                name=library.library_id,
                payload=payload,
            )
            self.backend.create_lineage(session, parent=parent, child=row, relationship_type="has_library")
            return True

    def get_library(self, library_id: str) -> Optional[Library]:
        with self.backend.session_scope(commit=False) as session:
            row = self.backend.find_instance_by_external_id(session, template_code=self.LIBRARY_TEMPLATE, key="library_id", value=library_id)
            if row is None:
                return None
            return self._to_library(from_json_addl(row))

    def update_library(self, library: Library) -> bool:
        return self.create_library(library)

    def list_libraries(self, customer_id: str, limit: int = 100) -> List[Library]:
        out: List[Library] = []
        for biosample in self.list_biosamples(customer_id, limit=10000):
            out.extend(self.list_libraries_for_biosample(biosample.biosample_id))
            if len(out) >= limit:
                break
        return out[:limit]

    def list_libraries_for_biosample(self, biosample_id: str) -> List[Library]:
        with self.backend.session_scope(commit=False) as session:
            parent = self.backend.find_instance_by_external_id(session, template_code=self.BIOSAMPLE_TEMPLATE, key="biosample_id", value=biosample_id)
            if parent is None:
                return []
            children = self.backend.list_children(session, parent=parent, relationship_type="has_library")
            out = []
            for row in children:
                payload = from_json_addl(row)
                if payload.get("library_id"):
                    out.append(self._to_library(payload))
            return out

    def get_subject_hierarchy(self, subject_id: str) -> Dict[str, Any]:
        subject = self.get_subject(subject_id)
        if subject is None:
            return {}
        biospecimens = self.list_biospecimens_for_subject(subject_id)
        hierarchy = {
            "subject": asdict(subject),
            "biospecimens": [],
        }
        for bspec in biospecimens:
            biosamples = self.list_biosamples_for_biospecimen(bspec.biospecimen_id)
            libs: List[Dict[str, Any]] = []
            for bs in biosamples:
                libs.append(
                    {
                        "biosample": asdict(bs),
                        "libraries": [asdict(lib) for lib in self.list_libraries_for_biosample(bs.biosample_id)],
                    }
                )
            hierarchy["biospecimens"].append({"biospecimen": asdict(bspec), "biosamples": libs})
        return hierarchy

    def get_statistics(self, customer_id: str) -> Dict[str, int]:
        subjects = self.list_subjects(customer_id, limit=100000)
        biospecimens = self.list_biospecimens(customer_id, limit=100000)
        biosamples = self.list_biosamples(customer_id, limit=100000)
        libraries = self.list_libraries(customer_id, limit=100000)
        return {
            "subjects": len(subjects),
            "biospecimens": len(biospecimens),
            "biosamples": len(biosamples),
            "libraries": len(libraries),
        }
