"""File registry and fileset management backed by TapDB graph objects."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError
from daylily_ursa.tapdb_graph import TapDBBackend, from_json_addl, utc_now_iso


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


FILE_FORMAT_PATTERNS = {
    "fastq": [r"\.fastq(\.gz)?$", r"\.fq(\.gz)?$"],
    "bam": [r"\.bam$"],
    "cram": [r"\.cram$"],
    "vcf": [r"\.vcf(\.gz)?$", r"\.gvcf(\.gz)?$"],
    "bed": [r"\.bed(\.gz)?$"],
    "fasta": [r"\.fa(sta)?(\.gz)?$", r"\.fna(\.gz)?$"],
}

LOGGER = logging.getLogger("daylily.file_registry")


def detect_file_format(s3_uri: str) -> str:
    lower = s3_uri.lower()
    for fmt, patterns in FILE_FORMAT_PATTERNS.items():
        if any(re.search(pattern, lower) for pattern in patterns):
            return fmt
    return "unknown"


def generate_file_id(s3_uri: str, customer_id: str) -> str:
    digest = hashlib.sha256(f"{customer_id}:{s3_uri}".encode("utf-8")).hexdigest()
    return f"file-{digest[:16]}"


@dataclass
class FileMetadata:
    file_id: str
    s3_uri: str
    file_size_bytes: int
    md5_checksum: Optional[str] = None
    file_format: str = "fastq"
    created_at: str = field(default_factory=_utc_now_iso)

    @property
    def filename(self) -> str:
        return PurePosixPath(self.s3_uri).name


@dataclass
class SequencingMetadata:
    platform: str = "ILLUMINA_NOVASEQ_X"
    vendor: str = "ILMN"
    run_id: str = ""
    lane: int = 0
    barcode_id: str = "S1"
    flowcell_id: Optional[str] = None
    run_date: Optional[str] = None


@dataclass
class BiosampleMetadata:
    biosample_id: str
    subject_id: str
    sample_type: str = "blood"
    tissue_type: Optional[str] = None
    collection_date: Optional[str] = None
    preservation_method: Optional[str] = None
    tumor_fraction: Optional[float] = None


@dataclass
class FileRegistration:
    file_id: str
    customer_id: str
    file_metadata: FileMetadata
    sequencing_metadata: SequencingMetadata
    biosample_metadata: BiosampleMetadata
    paired_with: Optional[str] = None
    read_number: int = 1
    quality_score: Optional[float] = None
    percent_q30: Optional[float] = None
    concordance_vcf_path: Optional[str] = None
    is_positive_control: bool = False
    is_negative_control: bool = False
    tags: List[str] = field(default_factory=list)
    registered_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    file_euid: Optional[str] = None


@dataclass
class FileSet:
    fileset_id: str
    customer_id: str
    name: str
    description: Optional[str] = None
    biosample_metadata: Optional[BiosampleMetadata] = None
    sequencing_metadata: Optional[SequencingMetadata] = None
    file_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    fileset_euid: Optional[str] = None


@dataclass
class FileWorksetUsage:
    file_id: str
    workset_euid: str
    customer_id: str
    usage_type: str = "input"
    added_at: str = field(default_factory=_utc_now_iso)
    workset_state: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class DiscoveredFile:
    """A file discovered while scanning an S3 bucket."""

    s3_uri: str
    bucket_name: str
    key: str
    file_size_bytes: int
    last_modified: str
    etag: str
    detected_format: str
    is_registered: bool = False
    file_id: Optional[str] = None


class BucketFileDiscovery:
    """Discover and optionally auto-register files in S3 buckets."""

    def __init__(
        self,
        region: str = "us-west-2",
        profile: Optional[str] = None,
    ):
        session_kwargs: Dict[str, Any] = {"region_name": region}
        if profile:
            session_kwargs["profile_name"] = profile
        session = boto3.Session(**session_kwargs)
        self.s3 = session.client("s3")
        self.region = region

    def discover_files(
        self,
        bucket_name: str,
        prefix: str = "",
        file_formats: Optional[List[str]] = None,
        max_files: int = 1000,
    ) -> List[DiscoveredFile]:
        discovered: List[DiscoveredFile] = []
        paginator = self.s3.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key", "")
                    if not key or key.endswith("/"):
                        continue
                    detected = detect_file_format(key)
                    if file_formats and detected not in file_formats:
                        continue
                    last_modified = obj.get("LastModified")
                    discovered.append(
                        DiscoveredFile(
                            s3_uri=f"s3://{bucket_name}/{key}",
                            bucket_name=bucket_name,
                            key=key,
                            file_size_bytes=int(obj.get("Size", 0)),
                            last_modified=last_modified.isoformat() if hasattr(last_modified, "isoformat") else str(last_modified or ""),
                            etag=str(obj.get("ETag", "")).strip('"'),
                            detected_format=detected,
                        )
                    )
                    if len(discovered) >= max_files:
                        return discovered
        except ClientError as exc:
            LOGGER.error("Failed to discover files in %s: %s", bucket_name, exc)
        return discovered

    def check_registration_status(
        self,
        discovered_files: List[DiscoveredFile],
        registry: "FileRegistry",
        customer_id: str,
    ) -> List[DiscoveredFile]:
        for discovered in discovered_files:
            try:
                file_id = generate_file_id(discovered.s3_uri, customer_id)
                existing = registry.get_file(file_id)
                if existing:
                    discovered.is_registered = True
                    discovered.file_id = file_id
            except Exception as exc:
                LOGGER.warning(
                    "Failed registration check for %s: %s",
                    discovered.s3_uri,
                    exc,
                )
        return discovered_files

    def auto_register_files(
        self,
        discovered_files: List[DiscoveredFile],
        registry: "FileRegistry",
        customer_id: str,
        biosample_id: str,
        subject_id: str,
        sequencing_platform: str = "ILLUMINA_NOVASEQ_X",
    ) -> Tuple[int, int, List[str]]:
        registered = 0
        skipped = 0
        errors: List[str] = []
        for discovered in discovered_files:
            if discovered.is_registered:
                skipped += 1
                continue
            file_id = generate_file_id(discovered.s3_uri, customer_id)
            read_number = 2 if any(token in discovered.key for token in ("_R2", "_2.fastq", "_2.fq")) else 1
            registration = FileRegistration(
                file_id=file_id,
                customer_id=customer_id,
                file_metadata=FileMetadata(
                    file_id=file_id,
                    s3_uri=discovered.s3_uri,
                    file_size_bytes=discovered.file_size_bytes,
                    md5_checksum=discovered.etag or None,
                    file_format=discovered.detected_format,
                ),
                sequencing_metadata=SequencingMetadata(platform=sequencing_platform),
                biosample_metadata=BiosampleMetadata(
                    biosample_id=biosample_id,
                    subject_id=subject_id,
                ),
                read_number=read_number,
            )
            try:
                if registry.register_file(registration):
                    registered += 1
                    discovered.is_registered = True
                    discovered.file_id = file_id
                else:
                    skipped += 1
            except Exception as exc:
                errors.append(f"Failed to register {discovered.s3_uri}: {exc}")
        return registered, skipped, errors


class FileRegistry:
    """TapDB-backed file registry."""

    FILE_TEMPLATE = "file/object/registered/1.0/"
    FILESET_TEMPLATE = "container/fileset/group/1.0/"
    CUSTOMER_TEMPLATE = "actor/customer/account/1.0/"
    WORKSET_TEMPLATE = "workflow/workset/analysis/1.0/"

    def __init__(
        self,
    ):
        self.backend = TapDBBackend(app_username="ursa-file")

    def bootstrap(self) -> None:
        with self.backend.session_scope(commit=True) as session:
            self.backend.ensure_templates(session)

    @staticmethod
    def _norm_file_id(file_id: str) -> str:
        if not file_id:
            raise ValueError("file_id is required; use generate_file_id() to create a deterministic name")
        return file_id

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

    @staticmethod
    def _serialize_registration(reg: FileRegistration) -> Dict[str, Any]:
        return {
            "file_id": reg.file_id,
            "customer_id": reg.customer_id,
            "file_metadata": asdict(reg.file_metadata),
            "sequencing_metadata": asdict(reg.sequencing_metadata),
            "biosample_metadata": asdict(reg.biosample_metadata),
            "paired_with": reg.paired_with,
            "read_number": reg.read_number,
            "quality_score": reg.quality_score,
            "percent_q30": reg.percent_q30,
            "concordance_vcf_path": reg.concordance_vcf_path,
            "is_positive_control": reg.is_positive_control,
            "is_negative_control": reg.is_negative_control,
            "tags": list(reg.tags),
            "registered_at": reg.registered_at,
            "updated_at": reg.updated_at,
            "workset_usage": [],
        }

    @staticmethod
    def _to_registration(payload: Dict[str, Any]) -> FileRegistration:
        file_meta = payload.get("file_metadata") or {}
        seq_meta = payload.get("sequencing_metadata") or {}
        bio_meta = payload.get("biosample_metadata") or {}
        return FileRegistration(
            file_id=payload["file_id"],
            customer_id=payload["customer_id"],
            file_metadata=FileMetadata(**file_meta),
            sequencing_metadata=SequencingMetadata(**seq_meta),
            biosample_metadata=BiosampleMetadata(**bio_meta),
            paired_with=payload.get("paired_with"),
            read_number=int(payload.get("read_number", 1) or 1),
            quality_score=payload.get("quality_score"),
            percent_q30=payload.get("percent_q30"),
            concordance_vcf_path=payload.get("concordance_vcf_path"),
            is_positive_control=bool(payload.get("is_positive_control", False)),
            is_negative_control=bool(payload.get("is_negative_control", False)),
            tags=list(payload.get("tags", [])),
            registered_at=payload.get("registered_at", _utc_now_iso()),
            updated_at=payload.get("updated_at", _utc_now_iso()),
            file_euid=payload.get("euid"),
        )

    def register_file(self, registration: FileRegistration) -> Optional[str]:
        """Register a file. Returns the EUID of the created/updated file, or None on failure."""
        registration.file_id = self._norm_file_id(registration.file_id)
        if not registration.file_metadata.file_format:
            registration.file_metadata.file_format = detect_file_format(registration.file_metadata.s3_uri)

        with self.backend.session_scope(commit=True) as session:
            existing = self.backend.find_instance_by_external_id(
                session,
                template_code=self.FILE_TEMPLATE,
                key="file_id",
                value=registration.file_id,
            )
            payload = self._serialize_registration(registration)
            payload["updated_at"] = utc_now_iso()
            if existing is not None:
                self.backend.update_instance_json(session, existing, payload)
                return existing.euid

            customer = self._ensure_customer(session, registration.customer_id)
            row = self.backend.create_instance(
                session,
                template_code=self.FILE_TEMPLATE,
                name=registration.file_metadata.filename,
                json_addl=payload,
                bstatus="active",
            )
            self.backend.create_lineage(
                session,
                parent=customer,
                child=row,
                relationship_type="owns",
            )
            return row.euid

    def get_file(self, file_id: str) -> Optional[FileRegistration]:
        with self.backend.session_scope(commit=False) as session:
            row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.FILE_TEMPLATE,
                key="file_id",
                value=file_id,
            )
            if row is None:
                return None
            reg = self._to_registration(from_json_addl(row))
            reg.file_euid = row.euid
            return reg

    def get_file_by_euid(self, euid: str) -> Optional[FileRegistration]:
        """Look up a file by its TapDB EUID."""
        with self.backend.session_scope(commit=False) as session:
            row = self.backend.find_instance_by_euid(session, euid)
            if row is None:
                return None
            reg = self._to_registration(from_json_addl(row))
            reg.file_euid = row.euid
            return reg

    def find_file_by_s3_uri(
        self,
        *,
        customer_id: str,
        s3_uri: str,
    ) -> Optional[FileRegistration]:
        needle = (s3_uri or "").strip().lower()
        for reg in self.list_customer_files(customer_id, limit=10000):
            if reg.file_metadata.s3_uri.strip().lower() == needle:
                return reg
        return None

    def list_customer_files(self, customer_id: str, limit: int = 100) -> List[FileRegistration]:
        with self.backend.session_scope(commit=False) as session:
            customer = self.backend.find_instance_by_external_id(
                session,
                template_code=self.CUSTOMER_TEMPLATE,
                key="customer_id",
                value=customer_id,
            )
            if customer is None:
                return []
            rows = self.backend.get_customer_owned(
                session,
                customer=customer,
                template_code=self.FILE_TEMPLATE,
                relationship_type="owns",
                limit=limit,
            )
            return [self._to_registration(from_json_addl(row)) for row in rows]

    def update_file(self, file_id: str, updates: Dict[str, Any]) -> bool:
        with self.backend.session_scope(commit=True) as session:
            row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.FILE_TEMPLATE,
                key="file_id",
                value=file_id,
            )
            if row is None:
                return False
            payload = from_json_addl(row)
            payload.update(updates)
            payload["updated_at"] = utc_now_iso()
            self.backend.update_instance_json(session, row, payload)
            return True

    def update_file_tags(self, file_id: str, tags: List[str]) -> bool:
        return self.update_file(file_id, {"tags": list(tags)})

    def search_files_by_tag(self, customer_id: str, tag: str) -> List[FileRegistration]:
        out = []
        needle = tag.strip().lower()
        for reg in self.list_customer_files(customer_id, limit=10000):
            tags = [t.lower() for t in reg.tags]
            if needle in tags:
                out.append(reg)
        return out

    def search_files_by_biosample(self, customer_id: str, biosample_id: str) -> List[FileRegistration]:
        out = []
        for reg in self.list_customer_files(customer_id, limit=10000):
            if reg.biosample_metadata.biosample_id == biosample_id:
                out.append(reg)
        return out

    @staticmethod
    def _serialize_fileset(fileset: FileSet) -> Dict[str, Any]:
        return {
            "fileset_id": fileset.fileset_id,
            "customer_id": fileset.customer_id,
            "name": fileset.name,
            "description": fileset.description,
            "biosample_metadata": asdict(fileset.biosample_metadata) if fileset.biosample_metadata else None,
            "sequencing_metadata": asdict(fileset.sequencing_metadata) if fileset.sequencing_metadata else None,
            "file_ids": list(fileset.file_ids),
            "tags": list(fileset.tags),
            "created_at": fileset.created_at,
            "updated_at": fileset.updated_at,
        }

    @staticmethod
    def _to_fileset(payload: Dict[str, Any]) -> FileSet:
        biosample_metadata = payload.get("biosample_metadata")
        sequencing_metadata = payload.get("sequencing_metadata")
        return FileSet(
            fileset_id=payload["fileset_id"],
            customer_id=payload["customer_id"],
            name=payload.get("name", payload["fileset_id"]),
            description=payload.get("description"),
            biosample_metadata=BiosampleMetadata(**biosample_metadata) if biosample_metadata else None,
            sequencing_metadata=SequencingMetadata(**sequencing_metadata) if sequencing_metadata else None,
            file_ids=list(payload.get("file_ids", [])),
            tags=list(payload.get("tags", [])),
            created_at=payload.get("created_at", _utc_now_iso()),
            updated_at=payload.get("updated_at", _utc_now_iso()),
            fileset_euid=payload.get("euid"),
        )

    def create_fileset(self, fileset: FileSet) -> Optional[str]:
        """Create or update a fileset. Returns the EUID of the fileset, or None on failure."""
        if not fileset.fileset_id:
            fileset.fileset_id = fileset.name  # use the human-readable name as the legacy id
        payload = self._serialize_fileset(fileset)
        with self.backend.session_scope(commit=True) as session:
            existing = self.backend.find_instance_by_external_id(
                session,
                template_code=self.FILESET_TEMPLATE,
                key="fileset_id",
                value=fileset.fileset_id,
            )
            if existing is not None:
                self.backend.update_instance_json(session, existing, payload)
                return existing.euid
            customer = self._ensure_customer(session, fileset.customer_id)
            row = self.backend.create_instance(
                session,
                template_code=self.FILESET_TEMPLATE,
                name=fileset.name,
                json_addl=payload,
                bstatus="active",
            )
            self.backend.create_lineage(session, parent=customer, child=row, relationship_type="owns")
            for fid in fileset.file_ids:
                file_row = self.backend.find_instance_by_external_id(
                    session,
                    template_code=self.FILE_TEMPLATE,
                    key="file_id",
                    value=fid,
                )
                if file_row is not None:
                    self.backend.create_lineage(session, parent=row, child=file_row, relationship_type="contains")
            return row.euid

    def get_fileset(self, fileset_id: str) -> Optional[FileSet]:
        with self.backend.session_scope(commit=False) as session:
            row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.FILESET_TEMPLATE,
                key="fileset_id",
                value=fileset_id,
            )
            if row is None:
                return None
            fs = self._to_fileset(from_json_addl(row))
            fs.fileset_euid = row.euid
            return fs

    def get_fileset_by_euid(self, euid: str) -> Optional[FileSet]:
        """Look up a fileset by its TapDB EUID."""
        with self.backend.session_scope(commit=False) as session:
            row = self.backend.find_instance_by_euid(session, euid)
            if row is None:
                return None
            fs = self._to_fileset(from_json_addl(row))
            fs.fileset_euid = row.euid
            return fs

    def list_customer_filesets(self, customer_id: str) -> List[FileSet]:
        with self.backend.session_scope(commit=False) as session:
            customer = self.backend.find_instance_by_external_id(
                session,
                template_code=self.CUSTOMER_TEMPLATE,
                key="customer_id",
                value=customer_id,
            )
            if customer is None:
                return []
            rows = self.backend.get_customer_owned(
                session,
                customer=customer,
                template_code=self.FILESET_TEMPLATE,
                relationship_type="owns",
                limit=10000,
            )
            return [self._to_fileset(from_json_addl(row)) for row in rows]

    def add_files_to_fileset(self, fileset_id: str, file_ids: List[str]) -> bool:
        fileset = self.get_fileset(fileset_id)
        if fileset is None:
            return False
        merged = list(dict.fromkeys(fileset.file_ids + file_ids))
        fileset.file_ids = merged
        fileset.updated_at = utc_now_iso()
        euid = self.create_fileset(fileset)
        if not euid:
            return False
        with self.backend.session_scope(commit=True) as session:
            fileset_row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.FILESET_TEMPLATE,
                key="fileset_id",
                value=fileset_id,
            )
            if fileset_row is None:
                return False
            for fid in file_ids:
                file_row = self.backend.find_instance_by_external_id(
                    session,
                    template_code=self.FILE_TEMPLATE,
                    key="file_id",
                    value=fid,
                )
                if file_row is not None:
                    self.backend.create_lineage(session, parent=fileset_row, child=file_row, relationship_type="contains")
        return True

    def remove_files_from_fileset(self, fileset_id: str, file_ids: List[str]) -> bool:
        fileset = self.get_fileset(fileset_id)
        if fileset is None:
            return False
        remove_set = set(file_ids)
        fileset.file_ids = [fid for fid in fileset.file_ids if fid not in remove_set]
        fileset.updated_at = utc_now_iso()
        return bool(self.create_fileset(fileset))

    def update_fileset_metadata(self, fileset_id: str, updates: Dict[str, Any]) -> bool:
        fileset = self.get_fileset(fileset_id)
        if fileset is None:
            return False
        payload = self._serialize_fileset(fileset)
        payload.update(updates)
        payload["updated_at"] = utc_now_iso()
        with self.backend.session_scope(commit=True) as session:
            row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.FILESET_TEMPLATE,
                key="fileset_id",
                value=fileset_id,
            )
            if row is None:
                return False
            self.backend.update_instance_json(session, row, payload)
            return True

    def delete_fileset(self, fileset_id: str) -> bool:
        with self.backend.session_scope(commit=True) as session:
            row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.FILESET_TEMPLATE,
                key="fileset_id",
                value=fileset_id,
            )
            if row is None:
                return False
            row.is_deleted = True
            row.bstatus = "deleted"
            session.flush()
            return True

    def clone_fileset(self, fileset_id: str, *, new_fileset_id: Optional[str] = None, name: Optional[str] = None) -> Optional[FileSet]:
        src = self.get_fileset(fileset_id)
        if src is None:
            return None
        clone_name = name or f"{src.name}-copy"
        cloned = FileSet(
            fileset_id=new_fileset_id or clone_name,
            customer_id=src.customer_id,
            name=clone_name,
            description=src.description,
            biosample_metadata=src.biosample_metadata,
            sequencing_metadata=src.sequencing_metadata,
            file_ids=list(src.file_ids),
            tags=list(src.tags),
        )
        euid = self.create_fileset(cloned)
        if euid:
            return self.get_fileset_by_euid(euid)
        return None

    def get_fileset_files(self, fileset_id: str) -> List[FileRegistration]:
        fileset = self.get_fileset(fileset_id)
        if fileset is None:
            return []
        out: List[FileRegistration] = []
        for fid in fileset.file_ids:
            reg = self.get_file(fid)
            if reg is not None:
                out.append(reg)
        return out

    def _get_workset_row(self, session, workset_euid: str):
        return self.backend.find_instance_by_euid(
            session,
            template_code=self.WORKSET_TEMPLATE,
            euid=workset_euid,
        )

    def register_file_workset_usage(
        self,
        *,
        file_id: str,
        workset_euid: str,
        customer_id: str,
        usage_type: str = "input",
        workset_state: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        with self.backend.session_scope(commit=True) as session:
            file_row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.FILE_TEMPLATE,
                key="file_id",
                value=file_id,
            )
            workset_row = self._get_workset_row(session, workset_euid)
            if file_row is None or workset_row is None:
                return False
            self.backend.create_lineage(
                session,
                parent=workset_row,
                child=file_row,
                relationship_type=f"uses_{usage_type}",
                name=f"{workset_euid}:{usage_type}:{file_id}",
            )
            payload = from_json_addl(file_row)
            usage = list(payload.get("workset_usage", []))
            usage.append(
                {
                    "file_id": file_id,
                    "workset_euid": workset_euid,
                    "customer_id": customer_id,
                    "usage_type": usage_type,
                    "added_at": utc_now_iso(),
                    "workset_state": workset_state,
                    "notes": notes,
                }
            )
            payload["workset_usage"] = usage
            payload["updated_at"] = utc_now_iso()
            self.backend.update_instance_json(session, file_row, payload)
            return True

    def record_file_workset_usage(
        self,
        *,
        file_id: str,
        workset_euid: str,
        customer_id: str,
        usage_type: str = "input",
        workset_state: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        return self.register_file_workset_usage(
            file_id=file_id,
            workset_euid=workset_euid,
            customer_id=customer_id,
            usage_type=usage_type,
            workset_state=workset_state,
            notes=notes,
        )

    def get_file_workset_history(self, file_id: str) -> List[FileWorksetUsage]:
        reg = self.get_file(file_id)
        if reg is None:
            return []
        with self.backend.session_scope(commit=False) as session:
            row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.FILE_TEMPLATE,
                key="file_id",
                value=file_id,
            )
            if row is None:
                return []
            payload = from_json_addl(row)
            hist = []
            for rec in payload.get("workset_usage", []):
                hist.append(FileWorksetUsage(**rec))
            return hist

    def get_workset_files(self, workset_euid: str) -> List[FileWorksetUsage]:
        usages: List[FileWorksetUsage] = []
        with self.backend.session_scope(commit=False) as session:
            workset = self._get_workset_row(session, workset_euid)
            if workset is None:
                return []
            file_template = self.backend.templates.get_template(session, self.FILE_TEMPLATE)
            if file_template is None:
                return []
            files = self.backend.list_children(session, parent=workset)
            for row in files:
                if row.template_uuid != file_template.uuid:
                    continue
                payload = from_json_addl(row)
                for rec in payload.get("workset_usage", []):
                    if rec.get("workset_euid") == workset_euid:
                        usages.append(FileWorksetUsage(**rec))
        return usages

    def update_workset_usage_state(self, workset_euid: str, new_state: str) -> int:
        updated = 0
        with self.backend.session_scope(commit=True) as session:
            files = self.get_workset_files(workset_euid)
            for usage in files:
                row = self.backend.find_instance_by_external_id(
                    session,
                    template_code=self.FILE_TEMPLATE,
                    key="file_id",
                    value=usage.file_id,
                )
                if row is None:
                    continue
                payload = from_json_addl(row)
                changed = False
                hist = []
                for rec in payload.get("workset_usage", []):
                    item = dict(rec)
                    if item.get("workset_euid") == workset_euid:
                        item["workset_state"] = new_state
                        changed = True
                    hist.append(item)
                if changed:
                    payload["workset_usage"] = hist
                    payload["updated_at"] = utc_now_iso()
                    self.backend.update_instance_json(session, row, payload)
                    updated += 1
        return updated

    def get_files_for_workset_recreation(self, workset_euid: str) -> List[FileRegistration]:
        out: List[FileRegistration] = []
        for usage in self.get_workset_files(workset_euid):
            if usage.usage_type != "input":
                continue
            file_reg = self.get_file(usage.file_id)
            if file_reg is not None:
                out.append(file_reg)
        dedup: Dict[str, FileRegistration] = {f.file_id: f for f in out}
        return list(dedup.values())
