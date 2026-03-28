"""TapDB-backed analysis persistence for Ursa beta flows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid

try:
    from daylib_ursa.tapdb_graph import TapDBBackend, from_json_addl, utc_now_iso
    from daylily_tapdb import generic_instance
except ImportError:  # pragma: no cover - import-time compatibility for reduced test envs
    TapDBBackend = Any  # type: ignore[assignment]
    generic_instance = Any  # type: ignore[assignment]

    def from_json_addl(instance) -> dict[str, Any]:
        return dict(getattr(instance, "json_addl", {}) or {})

    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class AnalysisState(str, Enum):
    INGESTED = "INGESTED"
    RUNNING = "RUNNING"
    REVIEW_PENDING = "REVIEW_PENDING"
    REVIEWED = "REVIEWED"
    RETURNED = "RETURNED"
    FAILED = "FAILED"


class ReviewState(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


ANALYSIS_TEMPLATE = "workflow/analysis/run-linked/1.0/"
ARTIFACT_TEMPLATE = "data/artifact/analysis-output/1.0/"
REVIEW_EVENT_TEMPLATE = "action/analysis/review-event/1.0/"
RETURN_EVENT_TEMPLATE = "action/analysis/atlas-return/1.0/"
RESOLVED_CONTEXT_TEMPLATE = "integration/reference/sequenced-assignment-context/1.0/"


@dataclass(frozen=True)
class RunResolution:
    run_euid: str
    flowcell_id: str
    lane: str
    library_barcode: str
    sequenced_library_assignment_euid: str
    tenant_id: uuid.UUID
    atlas_trf_euid: str
    atlas_test_euid: str
    atlas_test_fulfillment_item_euid: str


@dataclass(frozen=True)
class AnalysisArtifact:
    artifact_euid: str
    artifact_type: str
    storage_uri: str
    filename: str
    mime_type: str | None
    checksum_sha256: str | None
    size_bytes: int | None
    created_at: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AnalysisRecord:
    analysis_euid: str
    workset_euid: str | None
    run_euid: str
    flowcell_id: str
    lane: str
    library_barcode: str
    sequenced_library_assignment_euid: str
    tenant_id: uuid.UUID
    atlas_trf_euid: str
    atlas_test_euid: str
    atlas_test_fulfillment_item_euid: str
    analysis_type: str
    state: str
    review_state: str
    result_status: str
    run_folder: str
    internal_bucket: str
    input_references: list[dict[str, Any]]
    result_payload: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    atlas_return: dict[str, Any]
    artifacts: list[AnalysisArtifact]


class AnalysisStore:
    """Stores Ursa beta analysis execution state in TapDB generic instances."""

    def __init__(self) -> None:
        self.backend = TapDBBackend(app_username="ursa")

    def bootstrap(self) -> None:
        with self.backend.session_scope(commit=True) as session:
            self.backend.ensure_templates(session)

    @staticmethod
    def _parse_tenant_uuid(value: Any) -> uuid.UUID:
        return uuid.UUID(str(value or "").strip())

    def _find_analysis(
        self,
        session,
        analysis_euid: str,
        *,
        for_update: bool = False,
    ) -> generic_instance | None:
        return self.backend.find_instance_by_euid(
            session,
            template_code=ANALYSIS_TEMPLATE,
            value=analysis_euid,
            for_update=for_update,
        )

    def _find_analysis_by_ingest_key(
        self, session, idempotency_key: str
    ) -> generic_instance | None:
        return self.backend.find_instance_by_external_id(
            session,
            template_code=ANALYSIS_TEMPLATE,
            key="ingest_idempotency_key",
            value=idempotency_key,
        )

    @staticmethod
    def _payload(instance: generic_instance) -> dict[str, Any]:
        return dict(instance.json_addl or {})

    def _artifact_from_instance(self, instance: generic_instance) -> AnalysisArtifact:
        payload = from_json_addl(instance)
        return AnalysisArtifact(
            artifact_euid=str(payload.get("artifact_euid") or instance.euid),
            artifact_type=str(payload.get("artifact_type") or ""),
            storage_uri=str(payload.get("storage_uri") or ""),
            filename=str(payload.get("filename") or ""),
            mime_type=str(payload.get("mime_type") or "") or None,
            checksum_sha256=str(payload.get("checksum_sha256") or "") or None,
            size_bytes=int(payload["size_bytes"])
            if payload.get("size_bytes") is not None
            else None,
            created_at=str(payload.get("created_at") or utc_now_iso()),
            metadata=dict(payload.get("metadata") or {}),
        )

    def _context_payload(self, session, analysis: generic_instance) -> dict[str, Any]:
        contexts = self.backend.list_children(
            session,
            parent=analysis,
            relationship_type="resolved_context",
        )
        if not contexts:
            return {}
        contexts.sort(key=lambda item: item.created_dt or item.modified_dt, reverse=True)
        return dict(contexts[0].json_addl or {})

    def _atlas_return_payload(self, session, analysis: generic_instance) -> dict[str, Any]:
        events = self.backend.list_children(
            session,
            parent=analysis,
            relationship_type="atlas_return",
        )
        if not events:
            return {}
        events.sort(key=lambda item: item.created_dt or item.modified_dt, reverse=True)
        return dict(events[0].json_addl or {})

    def _artifacts(self, session, analysis: generic_instance) -> list[AnalysisArtifact]:
        return [
            self._artifact_from_instance(child)
            for child in self.backend.list_children(
                session,
                parent=analysis,
                relationship_type="analysis_artifact",
            )
        ]

    def _record_from_instance(
        self,
        session,
        instance: generic_instance,
        artifacts: list[AnalysisArtifact],
    ) -> AnalysisRecord:
        payload = from_json_addl(instance)
        context = self._context_payload(session, instance)
        atlas_return = self._atlas_return_payload(session, instance)
        workset_euid = None
        list_parents = getattr(self.backend, "list_parents", None)
        if callable(list_parents):
            workset_parents = list_parents(
                session,
                child=instance,
                relationship_type="workset_analysis",
            )
            if workset_parents:
                workset_euid = str(workset_parents[0].euid)
        return AnalysisRecord(
            analysis_euid=str(instance.euid),
            workset_euid=workset_euid,
            run_euid=str(context.get("run_euid") or ""),
            flowcell_id=str(context.get("flowcell_id") or ""),
            lane=str(context.get("lane") or ""),
            library_barcode=str(context.get("library_barcode") or ""),
            sequenced_library_assignment_euid=str(
                context.get("sequenced_library_assignment_euid") or ""
            ),
            tenant_id=self._parse_tenant_uuid(context.get("tenant_id")),
            atlas_trf_euid=str(context.get("atlas_trf_euid") or ""),
            atlas_test_euid=str(context.get("atlas_test_euid") or ""),
            atlas_test_fulfillment_item_euid=str(
                context.get("atlas_test_fulfillment_item_euid") or ""
            ),
            analysis_type=str(payload.get("analysis_type") or ""),
            state=str(payload.get("state") or instance.bstatus),
            review_state=str(payload.get("review_state") or ReviewState.PENDING.value),
            result_status=str(payload.get("result_status") or "PENDING"),
            run_folder=str(payload.get("run_folder") or ""),
            internal_bucket=str(payload.get("internal_bucket") or ""),
            input_references=list(payload.get("input_references") or []),
            result_payload=dict(payload.get("result_payload") or {}),
            metadata=dict(payload.get("metadata") or {}),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or payload.get("created_at") or utc_now_iso()),
            atlas_return=atlas_return,
            artifacts=artifacts,
        )

    def get_analysis(self, analysis_euid: str) -> AnalysisRecord | None:
        with self.backend.session_scope(commit=False) as session:
            analysis = self._find_analysis(session, analysis_euid)
            if analysis is None:
                return None
            return self._record_from_instance(session, analysis, self._artifacts(session, analysis))

    def list_analyses(
        self,
        *,
        tenant_id: uuid.UUID | None = None,
        workset_euid: str | None = None,
        limit: int = 200,
    ) -> list[AnalysisRecord]:
        with self.backend.session_scope(commit=False) as session:
            rows = self.backend.list_instances_by_template(
                session,
                template_code=ANALYSIS_TEMPLATE,
                limit=limit,
            )
            records: list[AnalysisRecord] = []
            for analysis in rows:
                record = self._record_from_instance(
                    session, analysis, self._artifacts(session, analysis)
                )
                if tenant_id and record.tenant_id != tenant_id:
                    continue
                if workset_euid and record.workset_euid != workset_euid:
                    continue
                records.append(record)
            return records

    def ingest_analysis(
        self,
        *,
        resolution: RunResolution,
        analysis_type: str,
        internal_bucket: str,
        idempotency_key: str,
        input_references: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AnalysisRecord:
        with self.backend.session_scope(commit=True) as session:
            existing = self._find_analysis_by_ingest_key(session, idempotency_key)
            if existing is not None:
                return self._record_from_instance(
                    session, existing, self._artifacts(session, existing)
                )

            now = utc_now_iso()
            analysis = self.backend.create_instance(
                session,
                ANALYSIS_TEMPLATE,
                f"analysis:{resolution.run_euid}:{resolution.lane}:{resolution.library_barcode}",
                json_addl={
                    "analysis_type": analysis_type,
                    "state": AnalysisState.INGESTED.value,
                    "review_state": ReviewState.PENDING.value,
                    "result_status": "PENDING",
                    "internal_bucket": internal_bucket,
                    "run_folder": f"s3://{internal_bucket}/{resolution.run_euid}/",
                    "input_references": list(input_references or []),
                    "metadata": dict(metadata or {}),
                    "ingest_idempotency_key": idempotency_key,
                    "created_at": now,
                    "updated_at": now,
                    "result_payload": {},
                    "history": [
                        {
                            "timestamp": now,
                            "state": AnalysisState.INGESTED.value,
                            "reason": "INGESTED",
                        }
                    ],
                },
                bstatus=AnalysisState.INGESTED.value,
                tenant_id=resolution.tenant_id,
            )
            context = self.backend.create_instance(
                session,
                RESOLVED_CONTEXT_TEMPLATE,
                f"context:{analysis.euid}",
                json_addl={
                    "run_euid": resolution.run_euid,
                    "flowcell_id": resolution.flowcell_id,
                    "lane": resolution.lane,
                    "library_barcode": resolution.library_barcode,
                    "sequenced_library_assignment_euid": resolution.sequenced_library_assignment_euid,
                    "tenant_id": str(resolution.tenant_id),
                    "atlas_trf_euid": resolution.atlas_trf_euid,
                    "atlas_test_euid": resolution.atlas_test_euid,
                    "atlas_test_fulfillment_item_euid": resolution.atlas_test_fulfillment_item_euid,
                    "created_at": now,
                },
                bstatus="active",
                tenant_id=resolution.tenant_id,
            )
            self.backend.create_lineage(
                session,
                parent=analysis,
                child=context,
                relationship_type="resolved_context",
            )
            return self._record_from_instance(session, analysis, [])

    def update_analysis_state(
        self,
        analysis_euid: str,
        *,
        state: AnalysisState,
        result_status: str | None = None,
        result_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> AnalysisRecord:
        with self.backend.session_scope(commit=True) as session:
            analysis = self._find_analysis(session, analysis_euid, for_update=True)
            if analysis is None:
                raise KeyError(f"analysis not found: {analysis_euid}")
            payload = self._payload(analysis)
            payload["state"] = state.value
            payload["updated_at"] = utc_now_iso()
            if result_status is not None:
                payload["result_status"] = result_status
            if result_payload is not None:
                payload["result_payload"] = dict(result_payload)
            if metadata:
                merged = dict(payload.get("metadata") or {})
                merged.update(metadata)
                payload["metadata"] = merged
            history = list(payload.get("history") or [])
            history.append(
                {
                    "timestamp": payload["updated_at"],
                    "state": state.value,
                    "reason": reason or state.value,
                }
            )
            payload["history"] = history
            analysis.bstatus = state.value
            analysis.json_addl = payload
            session.flush()
            return self._record_from_instance(session, analysis, self._artifacts(session, analysis))

    def add_artifact(
        self,
        analysis_euid: str,
        *,
        artifact_type: str,
        storage_uri: str,
        filename: str,
        mime_type: str | None = None,
        checksum_sha256: str | None = None,
        size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AnalysisArtifact:
        with self.backend.session_scope(commit=True) as session:
            analysis = self._find_analysis(session, analysis_euid, for_update=True)
            if analysis is None:
                raise KeyError(f"analysis not found: {analysis_euid}")
            dewey_artifact_euid = str((metadata or {}).get("dewey_artifact_euid") or "").strip()
            if not dewey_artifact_euid:
                raise ValueError("dewey_artifact_euid is required for persisted analysis artifacts")
            for child in self.backend.list_children(
                session,
                parent=analysis,
                relationship_type="analysis_artifact",
            ):
                payload = dict(child.json_addl or {})
                if str(payload.get("artifact_euid") or "") == dewey_artifact_euid:
                    return self._artifact_from_instance(child)

            artifact = self.backend.create_instance(
                session,
                ARTIFACT_TEMPLATE,
                filename,
                json_addl={
                    "artifact_euid": dewey_artifact_euid,
                    "artifact_type": artifact_type,
                    "storage_uri": storage_uri,
                    "filename": filename,
                    "mime_type": mime_type,
                    "checksum_sha256": checksum_sha256,
                    "size_bytes": size_bytes,
                    "metadata": dict(metadata or {}),
                    "created_at": utc_now_iso(),
                },
                bstatus="active",
            )
            self.backend.create_lineage(
                session,
                parent=analysis,
                child=artifact,
                relationship_type="analysis_artifact",
            )
            payload = self._payload(analysis)
            payload["state"] = AnalysisState.REVIEW_PENDING.value
            payload["updated_at"] = utc_now_iso()
            history = list(payload.get("history") or [])
            history.append(
                {
                    "timestamp": payload["updated_at"],
                    "state": AnalysisState.REVIEW_PENDING.value,
                    "reason": "ARTIFACT_ADDED",
                }
            )
            payload["history"] = history
            analysis.bstatus = AnalysisState.REVIEW_PENDING.value
            analysis.json_addl = payload
            session.flush()
            return self._artifact_from_instance(artifact)

    def set_review_state(
        self,
        analysis_euid: str,
        *,
        review_state: ReviewState,
        reviewer: str | None = None,
        notes: str | None = None,
    ) -> AnalysisRecord:
        with self.backend.session_scope(commit=True) as session:
            analysis = self._find_analysis(session, analysis_euid, for_update=True)
            if analysis is None:
                raise KeyError(f"analysis not found: {analysis_euid}")
            payload = self._payload(analysis)
            payload["review_state"] = review_state.value
            payload["updated_at"] = utc_now_iso()
            if review_state == ReviewState.APPROVED:
                payload["state"] = AnalysisState.REVIEWED.value
                analysis.bstatus = AnalysisState.REVIEWED.value
            elif review_state == ReviewState.REJECTED:
                payload["state"] = AnalysisState.FAILED.value
                payload["result_status"] = "REJECTED"
                analysis.bstatus = AnalysisState.FAILED.value
            history = list(payload.get("history") or [])
            history.append(
                {
                    "timestamp": payload["updated_at"],
                    "state": str(payload.get("state") or analysis.bstatus),
                    "reason": f"REVIEW_{review_state.value}",
                }
            )
            payload["history"] = history
            analysis.json_addl = payload
            session.flush()

            event = self.backend.create_instance(
                session,
                REVIEW_EVENT_TEMPLATE,
                f"review:{analysis_euid}:{payload['updated_at']}",
                json_addl={
                    "review_state": review_state.value,
                    "reviewer": reviewer,
                    "notes": notes,
                    "timestamp": payload["updated_at"],
                },
                bstatus=review_state.value,
            )
            self.backend.create_lineage(
                session,
                parent=analysis,
                child=event,
                relationship_type="review_event",
            )
            return self._record_from_instance(session, analysis, self._artifacts(session, analysis))

    def mark_returned(
        self,
        analysis_euid: str,
        *,
        atlas_return: dict[str, Any],
        idempotency_key: str,
    ) -> AnalysisRecord:
        with self.backend.session_scope(commit=True) as session:
            analysis = self._find_analysis(session, analysis_euid, for_update=True)
            if analysis is None:
                raise KeyError(f"analysis not found: {analysis_euid}")
            payload = self._payload(analysis)
            if (
                str(payload.get("review_state") or ReviewState.PENDING.value)
                != ReviewState.APPROVED.value
            ):
                raise ValueError("Analysis cannot be returned before manual approval")

            for event in self.backend.list_children(
                session,
                parent=analysis,
                relationship_type="atlas_return",
            ):
                event_payload = dict(event.json_addl or {})
                if str(event_payload.get("idempotency_key") or "") == idempotency_key:
                    return self._record_from_instance(
                        session, analysis, self._artifacts(session, analysis)
                    )

            returned_at = utc_now_iso()
            payload["state"] = AnalysisState.RETURNED.value
            payload["result_status"] = str(
                atlas_return.get("result_status") or payload.get("result_status") or "RETURNED"
            )
            payload["updated_at"] = returned_at
            history = list(payload.get("history") or [])
            history.append(
                {
                    "timestamp": returned_at,
                    "state": AnalysisState.RETURNED.value,
                    "reason": "ATLAS_RETURNED",
                }
            )
            payload["history"] = history
            analysis.bstatus = AnalysisState.RETURNED.value
            analysis.json_addl = payload
            session.flush()

            event = self.backend.create_instance(
                session,
                RETURN_EVENT_TEMPLATE,
                f"atlas-return:{analysis_euid}:{returned_at}",
                json_addl={
                    **dict(atlas_return),
                    "idempotency_key": idempotency_key,
                    "returned_at": returned_at,
                },
                bstatus="returned",
            )
            self.backend.create_lineage(
                session,
                parent=analysis,
                child=event,
                relationship_type="atlas_return",
            )
            return self._record_from_instance(session, analysis, self._artifacts(session, analysis))
