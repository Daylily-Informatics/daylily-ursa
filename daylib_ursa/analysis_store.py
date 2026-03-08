"""TapDB-backed analysis persistence for Ursa beta flows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from daylib_ursa.tapdb_graph import TapDBBackend, from_json_addl, utc_now_iso
from daylily_tapdb import generic_instance


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


@dataclass(frozen=True)
class RunResolution:
    run_euid: str
    index_string: str
    atlas_tenant_id: str
    atlas_order_euid: str
    atlas_test_order_euid: str
    source_euid: str


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
    run_euid: str
    index_string: str
    atlas_tenant_id: str
    atlas_order_euid: str
    atlas_test_order_euid: str
    source_euid: str
    analysis_type: str
    state: str
    review_state: str
    result_status: str
    run_folder: str
    artifact_bucket: str
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

    def _find_analysis(
        self,
        session,
        analysis_euid: str,
        *,
        for_update: bool = False,
    ) -> generic_instance | None:
        return self.backend.find_instance_by_euid_or_external_id(
            session,
            template_code=ANALYSIS_TEMPLATE,
            key="analysis_euid",
            value=analysis_euid,
            for_update=for_update,
        )

    def _find_analysis_by_ingest_key(
        self,
        session,
        idempotency_key: str,
    ) -> generic_instance | None:
        return self.backend.find_instance_by_external_id(
            session,
            template_code=ANALYSIS_TEMPLATE,
            key="ingest_idempotency_key",
            value=idempotency_key,
        )

    def _artifact_from_instance(self, instance: generic_instance) -> AnalysisArtifact:
        payload = from_json_addl(instance)
        return AnalysisArtifact(
            artifact_euid=str(instance.euid),
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

    def _record_from_instance(
        self, instance: generic_instance, artifacts: list[AnalysisArtifact]
    ) -> AnalysisRecord:
        payload = from_json_addl(instance)
        return AnalysisRecord(
            analysis_euid=str(instance.euid),
            run_euid=str(payload.get("run_euid") or ""),
            index_string=str(payload.get("index_string") or ""),
            atlas_tenant_id=str(payload.get("atlas_tenant_id") or ""),
            atlas_order_euid=str(payload.get("atlas_order_euid") or ""),
            atlas_test_order_euid=str(payload.get("atlas_test_order_euid") or ""),
            source_euid=str(payload.get("source_euid") or ""),
            analysis_type=str(payload.get("analysis_type") or ""),
            state=str(payload.get("state") or instance.bstatus),
            review_state=str(payload.get("review_state") or ReviewState.PENDING.value),
            result_status=str(payload.get("result_status") or "PENDING"),
            run_folder=str(payload.get("run_folder") or ""),
            artifact_bucket=str(payload.get("artifact_bucket") or ""),
            result_payload=dict(payload.get("result_payload") or {}),
            metadata=dict(payload.get("metadata") or {}),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or payload.get("created_at") or utc_now_iso()),
            atlas_return=dict(payload.get("atlas_return") or {}),
            artifacts=artifacts,
        )

    def get_analysis(self, analysis_euid: str) -> AnalysisRecord | None:
        with self.backend.session_scope(commit=False) as session:
            analysis = self._find_analysis(session, analysis_euid)
            if analysis is None:
                return None
            artifacts = [
                self._artifact_from_instance(child)
                for child in self.backend.list_children(
                    session,
                    parent=analysis,
                    relationship_type="analysis_artifact",
                )
            ]
            return self._record_from_instance(analysis, artifacts)

    def ingest_analysis(
        self,
        *,
        resolution: RunResolution,
        analysis_type: str,
        artifact_bucket: str,
        idempotency_key: str,
        input_files: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AnalysisRecord:
        with self.backend.session_scope(commit=True) as session:
            existing = self._find_analysis_by_ingest_key(session, idempotency_key)
            if existing is not None:
                artifacts = [
                    self._artifact_from_instance(child)
                    for child in self.backend.list_children(
                        session,
                        parent=existing,
                        relationship_type="analysis_artifact",
                    )
                ]
                return self._record_from_instance(existing, artifacts)

            now = utc_now_iso()
            payload: dict[str, Any] = {
                "run_euid": resolution.run_euid,
                "index_string": resolution.index_string,
                "atlas_tenant_id": resolution.atlas_tenant_id,
                "atlas_order_euid": resolution.atlas_order_euid,
                "atlas_test_order_euid": resolution.atlas_test_order_euid,
                "source_euid": resolution.source_euid,
                "analysis_type": analysis_type,
                "state": AnalysisState.INGESTED.value,
                "review_state": ReviewState.PENDING.value,
                "result_status": "PENDING",
                "artifact_bucket": artifact_bucket,
                "run_folder": f"s3://{artifact_bucket}/{resolution.run_euid}/",
                "input_files": list(input_files or []),
                "metadata": dict(metadata or {}),
                "ingest_idempotency_key": idempotency_key,
                "created_at": now,
                "updated_at": now,
                "result_payload": {},
                "atlas_return": {},
                "history": [
                    {
                        "timestamp": now,
                        "state": AnalysisState.INGESTED.value,
                        "reason": "INGESTED",
                    }
                ],
            }
            analysis = self.backend.create_instance(
                session,
                ANALYSIS_TEMPLATE,
                f"analysis:{resolution.run_euid}:{resolution.index_string}",
                json_addl=payload,
                bstatus=AnalysisState.INGESTED.value,
            )
            self.backend.update_instance_json(
                session,
                analysis,
                {"analysis_euid": str(analysis.euid)},
            )
            return self._record_from_instance(analysis, [])

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
            payload = dict(analysis.json_addl or {})
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

            artifacts = [
                self._artifact_from_instance(child)
                for child in self.backend.list_children(
                    session,
                    parent=analysis,
                    relationship_type="analysis_artifact",
                )
            ]
            return self._record_from_instance(analysis, artifacts)

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
            for child in self.backend.list_children(
                session,
                parent=analysis,
                relationship_type="analysis_artifact",
            ):
                payload = dict(child.json_addl or {})
                if str(payload.get("storage_uri") or "") == storage_uri:
                    return self._artifact_from_instance(child)

            artifact = self.backend.create_instance(
                session,
                ARTIFACT_TEMPLATE,
                filename,
                json_addl={
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
            payload = dict(analysis.json_addl or {})
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
            payload = dict(analysis.json_addl or {})
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
            artifacts = [
                self._artifact_from_instance(child)
                for child in self.backend.list_children(
                    session,
                    parent=analysis,
                    relationship_type="analysis_artifact",
                )
            ]
            return self._record_from_instance(analysis, artifacts)

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
            payload = dict(analysis.json_addl or {})
            existing_return = dict(payload.get("atlas_return") or {})
            if str(existing_return.get("idempotency_key") or "") == idempotency_key:
                artifacts = [
                    self._artifact_from_instance(child)
                    for child in self.backend.list_children(
                        session,
                        parent=analysis,
                        relationship_type="analysis_artifact",
                    )
                ]
                return self._record_from_instance(analysis, artifacts)

            payload["atlas_return"] = {
                **dict(atlas_return),
                "idempotency_key": idempotency_key,
                "returned_at": utc_now_iso(),
            }
            payload["state"] = AnalysisState.RETURNED.value
            payload["result_status"] = str(
                atlas_return.get("result_status") or payload.get("result_status") or "RETURNED"
            )
            payload["updated_at"] = utc_now_iso()
            history = list(payload.get("history") or [])
            history.append(
                {
                    "timestamp": payload["updated_at"],
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
                f"atlas-return:{analysis_euid}:{payload['updated_at']}",
                json_addl=dict(payload["atlas_return"]),
                bstatus="returned",
            )
            self.backend.create_lineage(
                session,
                parent=analysis,
                child=event,
                relationship_type="atlas_return",
            )
            artifacts = [
                self._artifact_from_instance(child)
                for child in self.backend.list_children(
                    session,
                    parent=analysis,
                    relationship_type="analysis_artifact",
                )
            ]
            return self._record_from_instance(analysis, artifacts)
