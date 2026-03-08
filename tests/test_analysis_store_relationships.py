from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from daylib.analysis_store import AnalysisStore, RunResolution


class _FakeBackend:
    def __init__(self) -> None:
        self.created = []
        self.lineages = []

    @contextmanager
    def session_scope(self, *, commit: bool):
        _ = commit
        yield object()

    def ensure_templates(self, session) -> None:
        _ = session

    def find_instance_by_external_id(self, session, *, template_code, key, value):
        _ = (session, template_code, key, value)
        return None

    def create_instance(self, session, template_code, name, *, json_addl, bstatus):
        _ = session
        instance = SimpleNamespace(
            euid=f"{template_code}:{len(self.created) + 1}",
            name=name,
            json_addl=dict(json_addl),
            bstatus=bstatus,
            created_dt=None,
            modified_dt=None,
        )
        self.created.append((template_code, name, instance))
        return instance

    def create_lineage(self, session, *, parent, child, relationship_type):
        _ = session
        self.lineages.append((parent, child, relationship_type))

    def list_children(self, session, *, parent, relationship_type):
        _ = session
        return [child for source, child, rel in self.lineages if source is parent and rel == relationship_type]


def test_ingest_analysis_keeps_relationship_truth_on_context_reference():
    store = AnalysisStore.__new__(AnalysisStore)
    store.backend = _FakeBackend()

    record = store.ingest_analysis(
        resolution=RunResolution(
            run_euid="RUN-1",
            flowcell_id="FLOW-1",
            lane="1",
            library_barcode="LIB-1",
            sequenced_library_assignment_euid="SQA-1",
            atlas_tenant_id="TEN-1",
            atlas_trf_euid="TRF-1",
            atlas_test_euid="TST-1",
            atlas_test_process_item_euid="TPC-1",
        ),
        analysis_type="germline",
        artifact_bucket="analysis-bucket",
        idempotency_key="idem-1",
        input_files=["s3://analysis-bucket/RUN-1/read1.fastq.gz"],
        metadata={"pipeline": "beta"},
    )

    analysis_template, _analysis_name, analysis = store.backend.created[0]
    context_template, _context_name, context = store.backend.created[1]

    assert analysis_template == "workflow/analysis/run-linked/1.0/"
    assert context_template == "integration/reference/sequenced-assignment-context/1.0/"

    analysis_payload = dict(analysis.json_addl)
    assert "run_euid" not in analysis_payload
    assert "sequenced_library_assignment_euid" not in analysis_payload
    assert "atlas_trf_euid" not in analysis_payload
    assert "atlas_test_euid" not in analysis_payload
    assert "atlas_test_process_item_euid" not in analysis_payload

    context_payload = dict(context.json_addl)
    assert context_payload["run_euid"] == "RUN-1"
    assert context_payload["sequenced_library_assignment_euid"] == "SQA-1"
    assert context_payload["atlas_trf_euid"] == "TRF-1"
    assert context_payload["atlas_test_euid"] == "TST-1"
    assert context_payload["atlas_test_process_item_euid"] == "TPC-1"

    assert store.backend.lineages[0][2] == "resolved_context"
    assert record.run_euid == "RUN-1"
    assert record.atlas_test_process_item_euid == "TPC-1"
