from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from daylib_ursa.file_metadata import (
    ANALYSIS_SAMPLES_COLUMNS,
    ANALYSIS_SAMPLES_TEMPLATE_PACKAGE,
    ANALYSIS_SAMPLES_TEMPLATE_RESOURCE,
    DEFAULT_STAGE_TARGET,
    AnalysisInput,
    create_analysis_inputs_from_files,
    require_daylily_ec_template_version,
)


ANALYSIS_SAMPLES_SCHEMA = "ursa.analysis_samples_manifest/1.0"
ANALYSIS_SAMPLES_FORMAT = "analysis_samples.tsv"
ANALYSIS_SAMPLES_COLUMN_SET = frozenset(ANALYSIS_SAMPLES_COLUMNS)
ANALYSIS_SAMPLES_SOURCE_COLUMNS = (
    "R1_FQ",
    "R2_FQ",
    "ILMN_R1_FQ",
    "ILMN_R2_FQ",
    "CG_R1_FQ",
    "CG_R2_FQ",
    "PACBIO_R1_FQ",
    "PACBIO_R2_FQ",
    "ONT_R1_FQ",
    "ONT_R2_FQ",
    "UG_R1_FQ",
    "UG_R2_FQ",
    "ULTIMA_CRAM",
    "ONT_CRAM",
    "PB_BAM",
    "ONT_BAM",
    "ROCHE_BAM",
)


@dataclass(frozen=True)
class AnalysisSamplesManifest:
    columns: tuple[str, ...]
    rows: list[dict[str, str]]
    content: str
    sha256: str
    row_count: int
    sample_count: int
    input_references: list[dict[str, Any]]
    artifact_euids: list[str]
    staging: dict[str, Any]
    analysis_defaults: dict[str, Any]
    filename: str = "analysis_samples.tsv"
    schema: str = ANALYSIS_SAMPLES_SCHEMA
    format: str = ANALYSIS_SAMPLES_FORMAT

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "format": self.format,
            "filename": self.filename,
            "columns": list(self.columns),
            "rows": [dict(row) for row in self.rows],
            "content": self.content,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "sample_count": self.sample_count,
            "input_references": [dict(item) for item in self.input_references],
            "artifact_euids": list(self.artifact_euids),
            "staging": dict(self.staging),
            "analysis_defaults": dict(self.analysis_defaults),
            "template_distribution": "daylily-ephemeral-cluster",
            "template_version": require_daylily_ec_template_version(),
            "template_resource": (
                f"{ANALYSIS_SAMPLES_TEMPLATE_PACKAGE}:{ANALYSIS_SAMPLES_TEMPLATE_RESOURCE}"
            ),
        }


def _clean_cell(value: Any) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").strip()


def _clean_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _stage_target_from_metadata(metadata: dict[str, Any]) -> str:
    run_config = _clean_mapping(metadata.get("editor_run_config"))
    return _clean_cell(run_config.get("stage_target")) or DEFAULT_STAGE_TARGET


def _analysis_defaults_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return _clean_mapping(metadata.get("editor_analysis_defaults"))


def _validate_editor_row_columns(row: dict[str, Any]) -> None:
    unknown_columns = sorted(str(column) for column in set(row) - ANALYSIS_SAMPLES_COLUMN_SET)
    if unknown_columns:
        raise ValueError(
            "editor_analysis_inputs contains columns outside daylily-ec "
            "analysis_samples_template.tsv: " + ", ".join(unknown_columns)
        )


def _canonical_rows_from_editor_inputs(
    rows: Iterable[dict[str, Any]],
    *,
    stage_target: str,
) -> list[dict[str, str]]:
    canonical: list[dict[str, str]] = []
    for row in rows:
        _validate_editor_row_columns(row)
        canonical_row = {
            column: _clean_cell(row.get(column)) for column in ANALYSIS_SAMPLES_COLUMNS
        }
        if not canonical_row["RUN_ID"]:
            canonical_row["RUN_ID"] = "R0"
        if not canonical_row["EXPERIMENTID"]:
            canonical_row["EXPERIMENTID"] = canonical_row["SAMPLE_ID"]
        if not canonical_row["STAGE_DIRECTIVE"]:
            canonical_row["STAGE_DIRECTIVE"] = "stage_data"
        if not canonical_row["STAGE_TARGET"]:
            canonical_row["STAGE_TARGET"] = stage_target
        if any(canonical_row.get(column) for column in ANALYSIS_SAMPLES_SOURCE_COLUMNS):
            canonical.append(canonical_row)
    canonical.sort(
        key=lambda item: (
            item["RUN_ID"],
            item["SAMPLE_ID"],
            item["EXPERIMENTID"],
            item["LANE"],
            item["SEQBC_ID"],
            item["R1_FQ"],
            item["R2_FQ"],
        )
    )
    return canonical


def _row_from_analysis_input(analysis_input: AnalysisInput) -> dict[str, str]:
    partial = analysis_input.to_tsv_row()
    return {column: _clean_cell(partial.get(column)) for column in ANALYSIS_SAMPLES_COLUMNS}


def _content_from_rows(rows: list[dict[str, str]]) -> str:
    lines = ["\t".join(ANALYSIS_SAMPLES_COLUMNS)]
    for row in rows:
        lines.append("\t".join(row.get(column, "") for column in ANALYSIS_SAMPLES_COLUMNS))
    return "\n".join(lines) + "\n"


def _s3_values_from_references(input_references: Iterable[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for ref in input_references:
        if str(ref.get("reference_type") or "") != "s3_uri":
            continue
        value = _clean_cell(ref.get("value"))
        if value.startswith("s3://"):
            values.append(value)
    return sorted(set(values))


def _build_manifest(
    *,
    rows: list[dict[str, str]],
    metadata: dict[str, Any],
    input_references: list[dict[str, Any]],
    artifact_euids: list[str],
    stage_target: str,
) -> AnalysisSamplesManifest:
    content = _content_from_rows(rows)
    sample_ids = {row.get("SAMPLE_ID", "") for row in rows if row.get("SAMPLE_ID", "")}
    return AnalysisSamplesManifest(
        columns=ANALYSIS_SAMPLES_COLUMNS,
        rows=rows,
        content=content,
        sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        row_count=len(rows),
        sample_count=len(sample_ids),
        input_references=[dict(item) for item in input_references],
        artifact_euids=list(artifact_euids),
        staging={"stage_target": stage_target},
        analysis_defaults=_analysis_defaults_from_metadata(metadata),
    )


def build_analysis_samples_manifest(
    *,
    metadata: dict[str, Any],
    input_references: list[dict[str, Any]],
    artifact_euids: list[str],
) -> AnalysisSamplesManifest:
    stage_target = _stage_target_from_metadata(metadata)
    editor_rows = metadata.get("editor_analysis_inputs")
    if isinstance(editor_rows, list) and editor_rows:
        rows = _canonical_rows_from_editor_inputs(
            (row for row in editor_rows if isinstance(row, dict)),
            stage_target=stage_target,
        )
        if not rows:
            raise ValueError("editor_analysis_inputs does not contain any staged input rows")
        return _build_manifest(
            rows=rows,
            metadata=metadata,
            input_references=input_references,
            artifact_euids=artifact_euids,
            stage_target=stage_target,
        )

    files = _s3_values_from_references(input_references)
    if not files:
        raise ValueError(
            "analysis_samples manifest creation requires editor_analysis_inputs or S3 FASTQ references"
        )
    analysis_inputs = create_analysis_inputs_from_files(files, stage_target=stage_target)
    if not analysis_inputs:
        raise ValueError("S3 references did not contain any R1 FASTQ inputs")
    rows = [_row_from_analysis_input(item) for item in analysis_inputs]
    return _build_manifest(
        rows=rows,
        metadata=metadata,
        input_references=input_references,
        artifact_euids=artifact_euids,
        stage_target=stage_target,
    )
