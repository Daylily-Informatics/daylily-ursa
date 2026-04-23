from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from daylib_ursa.file_metadata import (
    AnalysisInput,
    generate_stage_samples_tsv,
    create_analysis_inputs_from_files,
)


ANALYSIS_SAMPLES_COLUMNS = [
    "RUN_ID",
    "SAMPLE_ID",
    "EXPERIMENTID",
    "SAMPLE_TYPE",
    "LIB_PREP",
    "SEQ_VENDOR",
    "SEQ_PLATFORM",
    "LANE",
    "SEQBC_ID",
    "PATH_TO_CONCORDANCE_DATA_DIR",
    "R1_FQ",
    "R2_FQ",
    "STAGE_DIRECTIVE",
    "STAGE_TARGET",
    "SUBSAMPLE_PCT",
    "IS_POS_CTRL",
    "IS_NEG_CTRL",
    "N_X",
    "N_Y",
    "EXTERNAL_SAMPLE_ID",
]


@dataclass(frozen=True)
class StableManifest:
    content: str
    sha256: str
    row_count: int
    filename: str = "analysis_samples.tsv"

    def metadata(self) -> dict[str, Any]:
        return {
            "format": "analysis_samples.tsv",
            "filename": self.filename,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "content": self.content,
            "columns": list(ANALYSIS_SAMPLES_COLUMNS),
        }


def _clean_cell(value: Any) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").strip()


def _canonical_rows_from_editor_inputs(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    canonical: list[dict[str, str]] = []
    for row in rows:
        canonical_row = {
            column: _clean_cell(row.get(column)) for column in ANALYSIS_SAMPLES_COLUMNS
        }
        if not canonical_row["RUN_ID"]:
            canonical_row["RUN_ID"] = "R0"
        if not canonical_row["EXPERIMENTID"]:
            canonical_row["EXPERIMENTID"] = canonical_row["SAMPLE_ID"]
        if not canonical_row["STAGE_TARGET"]:
            canonical_row["STAGE_TARGET"] = "/fsx/staged_sample_data/"
        if canonical_row["R1_FQ"] or canonical_row["R2_FQ"]:
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


def build_stable_manifest(
    *,
    metadata: dict[str, Any],
    input_references: list[dict[str, Any]],
) -> StableManifest:
    editor_rows = metadata.get("editor_analysis_inputs")
    if isinstance(editor_rows, list) and editor_rows:
        rows = _canonical_rows_from_editor_inputs(
            row for row in editor_rows if isinstance(row, dict)
        )
        if not rows:
            raise ValueError("editor_analysis_inputs does not contain any FASTQ rows")
        content = _content_from_rows(rows)
        return StableManifest(
            content=content,
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            row_count=len(rows),
        )

    files = _s3_values_from_references(input_references)
    if not files:
        raise ValueError(
            "stable manifest creation requires editor_analysis_inputs or S3 FASTQ references"
        )
    analysis_inputs: list[AnalysisInput] = create_analysis_inputs_from_files(files)
    if not analysis_inputs:
        raise ValueError("S3 references did not contain any R1 FASTQ inputs")
    content = generate_stage_samples_tsv(analysis_inputs, include_header=True) + "\n"
    return StableManifest(
        content=content,
        sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        row_count=len(analysis_inputs),
    )
