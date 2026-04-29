from __future__ import annotations

import re
from typing import Any

from daylib_ursa.file_metadata import ANALYSIS_SAMPLES_COLUMNS, DEFAULT_STAGE_TARGET


MANIFEST_EDITOR_OPTION_TYPES = frozenset({"sample_type", "library_prep"})

BUILTIN_SAMPLE_TYPES: tuple[str, ...] = (
    "blood",
    "saliva",
    "gDNA",
    "buccal",
    "blood spot",
    "tissue",
    "tumor",
    "normal",
    "cfDNA",
    "ffpe",
    "organoid",
    "PBMC",
    "plasma",
    "serum",
    "cell line",
)

BUILTIN_LIBRARY_PREPS: tuple[str, ...] = (
    "noampwgs",
    "no-amp-wgs",
    "PCR-FREE",
    "wgs",
    "target-capture",
    "targeted",
    "wes",
    "rna-seq",
    "rnaseq",
    "amplicon",
    "pcr",
)

BUILTIN_SEQ_PLATFORMS: tuple[str, ...] = (
    "NOVASEQ",
    "NOVASEQX",
    "NOVASEQ6000",
    "HISEQX",
    "NEXTSEQ",
    "NEXTSEQ2000",
    "MISEQ",
    "REVIO",
    "SEQUEL",
    "SEQUELII",
    "PROMETHION",
    "GRIDION",
    "MINION",
    "UG100",
    "ULTIMA",
    "COMPLETE_GENOMICS",
    "DNBSEQ_G400",
    "DNBSEQ_T7",
    "AVITI",
    "ROCHE",
)

BUILTIN_SEQ_VENDORS: tuple[str, ...] = (
    "ILMN",
    "CG",
    "PACBIO",
    "ONT",
    "UG",
    "ROCHE",
    "ELEMENT",
    "BGI",
)

MANIFEST_SOURCE_COLUMNS: tuple[str, ...] = (
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

MANIFEST_BROWSE_COLUMNS: tuple[str, ...] = (
    "PATH_TO_CONCORDANCE_DATA_DIR",
    *MANIFEST_SOURCE_COLUMNS,
)

MANIFEST_DEFAULTS: dict[str, str] = {
    "RUN_ID": "R0",
    "SAMPLE_TYPE": "blood",
    "LIB_PREP": "noampwgs",
    "SEQ_VENDOR": "ILMN",
    "SEQ_PLATFORM": "NOVASEQX",
    "LANE": "0",
    "SEQBC_ID": "S1",
    "PATH_TO_CONCORDANCE_DATA_DIR": "",
    "STAGE_DIRECTIVE": "stage_data",
    "STAGE_TARGET": DEFAULT_STAGE_TARGET,
    "SUBSAMPLE_PCT": "na",
    "IS_POS_CTRL": "false",
    "IS_NEG_CTRL": "false",
    "N_X": "1",
    "N_Y": "1",
}

_COLUMN_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Identity And Run",
        ("RUN_ID", "SAMPLE_ID", "EXPERIMENTID", "EXTERNAL_SAMPLE_ID"),
    ),
    (
        "Sample, Library, And Platform",
        ("SAMPLE_TYPE", "LIB_PREP", "SEQ_VENDOR", "SEQ_PLATFORM", "LANE", "SEQBC_ID"),
    ),
    (
        "Short-Read FASTQs",
        (
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
        ),
    ),
    (
        "Long-Read And Aligned Inputs",
        (
            "ULTIMA_CRAM",
            "ULTIMA_CRAM_ALIGNER",
            "ULTIMA_CRAM_SNV_CALLER",
            "ULTIMA_SUBSAMPLE_PCT",
            "ONT_CRAM",
            "ONT_CRAM_ALIGNER",
            "ONT_CRAM_SNV_CALLER",
            "ONT_SUBSAMPLE_PCT",
            "PB_BAM",
            "PB_BAM_ALIGNER",
            "PB_BAM_SNV_CALLER",
            "ONT_BAM",
            "ONT_BAM_ALIGNER",
            "ONT_BAM_SNV_CALLER",
            "ROCHE_BAM",
            "ROCHE_BAM_ALIGNER",
            "ROCHE_BAM_SNV_CALLER",
            "ROCHE_DOWNSAMPLE_RATIO",
        ),
    ),
    (
        "Staging And Trimming",
        (
            "PATH_TO_CONCORDANCE_DATA_DIR",
            "STAGE_DIRECTIVE",
            "STAGE_TARGET",
            "SUBSAMPLE_PCT",
            "ILMN_TRIM_READ_LENGTH",
            "LONGREADTRIM_READ_LENGTH",
            "LONGREADTRIM_MODE",
        ),
    ),
    (
        "Models And Controls",
        (
            "SAMPLEUSE",
            "BWA_KMER",
            "DEEP_MODEL",
            "IS_POS_CTRL",
            "IS_NEG_CTRL",
            "N_X",
            "N_Y",
        ),
    ),
)


def normalize_editor_option_value(value: Any) -> tuple[str, str]:
    cleaned = re.sub(r"\s+", " ", str(value or "").strip())
    if not cleaned:
        raise ValueError("value is required")
    if "\t" in cleaned or "\n" in cleaned or "\r" in cleaned:
        raise ValueError("value cannot contain tab or newline characters")
    if len(cleaned) > 80:
        raise ValueError("value must be 80 characters or fewer")
    return cleaned, cleaned.casefold()


def validate_editor_option_type(option_type: Any) -> str:
    normalized = str(option_type or "").strip()
    if normalized not in MANIFEST_EDITOR_OPTION_TYPES:
        raise ValueError("option_type must be sample_type or library_prep")
    return normalized


def builtins_for_option_type(option_type: str) -> tuple[str, ...]:
    validated = validate_editor_option_type(option_type)
    if validated == "sample_type":
        return BUILTIN_SAMPLE_TYPES
    return BUILTIN_LIBRARY_PREPS


def is_builtin_editor_option(option_type: str, value: str) -> bool:
    _, normalized = normalize_editor_option_value(value)
    return normalized in {
        normalize_editor_option_value(item)[1] for item in builtins_for_option_type(option_type)
    }


def dedupe_option_values(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned, normalized = normalize_editor_option_value(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(cleaned)
    return deduped


def manifest_column_groups() -> list[dict[str, Any]]:
    current_columns = tuple(ANALYSIS_SAMPLES_COLUMNS)
    current_set = set(current_columns)
    used: set[str] = set()
    groups: list[dict[str, Any]] = []
    for title, fields in _COLUMN_GROUPS:
        active_fields = [field for field in fields if field in current_set]
        if not active_fields:
            continue
        used.update(active_fields)
        groups.append({"title": title, "fields": active_fields})
    remaining = [field for field in current_columns if field not in used]
    if remaining:
        groups.append({"title": "Miscellaneous", "fields": remaining})
    return groups


def manifest_editor_static_payload() -> dict[str, Any]:
    return {
        "columns": list(ANALYSIS_SAMPLES_COLUMNS),
        "source_columns": [
            field for field in MANIFEST_SOURCE_COLUMNS if field in ANALYSIS_SAMPLES_COLUMNS
        ],
        "browse_columns": [
            field for field in MANIFEST_BROWSE_COLUMNS if field in ANALYSIS_SAMPLES_COLUMNS
        ],
        "column_groups": manifest_column_groups(),
        "defaults": dict(MANIFEST_DEFAULTS),
        "sample_types": list(BUILTIN_SAMPLE_TYPES),
        "library_preps": list(BUILTIN_LIBRARY_PREPS),
        "seq_platforms": list(BUILTIN_SEQ_PLATFORMS),
        "seq_vendors": list(BUILTIN_SEQ_VENDORS),
    }
