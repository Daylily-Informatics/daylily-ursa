"""Federated global search for portal entities.

This module provides a single search entrypoint used by both HTML and JSON
portal search routes. It aggregates results across multiple system domains
without introducing a dedicated search index table.
"""

from __future__ import annotations

import logging
import shlex
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote_plus

from daylily_ursa.routes.dependencies import verify_workset_access
from daylily_ursa.workset_state_db import WorksetState

LOGGER = logging.getLogger("daylily.search.global")

ALL_TYPES: Set[str] = {
    "workset",
    "file",
    "subject",
    "biosample",
    "library",
    "manifest",
    "cluster",
    "user",
    "monitor_log",
}
ADMIN_ONLY_TYPES: Set[str] = {"user", "monitor_log"}

DEFAULT_LIMIT_PER_TYPE = 20
MAX_LIMIT_PER_TYPE = 100
MAX_CUSTOMERS_SCOPE_ALL = 200
MAX_CANDIDATES_PER_TYPE = 500
MAX_MONITOR_LINES = 2000

SUPPORTED_OPERATORS: Set[str] = {
    "type",
    "state",
    "region",
    "customer",
    "tag",
    "file_format",
    "sample_type",
    "platform",
}


@dataclass
class ParsedQuery:
    """Represents parsed search query text and inline operators."""

    terms: List[str] = field(default_factory=list)
    operators: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def free_text(self) -> str:
        return " ".join(self.terms).strip()


@dataclass
class SearchScope:
    """Session-derived scope and identity context."""

    customer_id: str
    user_email: str
    is_admin: bool
    requested_scope: str


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _to_lower(value: Any) -> str:
    return _to_str(value).strip().lower()


def _listify(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [_to_str(v).strip() for v in value if _to_str(v).strip()]
    raw = _to_str(value).strip()
    return [raw] if raw else []


def _normalize_type(value: str) -> str:
    normalized = _to_lower(value)
    if normalized.endswith("s") and normalized[:-1] in ALL_TYPES:
        return normalized[:-1]
    aliases = {
        "worksets": "workset",
        "files": "file",
        "subjects": "subject",
        "biosamples": "biosample",
        "libraries": "library",
        "manifests": "manifest",
        "clusters": "cluster",
        "users": "user",
        "monitor": "monitor_log",
        "logs": "monitor_log",
        "monitor_logs": "monitor_log",
    }
    return aliases.get(normalized, normalized)


def _parse_iso_to_ts(value: Any) -> float:
    text = _to_str(value).strip()
    if not text:
        return 0.0
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _parse_query(query: str) -> ParsedQuery:
    parsed = ParsedQuery(operators={op: [] for op in SUPPORTED_OPERATORS})
    if not query or not query.strip():
        return parsed

    try:
        tokens = shlex.split(query)
    except ValueError:
        # Fallback to split when user input has unmatched quotes.
        tokens = query.split()

    for token in tokens:
        if ":" in token:
            key, raw_value = token.split(":", 1)
            key_l = key.strip().lower()
            value = raw_value.strip()
            if key_l in SUPPORTED_OPERATORS and value:
                parsed.operators[key_l].append(value)
                continue
        parsed.terms.append(token)

    # Remove empty operator buckets for cleaner payloads
    parsed.operators = {k: v for k, v in parsed.operators.items() if v}
    return parsed


def _resolve_scope(filters: Dict[str, Any], session_context: Dict[str, Any], warnings: List[str]) -> SearchScope:
    is_admin = bool(session_context.get("is_admin", False))
    requested = _to_lower(filters.get("scope") or ("all" if is_admin else "mine"))
    if requested not in {"all", "mine"}:
        warnings.append(f"Invalid scope '{requested}', falling back to default scope")
        requested = "all" if is_admin else "mine"

    effective = requested
    if not is_admin and requested == "all":
        effective = "mine"
        warnings.append("Non-admin searches are always restricted to scope=mine")

    return SearchScope(
        customer_id=_to_str(session_context.get("customer_id")),
        user_email=_to_str(session_context.get("user_email")),
        is_admin=is_admin,
        requested_scope=effective,
    )


def _extract_filters(filters: Dict[str, Any], parsed_query: ParsedQuery) -> Dict[str, Any]:
    def _first(name: str) -> str:
        explicit = _to_str(filters.get(name)).strip()
        if explicit:
            return explicit
        op_values = parsed_query.operators.get(name, [])
        return op_values[0].strip() if op_values else ""

    combined_types = set(_normalize_type(t) for t in _listify(filters.get("types")))
    combined_types.update(_normalize_type(t) for t in parsed_query.operators.get("type", []))
    combined_types = {t for t in combined_types if t}

    tags = [t.lower() for t in _listify(filters.get("tag"))]
    tags.extend(_to_lower(t) for t in parsed_query.operators.get("tag", []))
    tags = [t for t in tags if t]

    return {
        "types": sorted(combined_types),
        "state": _to_lower(_first("state")),
        "region": _to_lower(_first("region")),
        "customer": _to_lower(_first("customer")),
        "tags": tags,
        "file_format": _to_lower(_first("file_format")),
        "sample_type": _to_lower(_first("sample_type")),
        "platform": _to_lower(_first("platform")),
    }


def _resolve_requested_types(extracted_filters: Dict[str, Any], scope: SearchScope, warnings: List[str]) -> Set[str]:
    requested = set(extracted_filters.get("types") or [])
    if not requested:
        requested = set(ALL_TYPES)

    unknown = {t for t in requested if t not in ALL_TYPES}
    if unknown:
        warnings.append(f"Ignoring unknown type filters: {', '.join(sorted(unknown))}")
        requested = {t for t in requested if t in ALL_TYPES}

    if not scope.is_admin:
        requested -= ADMIN_ONLY_TYPES

    return requested


def _prepare_customer_targets(
    scope: SearchScope,
    extracted_filters: Dict[str, Any],
    customer_manager: Optional[Any],
    warnings: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    customer_filter = _to_lower(extracted_filters.get("customer"))
    customer_email_map: Dict[str, str] = {}

    if scope.requested_scope == "mine":
        if not scope.customer_id:
            warnings.append("Session has no customer_id; customer-scoped results may be empty")
            return [], customer_email_map
        if customer_filter and not _matches_customer_filter(scope.customer_id, scope.user_email, customer_filter):
            return [], customer_email_map
        customer_email_map[scope.customer_id] = scope.user_email
        return [scope.customer_id], customer_email_map

    # scope=all for admin
    if not customer_manager:
        warnings.append("Customer manager unavailable; falling back to current session customer")
        if scope.customer_id:
            customer_email_map[scope.customer_id] = scope.user_email
            return [scope.customer_id], customer_email_map
        return [], customer_email_map

    try:
        customers = customer_manager.list_customers()
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to enumerate customers for scope=all search: %s", exc)
        warnings.append("Failed to enumerate customers for scope=all; falling back to session customer")
        if scope.customer_id:
            customer_email_map[scope.customer_id] = scope.user_email
            return [scope.customer_id], customer_email_map
        return [], customer_email_map

    customer_ids: List[str] = []
    for customer in customers:
        cid = _to_str(getattr(customer, "customer_id", ""))
        email = _to_str(getattr(customer, "email", ""))
        if not cid:
            continue

        if customer_filter:
            cname = _to_lower(getattr(customer, "customer_name", ""))
            if (
                customer_filter not in cid.lower()
                and customer_filter not in email.lower()
                and customer_filter not in cname
            ):
                continue

        customer_ids.append(cid)
        customer_email_map[cid] = email

        if len(customer_ids) >= MAX_CUSTOMERS_SCOPE_ALL:
            warnings.append(
                f"Customer scope limited to first {MAX_CUSTOMERS_SCOPE_ALL} matches for performance"
            )
            break

    if not customer_ids and scope.customer_id:
        customer_ids = [scope.customer_id]
        customer_email_map[scope.customer_id] = scope.user_email

    return customer_ids, customer_email_map


def _matches_customer_filter(customer_id: str, customer_email: str, customer_filter: str) -> bool:
    if not customer_filter:
        return True
    token = customer_filter.lower()
    return token in customer_id.lower() or token in customer_email.lower()


def _all_tags_present(available_tags: Sequence[str], required_tags: Sequence[str]) -> bool:
    if not required_tags:
        return True
    normalized = {_to_lower(tag) for tag in available_tags if _to_lower(tag)}
    return all(required in normalized for required in required_tags)


def _match_terms(terms: Sequence[str], field_map: Dict[str, str]) -> Tuple[bool, float, List[str]]:
    if not terms:
        return True, 1.0, []

    score = 0.0
    matched_fields: Set[str] = set()

    for raw_term in terms:
        term = _to_lower(raw_term)
        if not term:
            continue

        term_found = False
        term_score = 0.0

        for field_name, value in field_map.items():
            haystack = _to_lower(value)
            if not haystack:
                continue

            candidate_score = 0.0
            if haystack == term:
                candidate_score = 12.0
            elif any(part == term for part in haystack.replace("/", " ").replace("_", " ").replace("-", " ").split()):
                candidate_score = 11.0
            elif haystack.startswith(term):
                candidate_score = 9.0
            elif f" {term}" in haystack:
                candidate_score = 7.0
            elif term in haystack:
                candidate_score = 5.0

            if candidate_score > 0:
                term_found = True
                term_score = max(term_score, candidate_score)
                matched_fields.add(field_name)

        if not term_found:
            return False, 0.0, []

        score += term_score

    if len(terms) > 1:
        score += 1.0

    return True, score, sorted(matched_fields)


def _sort_and_limit_hits(hits: List[Dict[str, Any]], limit_per_type: int) -> Tuple[List[Dict[str, Any]], bool]:
    sorted_hits = sorted(
        hits,
        key=lambda item: (
            float(item.get("score", 0.0)),
            _parse_iso_to_ts(item.get("created_at")),
        ),
        reverse=True,
    )
    truncated = len(sorted_hits) > limit_per_type
    return sorted_hits[:limit_per_type], truncated


def _build_hit(
    *,
    hit_id: str,
    hit_type: str,
    title: str,
    subtitle: str,
    url: str,
    customer_id: str,
    badges: Sequence[str],
    score: float,
    matched_fields: Sequence[str],
    created_at: str,
) -> Dict[str, Any]:
    return {
        "id": hit_id,
        "type": hit_type,
        "title": title,
        "subtitle": subtitle,
        "url": url,
        "customer_id": customer_id,
        "badges": [b for b in badges if b],
        "score": round(score, 4),
        "matched_fields": list(matched_fields),
        "created_at": created_at,
    }


def _search_worksets(
    *,
    parsed_query: ParsedQuery,
    extracted_filters: Dict[str, Any],
    scope: SearchScope,
    customer_ids: Sequence[str],
    customer_email_map: Dict[str, str],
    deps: Dict[str, Any],
    candidate_cap: int,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    warnings: List[str] = []
    hits: List[Dict[str, Any]] = []
    candidates = 0

    state_db = deps.get("state_db")
    if state_db is None:
        warnings.append("Workset database unavailable")
        return [], warnings, {"candidates": 0, "matched": 0}

    state_filter = _to_lower(extracted_filters.get("state"))
    region_filter = _to_lower(extracted_filters.get("region"))
    customer_filter = _to_lower(extracted_filters.get("customer"))

    state_obj: Optional[WorksetState] = None
    if state_filter:
        for ws_state in WorksetState:
            if ws_state.value == state_filter:
                state_obj = ws_state
                break
        if state_obj is None:
            # Invalid state means no workset matches
            return [], warnings, {"candidates": 0, "matched": 0}

    for customer_id in customer_ids:
        if not _matches_customer_filter(customer_id, customer_email_map.get(customer_id, ""), customer_filter):
            continue

        try:
            batch = state_db.list_worksets_by_customer(customer_id, state=state_obj, limit=candidate_cap)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Failed to query worksets for customer '{customer_id}'")
            LOGGER.warning("Workset search failed for customer %s: %s", customer_id, exc)
            continue

        if len(batch) >= candidate_cap:
            warnings.append(f"Workset candidate cap reached for customer '{customer_id}'")

        for workset in batch:
            candidates += 1
            if not verify_workset_access(
                workset,
                customer_id=scope.customer_id,
                user_email=scope.user_email,
                is_admin=scope.is_admin,
            ):
                continue

            ws_id = _to_str(workset.get("workset_id"))
            ws_name = _to_str(workset.get("name"))
            ws_state = _to_str(workset.get("state"))
            metadata = workset.get("metadata") if isinstance(workset.get("metadata"), dict) else {}

            ws_region = _to_lower(
                workset.get("execution_cluster_region")
                or workset.get("cluster_region")
                or metadata.get("cluster_region")
                or ""
            )
            if region_filter and ws_region != region_filter:
                continue

            if state_filter and _to_lower(ws_state) != state_filter:
                continue

            tags = [
                _to_str(metadata.get("workset_type")),
                _to_str(metadata.get("pipeline_type")),
                _to_str(metadata.get("reference_genome")),
            ]
            if not _all_tags_present(tags, extracted_filters.get("tags", [])):
                continue

            fields = {
                "workset_id": ws_id,
                "name": ws_name,
                "state": ws_state,
                "workset_type": _to_str(metadata.get("workset_type")),
                "pipeline_type": _to_str(metadata.get("pipeline_type")),
                "notification_email": _to_str(metadata.get("notification_email")),
                "submitted_by": _to_str(metadata.get("submitted_by")),
                "customer_id": _to_str(workset.get("customer_id")),
            }
            matched, score, matched_fields = _match_terms(parsed_query.terms, fields)
            if not matched:
                continue

            subtitle_parts = [ws_state or "unknown"]
            if fields["pipeline_type"]:
                subtitle_parts.append(fields["pipeline_type"])
            subtitle_parts.append(ws_id)
            if ws_region:
                subtitle_parts.append(ws_region)

            hit = _build_hit(
                hit_id=ws_id,
                hit_type="workset",
                title=ws_name or ws_id,
                subtitle=" • ".join(part for part in subtitle_parts if part),
                url=f"/portal/worksets/{quote_plus(ws_id)}",
                customer_id=_to_str(workset.get("customer_id")) or customer_id,
                badges=[ws_state, fields["workset_type"], ws_region],
                score=score,
                matched_fields=matched_fields,
                created_at=_to_str(workset.get("created_at")),
            )
            hits.append(hit)

    return hits, warnings, {"candidates": candidates, "matched": len(hits)}


def _search_files(
    *,
    parsed_query: ParsedQuery,
    extracted_filters: Dict[str, Any],
    customer_ids: Sequence[str],
    customer_email_map: Dict[str, str],
    deps: Dict[str, Any],
    candidate_cap: int,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    warnings: List[str] = []
    hits: List[Dict[str, Any]] = []
    candidates = 0

    file_registry = deps.get("file_registry")
    if file_registry is None:
        warnings.append("File registry unavailable")
        return [], warnings, {"candidates": 0, "matched": 0}

    customer_filter = _to_lower(extracted_filters.get("customer"))
    required_tags = extracted_filters.get("tags", [])
    file_format_filter = _to_lower(extracted_filters.get("file_format"))
    sample_type_filter = _to_lower(extracted_filters.get("sample_type"))
    platform_filter = _to_lower(extracted_filters.get("platform"))

    for customer_id in customer_ids:
        if not _matches_customer_filter(customer_id, customer_email_map.get(customer_id, ""), customer_filter):
            continue

        try:
            files = file_registry.list_customer_files(customer_id, limit=candidate_cap)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Failed to query files for customer '{customer_id}'")
            LOGGER.warning("File search failed for customer %s: %s", customer_id, exc)
            continue

        if len(files) >= candidate_cap:
            warnings.append(f"File candidate cap reached for customer '{customer_id}'")

        for file_reg in files:
            candidates += 1

            file_id = _to_str(getattr(file_reg, "file_id", ""))
            file_customer_id = _to_str(getattr(file_reg, "customer_id", "")) or customer_id
            file_meta = getattr(file_reg, "file_metadata", None)
            seq_meta = getattr(file_reg, "sequencing_metadata", None)
            bio_meta = getattr(file_reg, "biosample_metadata", None)
            tags = getattr(file_reg, "tags", []) or []

            s3_uri = _to_str(getattr(file_meta, "s3_uri", ""))
            filename = s3_uri.rsplit("/", 1)[-1] if s3_uri else ""
            file_format = _to_lower(getattr(file_meta, "file_format", ""))
            sample_type = _to_lower(getattr(bio_meta, "sample_type", ""))
            platform = _to_lower(getattr(seq_meta, "platform", ""))

            if file_format_filter and file_format != file_format_filter:
                continue
            if sample_type_filter and sample_type != sample_type_filter:
                continue
            if platform_filter and platform_filter not in platform:
                continue
            if not _all_tags_present(tags, required_tags):
                continue

            fields = {
                "file_id": file_id,
                "filename": filename,
                "s3_uri": s3_uri,
                "subject_id": _to_str(getattr(bio_meta, "subject_id", "")),
                "biosample_id": _to_str(getattr(bio_meta, "biosample_id", "")),
                "tags": " ".join(_to_str(t) for t in tags),
                "file_format": file_format,
                "sample_type": sample_type,
                "platform": platform,
                "customer_id": file_customer_id,
            }
            matched, score, matched_fields = _match_terms(parsed_query.terms, fields)
            if not matched:
                continue

            subtitle_parts = [filename or file_id]
            if fields["subject_id"]:
                subtitle_parts.append(f"subject:{fields['subject_id']}")
            if file_format:
                subtitle_parts.append(file_format)

            hit = _build_hit(
                hit_id=file_id,
                hit_type="file",
                title=filename or file_id,
                subtitle=" • ".join(subtitle_parts),
                url=f"/portal/files/{quote_plus(file_id)}",
                customer_id=file_customer_id,
                badges=[file_format, sample_type, platform],
                score=score,
                matched_fields=matched_fields,
                created_at=_to_str(getattr(file_reg, "registered_at", "")),
            )
            hits.append(hit)

    return hits, warnings, {"candidates": candidates, "matched": len(hits)}


def _search_subjects(
    *,
    parsed_query: ParsedQuery,
    extracted_filters: Dict[str, Any],
    customer_ids: Sequence[str],
    customer_email_map: Dict[str, str],
    deps: Dict[str, Any],
    candidate_cap: int,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    warnings: List[str] = []
    hits: List[Dict[str, Any]] = []
    candidates = 0

    registry = deps.get("biospecimen_registry")
    if registry is None:
        warnings.append("Biospecimen registry unavailable")
        return [], warnings, {"candidates": 0, "matched": 0}

    customer_filter = _to_lower(extracted_filters.get("customer"))
    required_tags = extracted_filters.get("tags", [])

    for customer_id in customer_ids:
        if not _matches_customer_filter(customer_id, customer_email_map.get(customer_id, ""), customer_filter):
            continue

        try:
            subjects = registry.list_subjects(customer_id, limit=candidate_cap)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Failed to query subjects for customer '{customer_id}'")
            LOGGER.warning("Subject search failed for customer %s: %s", customer_id, exc)
            continue

        if len(subjects) >= candidate_cap:
            warnings.append(f"Subject candidate cap reached for customer '{customer_id}'")

        for subject in subjects:
            candidates += 1
            subject_tags = list(getattr(subject, "tags", []) or [])
            if not _all_tags_present(subject_tags, required_tags):
                continue

            subject_id = _to_str(getattr(subject, "subject_id", ""))
            identifier = _to_str(getattr(subject, "identifier", ""))
            display_name = _to_str(getattr(subject, "display_name", ""))
            cohort = _to_str(getattr(subject, "cohort", ""))

            fields = {
                "subject_id": subject_id,
                "identifier": identifier,
                "display_name": display_name,
                "cohort": cohort,
                "sex": _to_str(getattr(subject, "sex", "")),
                "tags": " ".join(_to_str(tag) for tag in subject_tags),
                "customer_id": customer_id,
            }
            matched, score, matched_fields = _match_terms(parsed_query.terms, fields)
            if not matched:
                continue

            subtitle_parts = [identifier or subject_id]
            if cohort:
                subtitle_parts.append(f"cohort:{cohort}")

            hit = _build_hit(
                hit_id=subject_id,
                hit_type="subject",
                title=display_name or identifier or subject_id,
                subtitle=" • ".join(part for part in subtitle_parts if part),
                url=f"/portal/biospecimen/subjects?subject_id={quote_plus(subject_id)}",
                customer_id=customer_id,
                badges=[_to_str(getattr(subject, "sex", "")), cohort],
                score=score,
                matched_fields=matched_fields,
                created_at=_to_str(getattr(subject, "created_at", "")),
            )
            hits.append(hit)

    return hits, warnings, {"candidates": candidates, "matched": len(hits)}


def _search_biosamples(
    *,
    parsed_query: ParsedQuery,
    extracted_filters: Dict[str, Any],
    customer_ids: Sequence[str],
    customer_email_map: Dict[str, str],
    deps: Dict[str, Any],
    candidate_cap: int,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    warnings: List[str] = []
    hits: List[Dict[str, Any]] = []
    candidates = 0

    registry = deps.get("biospecimen_registry")
    if registry is None:
        warnings.append("Biospecimen registry unavailable")
        return [], warnings, {"candidates": 0, "matched": 0}

    customer_filter = _to_lower(extracted_filters.get("customer"))
    required_tags = extracted_filters.get("tags", [])
    sample_type_filter = _to_lower(extracted_filters.get("sample_type"))

    for customer_id in customer_ids:
        if not _matches_customer_filter(customer_id, customer_email_map.get(customer_id, ""), customer_filter):
            continue

        try:
            biosamples = registry.list_biosamples(customer_id, limit=candidate_cap)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Failed to query biosamples for customer '{customer_id}'")
            LOGGER.warning("Biosample search failed for customer %s: %s", customer_id, exc)
            continue

        if len(biosamples) >= candidate_cap:
            warnings.append(f"Biosample candidate cap reached for customer '{customer_id}'")

        for biosample in biosamples:
            candidates += 1
            biosample_tags = list(getattr(biosample, "tags", []) or [])
            if not _all_tags_present(biosample_tags, required_tags):
                continue

            sample_type = _to_lower(getattr(biosample, "sample_type", ""))
            if sample_type_filter and sample_type != sample_type_filter:
                continue

            biosample_id = _to_str(getattr(biosample, "biosample_id", ""))
            subject_id = _to_str(getattr(biosample, "subject_id", ""))

            fields = {
                "biosample_id": biosample_id,
                "subject_id": subject_id,
                "sample_type": sample_type,
                "tissue_type": _to_str(getattr(biosample, "tissue_type", "")),
                "tags": " ".join(_to_str(tag) for tag in biosample_tags),
                "customer_id": customer_id,
            }
            matched, score, matched_fields = _match_terms(parsed_query.terms, fields)
            if not matched:
                continue

            subtitle_parts = [subject_id] if subject_id else []
            if sample_type:
                subtitle_parts.append(sample_type)

            if subject_id:
                target_url = f"/portal/biospecimen/subjects?subject_id={quote_plus(subject_id)}"
            else:
                target_url = f"/portal/biospecimen/subjects?q={quote_plus(biosample_id)}"

            hit = _build_hit(
                hit_id=biosample_id,
                hit_type="biosample",
                title=biosample_id,
                subtitle=" • ".join(part for part in subtitle_parts if part),
                url=target_url,
                customer_id=customer_id,
                badges=[sample_type, _to_str(getattr(biosample, "tissue_type", ""))],
                score=score,
                matched_fields=matched_fields,
                created_at=_to_str(getattr(biosample, "created_at", "")),
            )
            hits.append(hit)

    return hits, warnings, {"candidates": candidates, "matched": len(hits)}


def _search_libraries(
    *,
    parsed_query: ParsedQuery,
    extracted_filters: Dict[str, Any],
    customer_ids: Sequence[str],
    customer_email_map: Dict[str, str],
    deps: Dict[str, Any],
    candidate_cap: int,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    warnings: List[str] = []
    hits: List[Dict[str, Any]] = []
    candidates = 0

    registry = deps.get("biospecimen_registry")
    if registry is None:
        warnings.append("Biospecimen registry unavailable")
        return [], warnings, {"candidates": 0, "matched": 0}

    customer_filter = _to_lower(extracted_filters.get("customer"))
    required_tags = extracted_filters.get("tags", [])

    for customer_id in customer_ids:
        if not _matches_customer_filter(customer_id, customer_email_map.get(customer_id, ""), customer_filter):
            continue

        try:
            libraries = registry.list_libraries(customer_id, limit=candidate_cap)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Failed to query libraries for customer '{customer_id}'")
            LOGGER.warning("Library search failed for customer %s: %s", customer_id, exc)
            continue

        if len(libraries) >= candidate_cap:
            warnings.append(f"Library candidate cap reached for customer '{customer_id}'")

        for library in libraries:
            candidates += 1
            library_tags = list(getattr(library, "tags", []) or [])
            if not _all_tags_present(library_tags, required_tags):
                continue

            library_id = _to_str(getattr(library, "library_id", ""))
            biosample_id = _to_str(getattr(library, "biosample_id", ""))
            library_prep = _to_str(getattr(library, "library_prep", ""))

            fields = {
                "library_id": library_id,
                "biosample_id": biosample_id,
                "library_prep": library_prep,
                "library_kit": _to_str(getattr(library, "library_kit", "")),
                "protocol_id": _to_str(getattr(library, "protocol_id", "")),
                "tags": " ".join(_to_str(tag) for tag in library_tags),
                "customer_id": customer_id,
            }
            matched, score, matched_fields = _match_terms(parsed_query.terms, fields)
            if not matched:
                continue

            hit = _build_hit(
                hit_id=library_id,
                hit_type="library",
                title=library_id,
                subtitle=" • ".join(part for part in [biosample_id, library_prep] if part),
                url=f"/portal/biospecimen/subjects?q={quote_plus(library_id or biosample_id)}",
                customer_id=customer_id,
                badges=[library_prep],
                score=score,
                matched_fields=matched_fields,
                created_at=_to_str(getattr(library, "created_at", "")),
            )
            hits.append(hit)

    return hits, warnings, {"candidates": candidates, "matched": len(hits)}


def _search_manifests(
    *,
    parsed_query: ParsedQuery,
    extracted_filters: Dict[str, Any],
    customer_ids: Sequence[str],
    customer_email_map: Dict[str, str],
    deps: Dict[str, Any],
    candidate_cap: int,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    warnings: List[str] = []
    hits: List[Dict[str, Any]] = []
    candidates = 0

    manifest_registry = deps.get("manifest_registry")
    if manifest_registry is None:
        warnings.append("Manifest registry unavailable")
        return [], warnings, {"candidates": 0, "matched": 0}

    customer_filter = _to_lower(extracted_filters.get("customer"))

    for customer_id in customer_ids:
        if not _matches_customer_filter(customer_id, customer_email_map.get(customer_id, ""), customer_filter):
            continue

        try:
            manifests = manifest_registry.list_customer_manifests(customer_id, limit=candidate_cap)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Failed to query manifests for customer '{customer_id}'")
            LOGGER.warning("Manifest search failed for customer %s: %s", customer_id, exc)
            continue

        if len(manifests) >= candidate_cap:
            warnings.append(f"Manifest candidate cap reached for customer '{customer_id}'")

        for manifest in manifests:
            candidates += 1
            manifest_id = _to_str(manifest.get("manifest_id"))
            name = _to_str(manifest.get("name"))
            description = _to_str(manifest.get("description"))
            sample_count = _to_str(manifest.get("sample_count"))

            fields = {
                "manifest_id": manifest_id,
                "name": name,
                "description": description,
                "sample_count": sample_count,
                "customer_id": customer_id,
            }
            matched, score, matched_fields = _match_terms(parsed_query.terms, fields)
            if not matched:
                continue

            subtitle_parts = []
            if description:
                subtitle_parts.append(description)
            if sample_count:
                subtitle_parts.append(f"samples:{sample_count}")

            hit = _build_hit(
                hit_id=manifest_id,
                hit_type="manifest",
                title=name or manifest_id,
                subtitle=" • ".join(subtitle_parts) or manifest_id,
                url=f"/portal/manifest-generator?manifest_id={quote_plus(manifest_id)}",
                customer_id=customer_id,
                badges=[f"samples:{sample_count}" if sample_count else ""],
                score=score,
                matched_fields=matched_fields,
                created_at=_to_str(manifest.get("created_at")),
            )
            hits.append(hit)

    return hits, warnings, {"candidates": candidates, "matched": len(hits)}


def _search_users(
    *,
    parsed_query: ParsedQuery,
    extracted_filters: Dict[str, Any],
    scope: SearchScope,
    customer_ids: Sequence[str],
    deps: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    warnings: List[str] = []
    hits: List[Dict[str, Any]] = []
    candidates = 0

    if not scope.is_admin:
        return [], warnings, {"candidates": 0, "matched": 0}

    customer_manager = deps.get("customer_manager")
    if customer_manager is None:
        warnings.append("Customer manager unavailable for user search")
        return [], warnings, {"candidates": 0, "matched": 0}

    customer_filter = _to_lower(extracted_filters.get("customer"))
    if not customer_ids:
        return [], warnings, {"candidates": 0, "matched": 0}
    allowed_customer_ids = set(customer_ids)

    try:
        customers = customer_manager.list_customers()
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append("Failed to list customers for user search")
        LOGGER.warning("User search failed while listing customers: %s", exc)
        return [], warnings, {"candidates": 0, "matched": 0}

    for customer in customers:
        candidates += 1
        customer_id = _to_str(getattr(customer, "customer_id", ""))
        customer_name = _to_str(getattr(customer, "customer_name", ""))
        email = _to_str(getattr(customer, "email", ""))

        if allowed_customer_ids and customer_id not in allowed_customer_ids:
            continue

        if customer_filter and customer_filter not in customer_id.lower() and customer_filter not in email.lower() and customer_filter not in customer_name.lower():
            continue

        fields = {
            "customer_id": customer_id,
            "customer_name": customer_name,
            "email": email,
            "cost_center": _to_str(getattr(customer, "cost_center", "")),
            "s3_bucket": _to_str(getattr(customer, "s3_bucket", "")),
        }
        matched, score, matched_fields = _match_terms(parsed_query.terms, fields)
        if not matched:
            continue

        title = email or customer_name or customer_id
        subtitle_parts = [customer_id]
        if customer_name:
            subtitle_parts.append(customer_name)

        hit = _build_hit(
            hit_id=customer_id,
            hit_type="user",
            title=title,
            subtitle=" • ".join(subtitle_parts),
            url=f"/portal/admin/users?q={quote_plus(email or customer_id)}",
            customer_id=customer_id,
            badges=["admin" if bool(getattr(customer, "is_admin", False)) else "user"],
            score=score,
            matched_fields=matched_fields,
            created_at="",
        )
        hits.append(hit)

    return hits, warnings, {"candidates": candidates, "matched": len(hits)}


def _search_clusters(
    *,
    parsed_query: ParsedQuery,
    extracted_filters: Dict[str, Any],
    deps: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    warnings: List[str] = []
    hits: List[Dict[str, Any]] = []
    candidates = 0

    settings = deps.get("settings")
    if settings is None:
        warnings.append("Settings unavailable for cluster search")
        return [], warnings, {"candidates": 0, "matched": 0}

    region_filter = _to_lower(extracted_filters.get("region"))

    try:
        from daylily_ursa.cluster_service import get_cluster_service
        from daylily_ursa.ursa_config import get_ursa_config

        ursa_config = get_ursa_config()
        if ursa_config.is_configured:
            allowed_regions = ursa_config.get_allowed_regions()
            aws_profile = ursa_config.aws_profile or settings.aws_profile
        else:
            allowed_regions = settings.get_allowed_regions()
            aws_profile = settings.aws_profile

        if not allowed_regions:
            return [], warnings, {"candidates": 0, "matched": 0}

        service = get_cluster_service(
            regions=allowed_regions,
            aws_profile=aws_profile,
            cache_ttl_seconds=300,
        )
        clusters = service.get_all_clusters_with_status(fetch_ssh_status=False)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append("Failed to query cluster status")
        LOGGER.warning("Cluster search failed: %s", exc)
        return [], warnings, {"candidates": 0, "matched": 0}

    for cluster_obj in clusters:
        try:
            cluster = cluster_obj.to_dict(include_sensitive=False)
        except Exception:
            # Fallback for unexpected cluster object shapes
            cluster = _to_str(cluster_obj)
        if not isinstance(cluster, dict):
            continue

        candidates += 1
        region = _to_lower(cluster.get("region"))
        if region_filter and region != region_filter:
            continue

        cluster_name = _to_str(cluster.get("cluster_name"))
        cluster_status = _to_str(cluster.get("cluster_status"))
        fleet_status = _to_str(cluster.get("compute_fleet_status"))

        fields = {
            "cluster_name": cluster_name,
            "region": region,
            "cluster_status": cluster_status,
            "fleet_status": fleet_status,
        }
        matched, score, matched_fields = _match_terms(parsed_query.terms, fields)
        if not matched:
            continue

        hit = _build_hit(
            hit_id=f"{cluster_name}:{region}",
            hit_type="cluster",
            title=cluster_name,
            subtitle=" • ".join(part for part in [region, cluster_status, fleet_status] if part),
            url="/portal/clusters",
            customer_id="",
            badges=[region, cluster_status],
            score=score,
            matched_fields=matched_fields,
            created_at="",
        )
        hits.append(hit)

    return hits, warnings, {"candidates": candidates, "matched": len(hits)}


def _search_monitor_logs(
    *,
    parsed_query: ParsedQuery,
    scope: SearchScope,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    warnings: List[str] = []
    hits: List[Dict[str, Any]] = []
    candidates = 0

    if not scope.is_admin:
        return [], warnings, {"candidates": 0, "matched": 0}

    # Log search requires terms; otherwise returning logs would be noisy.
    if not parsed_query.terms:
        return [], warnings, {"candidates": 0, "matched": 0}

    log_dir = Path.home() / ".ursa" / "logs"
    log_files = sorted(log_dir.glob("monitor_*.log"), reverse=True) if log_dir.exists() else []
    if not log_files:
        return [], warnings, {"candidates": 0, "matched": 0}

    latest_log = log_files[0]
    try:
        lines = latest_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:  # pragma: no cover - defensive
        warnings.append("Failed to read monitor log file")
        LOGGER.warning("Monitor log search failed: %s", exc)
        return [], warnings, {"candidates": 0, "matched": 0}

    if len(lines) > MAX_MONITOR_LINES:
        lines = lines[-MAX_MONITOR_LINES:]
        warnings.append(f"Monitor log search limited to last {MAX_MONITOR_LINES} lines")

    for idx, line in enumerate(lines, start=1):
        candidates += 1
        fields = {"line": line, "log_file": latest_log.name}
        matched, score, matched_fields = _match_terms(parsed_query.terms, fields)
        if not matched:
            continue

        snippet = line.strip()
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."

        hit = _build_hit(
            hit_id=f"{latest_log.name}:{idx}",
            hit_type="monitor_log",
            title=snippet,
            subtitle=f"{latest_log.name} • line {idx}",
            url="/portal/monitor",
            customer_id="",
            badges=["monitor"],
            score=score,
            matched_fields=matched_fields,
            created_at="",
        )
        hits.append(hit)

    return hits, warnings, {"candidates": candidates, "matched": len(hits)}


def search_portal(
    *,
    query: str,
    filters: Dict[str, Any],
    session_context: Dict[str, Any],
    deps: Dict[str, Any],
) -> Dict[str, Any]:
    """Search across portal entities using federated source queries.

    Args:
        query: Raw query text from user input.
        filters: Structured filters from request params.
        session_context: Session-derived auth context.
        deps: Service dependencies (state_db, registries, settings, etc.).

    Returns:
        Normalized search payload with grouped hits and metadata.
    """
    started_at = time.perf_counter()
    warnings: List[str] = []

    parsed_query = _parse_query(query)
    extracted_filters = _extract_filters(filters, parsed_query)
    scope = _resolve_scope(filters, session_context, warnings)

    try:
        limit_per_type = int(filters.get("limit_per_type") or DEFAULT_LIMIT_PER_TYPE)
    except (TypeError, ValueError):
        limit_per_type = DEFAULT_LIMIT_PER_TYPE
        warnings.append("Invalid limit_per_type provided; using default value")
    if limit_per_type < 1:
        limit_per_type = DEFAULT_LIMIT_PER_TYPE
    if limit_per_type > MAX_LIMIT_PER_TYPE:
        limit_per_type = MAX_LIMIT_PER_TYPE
        warnings.append(f"limit_per_type capped at {MAX_LIMIT_PER_TYPE}")

    requested_types = _resolve_requested_types(extracted_filters, scope, warnings)
    customer_ids, customer_email_map = _prepare_customer_targets(
        scope=scope,
        extracted_filters=extracted_filters,
        customer_manager=deps.get("customer_manager"),
        warnings=warnings,
    )

    has_structured_filter = any(
        [
            extracted_filters.get("state"),
            extracted_filters.get("region"),
            extracted_filters.get("customer"),
            extracted_filters.get("tags"),
            extracted_filters.get("file_format"),
            extracted_filters.get("sample_type"),
            extracted_filters.get("platform"),
            extracted_filters.get("types"),
        ]
    )

    if not parsed_query.terms and not has_structured_filter:
        return {
            "query": query,
            "free_text": parsed_query.free_text,
            "operators": parsed_query.operators,
            "scope": scope.requested_scope,
            "limit_per_type": limit_per_type,
            "filters": extracted_filters,
            "types": sorted(requested_types),
            "results": {t: [] for t in sorted(requested_types)},
            "counts": {t: 0 for t in sorted(requested_types)},
            "returned_counts": {t: 0 for t in sorted(requested_types)},
            "total": 0,
            "warnings": warnings,
            "timings_ms": {},
        }

    candidate_cap = min(MAX_CANDIDATES_PER_TYPE, max(100, limit_per_type * 12))

    results: Dict[str, List[Dict[str, Any]]] = {t: [] for t in sorted(requested_types)}
    counts: Dict[str, int] = {t: 0 for t in sorted(requested_types)}
    returned_counts: Dict[str, int] = {t: 0 for t in sorted(requested_types)}
    timings_ms: Dict[str, float] = {}
    adapter_stats: Dict[str, Dict[str, int]] = {}

    def _run_adapter(name: str, func: Any, **kwargs: Any) -> None:
        if name not in requested_types:
            return
        start = time.perf_counter()
        try:
            hits, adapter_warnings, stats = func(**kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Search adapter '%s' failed", name)
            hits, adapter_warnings, stats = [], [f"Adapter '{name}' failed: {exc}"], {"candidates": 0, "matched": 0}
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms[name] = round(elapsed_ms, 2)
        adapter_stats[name] = {
            "candidates": int(stats.get("candidates", 0)),
            "matched": int(stats.get("matched", 0)),
        }

        warnings.extend(adapter_warnings)

        counts[name] = len(hits)
        limited_hits, truncated = _sort_and_limit_hits(hits, limit_per_type)
        if truncated:
            warnings.append(f"Results for '{name}' truncated to {limit_per_type} entries")
        results[name] = limited_hits
        returned_counts[name] = len(limited_hits)

    _run_adapter(
        "workset",
        _search_worksets,
        parsed_query=parsed_query,
        extracted_filters=extracted_filters,
        scope=scope,
        customer_ids=customer_ids,
        customer_email_map=customer_email_map,
        deps=deps,
        candidate_cap=candidate_cap,
    )

    _run_adapter(
        "file",
        _search_files,
        parsed_query=parsed_query,
        extracted_filters=extracted_filters,
        customer_ids=customer_ids,
        customer_email_map=customer_email_map,
        deps=deps,
        candidate_cap=candidate_cap,
    )

    _run_adapter(
        "subject",
        _search_subjects,
        parsed_query=parsed_query,
        extracted_filters=extracted_filters,
        customer_ids=customer_ids,
        customer_email_map=customer_email_map,
        deps=deps,
        candidate_cap=candidate_cap,
    )

    _run_adapter(
        "biosample",
        _search_biosamples,
        parsed_query=parsed_query,
        extracted_filters=extracted_filters,
        customer_ids=customer_ids,
        customer_email_map=customer_email_map,
        deps=deps,
        candidate_cap=candidate_cap,
    )

    _run_adapter(
        "library",
        _search_libraries,
        parsed_query=parsed_query,
        extracted_filters=extracted_filters,
        customer_ids=customer_ids,
        customer_email_map=customer_email_map,
        deps=deps,
        candidate_cap=candidate_cap,
    )

    _run_adapter(
        "manifest",
        _search_manifests,
        parsed_query=parsed_query,
        extracted_filters=extracted_filters,
        customer_ids=customer_ids,
        customer_email_map=customer_email_map,
        deps=deps,
        candidate_cap=candidate_cap,
    )

    _run_adapter(
        "cluster",
        _search_clusters,
        parsed_query=parsed_query,
        extracted_filters=extracted_filters,
        deps=deps,
    )

    _run_adapter(
        "user",
        _search_users,
        parsed_query=parsed_query,
        extracted_filters=extracted_filters,
        scope=scope,
        customer_ids=customer_ids,
        deps=deps,
    )

    _run_adapter(
        "monitor_log",
        _search_monitor_logs,
        parsed_query=parsed_query,
        scope=scope,
    )

    total = sum(returned_counts.values())
    total_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    timings_ms["total"] = total_ms

    LOGGER.info(
        "portal_search: scope=%s terms=%s types=%s total=%d duration_ms=%.2f adapters=%s",
        scope.requested_scope,
        parsed_query.terms,
        sorted(requested_types),
        total,
        total_ms,
        adapter_stats,
    )

    return {
        "query": query,
        "free_text": parsed_query.free_text,
        "operators": parsed_query.operators,
        "scope": scope.requested_scope,
        "limit_per_type": limit_per_type,
        "filters": extracted_filters,
        "types": sorted(requested_types),
        "results": results,
        "counts": counts,
        "returned_counts": returned_counts,
        "total": total,
        "warnings": warnings,
        "timings_ms": timings_ms,
        "adapter_stats": adapter_stats,
    }
