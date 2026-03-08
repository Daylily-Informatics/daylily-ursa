# Ursa Beta Refactor Execution Summary

## Scope

Ursa is reduced to the beta analysis path only:

- ingest run-linked analysis requests
- resolve `run_euid + flowcell_id + lane + library_barcode` through Bloom
- persist analysis execution and review state in TapDB
- register analysis artifacts
- return analysis results and artifact references to Atlas

Retired from the main runtime path:

- workset lifecycle APIs
- customer portal and customer ownership APIs
- biospecimen and manifest ownership APIs
- file-registry-as-primary truth
- workset scheduler, monitor, and cluster orchestration as beta entrypoints

## Runtime Contract

Incoming request:

- `run_euid`
- `flowcell_id`
- `lane`
- `library_barcode`
- analysis metadata
- idempotency key

Resolution step:

- Ursa calls Bloom `GET /api/v1/external/atlas/beta/runs/{run_euid}/resolve?flowcell_id=...&lane=...&library_barcode=...`
- Bloom returns `sequenced_library_assignment_euid`, `atlas_tenant_id`, `atlas_trf_euid`, `atlas_test_euid`, and `atlas_test_process_item_euid`

Persistence:

- Ursa stores one analysis record keyed by opaque EUID
- artifacts, review events, and return events are linked explicitly to the analysis record
- no public API leaks internal UUIDs

Return step:

- Ursa sends Atlas the resolved TRF/Test/process-item EUIDs, Ursa analysis EUID, review state, result status, and artifact references
- return is blocked unless review state is `APPROVED`
- return is idempotent by request key

## Deliverables

- new beta-only FastAPI app in `daylib/workset_api.py`
- TapDB-backed analysis store
- Bloom resolver client
- Atlas result-return client
- updated CLI boot path
- updated README and Ursa beta contract doc
- targeted tests for ingest, Bloom resolution, result return, and TapDB backend templates

## Breaking Changes

- `daylily-workset-api` now serves an analysis integration API, not the legacy workset/customer portal
- legacy workset/customer/biospecimen routes are removed from the main app
- Ursa no longer treats customer IDs, biospecimen IDs, or workset IDs as authoritative beta identifiers
- Bloom resolution and Atlas return use opaque EUIDs only
- the old `run_euid + index_string` resolver contract is retired

## Validation

Focused validation:

```bash
cd /Users/jmajor/projects/lims3/daylily-ursa
pytest tests/test_tapdb_backend.py tests/test_file_metadata.py tests/test_analysis_ingest.py tests/test_result_return.py tests/test_bloom_resolver_client.py tests/test_console_scripts.py
ruff check daylib tests/test_tapdb_backend.py tests/test_file_metadata.py tests/test_analysis_ingest.py tests/test_result_return.py tests/test_bloom_resolver_client.py tests/test_console_scripts.py
```
