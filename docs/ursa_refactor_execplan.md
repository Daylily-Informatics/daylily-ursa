# Ursa Beta Refactor Execution Plan

## Scope

Ursa is reduced to the beta analysis path only:

- ingest run-linked analysis requests
- resolve `run_euid + index_string` through Bloom
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
- `index_string`
- analysis metadata
- idempotency key

Resolution step:

- Ursa calls Bloom `GET /api/v1/external/atlas/beta/runs/{run_euid}/resolve?index_string=...`
- Bloom returns `atlas_tenant_id`, `atlas_order_euid`, `atlas_test_order_euid`

Persistence:

- Ursa stores one analysis record keyed by opaque EUID
- artifacts and review events are linked explicitly to the analysis record
- no public API leaks internal UUIDs

Return step:

- Ursa sends Atlas the resolved order/test-order EUIDs, Ursa analysis EUID, review state, result status, and artifact references
- return is idempotent by request key

## Deliverables

- new beta-only FastAPI app in `daylib_ursa/workset_api.py`
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

## Validation

Planned phase validation:

```bash
cd /Users/jmajor/projects/lims3/daylily-ursa
pytest tests/test_tapdb_backend.py tests/test_file_metadata.py tests/test_analysis_ingest.py tests/test_result_return.py tests/test_bloom_resolver_client.py tests/test_console_scripts.py
ruff check daylib_ursa tests/test_tapdb_backend.py tests/test_file_metadata.py tests/test_analysis_ingest.py tests/test_result_return.py tests/test_bloom_resolver_client.py tests/test_console_scripts.py
```
