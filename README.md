# daylily-ursa

**Daylily Ursa beta analysis + customer portal API**.

Ursa serves analysis workflows and TapDB-backed portal/manifest/workset surfaces in one FastAPI app.
Persistent objects and canonical IDs are TapDB-managed EUIDs.

## Scope

Ursa handles:

- run-linked analysis ingest
- Bloom resolver lookups for `run_euid + flowcell_id + lane + library_barcode`
- TapDB-backed analysis execution and review state
- analysis artifact registration
- Atlas result and artifact return
- TapDB-backed portal customer/manifest/workset surfaces
- customer portal pages (`/portal/...`) and namespaced customer APIs (`/api/customers/{customer_id}/...`)

## Runtime Contract

1. Atlas and Bloom create the order, material, queue, and run context.
2. Ursa ingests `run_euid`, `flowcell_id`, `lane`, and `library_barcode`.
3. Ursa resolves Atlas TRF/Test/fulfillment-item identity through Bloom.
4. Ursa records analysis state, review state, and artifacts under the resolved identity.
5. Ursa requires explicit `APPROVED` review state before returning result and artifact references to Atlas.
6. Portal/customer/manifest/workset objects are persisted through TapDB graph templates and lineage.

## Timezone Policy

- Ursa stores runtime/persisted timestamps in UTC (`GMT+00:00`).
- Portal display timezone is user-configurable and shared across apps through TapDB `system_user` preferences:
  - key: `display_timezone`
  - format: IANA timezone name
  - default: `UTC`
- Portal JS formatters use the configured timezone explicitly (`window.UrsaConfig.displayTimezone`).

## Quick Start

```bash
source ./ursa_activate

export TAPDB_STRICT_NAMESPACE=1
export TAPDB_CLIENT_ID=local
export TAPDB_DATABASE_NAME=ursa
export TAPDB_ENV=dev

export URSA_INTERNAL_API_KEY=ursa-dev-internal-key
export BLOOM_BASE_URL=https://localhost:8912
export BLOOM_API_TOKEN=replace-with-bloom-bearer-token
export BLOOM_VERIFY_SSL=true
export ATLAS_BASE_URL=https://localhost:8915
export ATLAS_INTERNAL_API_KEY=replace-me
export ATLAS_VERIFY_SSL=true
export DEWEY_ENABLED=true
export DEWEY_BASE_URL=https://localhost:8913
export DEWEY_API_TOKEN=replace-with-dewey-bearer-token
export DEWEY_VERIFY_SSL=true

ursa server start
```

## Important Environment Variables

```bash
AWS_PROFILE=your-profile
URSA_ALLOWED_REGIONS=us-west-2

TAPDB_STRICT_NAMESPACE=1
TAPDB_CLIENT_ID=local
TAPDB_DATABASE_NAME=ursa
TAPDB_ENV=dev

URSA_INTERNAL_API_KEY=ursa-dev-internal-key
BLOOM_BASE_URL=https://localhost:8912
BLOOM_API_TOKEN=replace-with-bloom-bearer-token
BLOOM_VERIFY_SSL=true
ATLAS_BASE_URL=https://localhost:8915
ATLAS_INTERNAL_API_KEY=replace-with-atlas-internal-api-key
ATLAS_VERIFY_SSL=true
DEWEY_ENABLED=true
DEWEY_BASE_URL=https://localhost:8913
DEWEY_API_TOKEN=replace-with-dewey-bearer-token
DEWEY_VERIFY_SSL=true
URSA_HOST=0.0.0.0
URSA_PORT=8914
```

## API Surface

Analysis:

- `POST /api/analyses/ingest`
- `GET /api/analyses/{analysis_euid}`
- `POST /api/analyses/{analysis_euid}/status`
- `POST /api/analyses/{analysis_euid}/artifacts`
- `POST /api/analyses/{analysis_euid}/review`
- `POST /api/analyses/{analysis_euid}/return`

Portal/customer/manifest/workset highlights:

- `GET /portal/*`
- `GET|POST /api/customers/{customer_id}/manifests*`
- `GET|POST /api/customers/{customer_id}/worksets*`

Analysis write routes require:

- `X-API-Key`
- `Idempotency-Key` on ingest and result return

Cross-system integration calls are `https://` only and always authenticated:

- Ursa -> Bloom resolver: `Authorization: Bearer <BLOOM_API_TOKEN>`
- Ursa -> Atlas result return: `X-API-Key: <ATLAS_INTERNAL_API_KEY>`
- Ursa -> Dewey artifact resolve/register: `Authorization: Bearer <DEWEY_API_TOKEN>`

`POST /api/analyses/{analysis_euid}/artifacts` accepts either:

- raw `storage_uri` (+ `artifact_type`)
- `artifact_euid` (resolved via Dewey when Dewey integration is configured)

The ingest payload includes:

- `run_euid`
- `flowcell_id`
- `lane`
- `library_barcode`

The stored and returned Atlas context includes:

- `sequenced_library_assignment_euid`
- `atlas_tenant_id`
- `atlas_trf_euid`
- `atlas_test_euid`
- `atlas_test_fulfillment_item_euid`

## Repo Notes

- customer portal runtime behavior: [docs/CUSTOMER_PORTAL.md](/Users/jmajor/projects/lims3/daylily-ursa/docs/CUSTOMER_PORTAL.md)
- execution plan: [docs/ursa_refactor_execplan.md](/Users/jmajor/projects/lims3/daylily-ursa/docs/ursa_refactor_execplan.md)
- Atlas return contract: [docs/ursa_atlas_return_contract.md](/Users/jmajor/projects/lims3/daylily-ursa/docs/ursa_atlas_return_contract.md)

## Validation

```bash
pytest tests/test_route_coverage.py tests/test_portal_onboarding.py tests/test_portal_features.py tests/test_portal_restore_scope.py tests/test_restore_guardrails.py tests/test_analysis_ingest.py tests/test_result_return.py tests/test_bloom_resolver_client.py tests/test_analysis_store_relationships.py
ruff check daylib_ursa tests
```
