# daylily-ursa

**Daylily Ursa beta analysis API**.

Ursa is the analysis-only service in the beta stack. It no longer owns customer, biospecimen, manifest, portal, or workset truth.

## Scope

Ursa now handles:

- run-linked analysis ingest
- Bloom resolver lookups for `run_euid + index_string`
- TapDB-backed analysis execution and review state
- analysis artifact registration
- Atlas result and artifact return

Ursa no longer serves:

- workset lifecycle APIs
- customer portal routes
- biospecimen or manifest ownership APIs
- file-registry-as-primary ownership
- monitor or workflow-runtime beta entrypoints

## Runtime Contract

1. Atlas and Bloom create the order, material, queue, and run context.
2. Ursa ingests `run_euid` and `index_string`.
3. Ursa resolves Atlas order and TRF.test identity through Bloom.
4. Ursa records analysis state and artifacts under the resolved identity.
5. Ursa returns result and artifact references to Atlas with an idempotency key.

## Quick Start

```bash
source ./ursa_activate

export TAPDB_STRICT_NAMESPACE=1
export TAPDB_CLIENT_ID=local
export TAPDB_DATABASE_NAME=ursa
export TAPDB_ENV=dev

export URSA_INTERNAL_API_KEY=ursa-dev-internal-key
export BLOOM_BASE_URL=http://localhost:8001
export ATLAS_BASE_URL=http://localhost:8000
export ATLAS_INTERNAL_API_KEY=replace-me

ursa server start
```

Direct entrypoint:

```bash
daylily-workset-api --host 0.0.0.0 --port 8914
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
BLOOM_BASE_URL=http://localhost:8001
BLOOM_API_TOKEN=
ATLAS_BASE_URL=http://localhost:8000
ATLAS_INTERNAL_API_KEY=
URSA_HOST=0.0.0.0
URSA_PORT=8914
```

## API Surface

- `POST /api/analyses/ingest`
- `GET /api/analyses/{analysis_euid}`
- `POST /api/analyses/{analysis_euid}/status`
- `POST /api/analyses/{analysis_euid}/artifacts`
- `POST /api/analyses/{analysis_euid}/review`
- `POST /api/analyses/{analysis_euid}/return`

All write routes require:

- `X-API-Key`
- `Idempotency-Key` on ingest and result return

## Repo Notes

- execution plan: [docs/ursa_refactor_execplan.md](/Users/jmajor/projects/lims3/daylily-ursa/docs/ursa_refactor_execplan.md)
- Atlas return contract: [docs/ursa_atlas_return_contract.md](/Users/jmajor/projects/lims3/daylily-ursa/docs/ursa_atlas_return_contract.md)

## Validation

```bash
pytest tests/test_tapdb_backend.py tests/test_file_metadata.py tests/test_analysis_ingest.py tests/test_result_return.py tests/test_bloom_resolver_client.py tests/test_console_scripts.py
ruff check daylib tests/test_tapdb_backend.py tests/test_file_metadata.py tests/test_analysis_ingest.py tests/test_result_return.py tests/test_bloom_resolver_client.py tests/test_console_scripts.py
```
