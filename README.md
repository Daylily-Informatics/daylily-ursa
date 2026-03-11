# daylily-ursa

Daylily Ursa is the analysis execution, review, artifact-linking, and Atlas result-return service.

## Authority Boundary

Ursa is analysis-only.

It owns:

- analysis ingest records linked to sequencing context
- TapDB-backed analysis state and review state
- Bloom resolver calls for canonical run context
- Dewey resolve/register flows for analysis inputs and outputs
- Atlas result return after approval

It does not own:

- customer portal routes
- onboarding or bucket-ownership flows
- file/file-set authority
- release visibility policy

## Runtime Shape

Primary package: `daylib_ursa`

Primary entrypoints:

- app factory: `daylib_ursa.workset_api:create_app`
- CLI command: `daylily-workset-api`
- package exports: analysis store, Bloom client, Dewey client, app factory

## API Surface

Current service routes:

- `GET /healthz`
- `POST /api/analyses/ingest`
- `GET /api/analyses/{analysis_euid}`
- `POST /api/analyses/{analysis_euid}/status`
- `POST /api/analyses/{analysis_euid}/artifacts`
- `POST /api/analyses/{analysis_euid}/review`
- `POST /api/analyses/{analysis_euid}/return`

Auth rules:

- analysis write routes require `X-API-Key`
- ingest and return also require `Idempotency-Key`

## Integration Contracts

Ursa expects three external service seams:

- Bloom for run resolution
- Dewey for artifact resolution and registration
- Atlas for result return and target resolution

Supported ingest input references:

- `{"reference_type":"s3_uri","value":"s3://..."}`
- `{"reference_type":"artifact_euid","value":"AT-..."}`
- `{"reference_type":"artifact_set_euid","value":"AS-..."}`

Artifact add requires exactly one of:

- `artifact_euid`
- `storage_uri` plus `artifact_type`

Result return requires:

- review state `APPROVED`
- Dewey-linked artifacts for all returned outputs

## Required Environment

```bash
URSA_INTERNAL_API_KEY=...
URSA_INTERNAL_OUTPUT_BUCKET=...

BLOOM_BASE_URL=https://...
BLOOM_API_TOKEN=...
BLOOM_VERIFY_SSL=true

ATLAS_BASE_URL=https://...
ATLAS_INTERNAL_API_KEY=...
ATLAS_VERIFY_SSL=true

DEWEY_ENABLED=true
DEWEY_BASE_URL=https://...
DEWEY_API_TOKEN=...
DEWEY_VERIFY_SSL=true
```

Cross-system integrations are authenticated and should run over HTTPS.

## Local Development

```bash
pip install -e .[dev]
daylily-workset-api --port 8914 --reload
```

Validation:

```bash
pytest -q
```

## Current Docs

- [Docs index](docs/README.md)
- [Ursa-Atlas return contract](docs/ursa_atlas_return_contract.md)

Legacy workset-monitor notes remain in `docs/`, but they are no longer the primary repo contract.

<!-- release-sweep: 2026-03-10 -->
 
 
 
 
