# daylily-ursa

Daylily Ursa is the analysis execution/review/result-return API.

## Scope

Ursa is analysis-only. It is not a customer portal and does not own customer bucket/file identity.

Ursa responsibilities:

- run-linked analysis ingest
- Bloom resolver lookup for run context
- TapDB-backed analysis execution/review state
- Dewey artifact resolve/register for analysis inputs/outputs
- Atlas result return after manual approval

Ursa does **not** provide:

- `/portal/*` routes
- customer onboarding APIs
- customer bucket ownership flows
- customer file/fileset ownership APIs

## API Surface

- `POST /api/analyses/ingest`
- `GET /api/analyses/{analysis_euid}`
- `POST /api/analyses/{analysis_euid}/status`
- `POST /api/analyses/{analysis_euid}/artifacts`
- `POST /api/analyses/{analysis_euid}/review`
- `POST /api/analyses/{analysis_euid}/return`
- `GET /healthz`

Analysis write routes require `X-API-Key`.
Ingest and return also require `Idempotency-Key`.

## Contracts

Ingest input references (`input_references`) support exactly:

- raw S3 object URI: `{"reference_type":"s3_uri","value":"s3://..."}`
- Dewey artifact reference: `{"reference_type":"artifact_euid","value":"AT-..."}`
- Dewey artifact set reference: `{"reference_type":"artifact_set_euid","value":"AS-..."}`

Ursa validates fetchability for raw `s3://` inputs (`head_object` visibility check).

Artifact add (`POST /api/analyses/{analysis_euid}/artifacts`) supports exactly one of:

- `artifact_euid` (resolved via Dewey)
- `storage_uri` + `artifact_type` (must be in `URSA_INTERNAL_OUTPUT_BUCKET`, then registered in Dewey)

Atlas return enforces:

- review state must be `APPROVED`
- all artifacts must carry a Dewey link (`dewey_artifact_euid`)

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

Cross-system integrations are HTTPS-only and authenticated.

## Validation

```bash
pytest -q
```
