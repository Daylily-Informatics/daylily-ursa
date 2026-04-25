# Ursa Atlas Return Contract

## Purpose

Atlas remains the customer-facing results authority. Ursa returns analysis completion, review state, and artifact references after resolving the canonical test fulfillment item identity through Bloom and registering or resolving artifacts through Dewey.

## Endpoint

- `POST /api/integrations/ursa/v1/fulfillment-items/{fulfillment_item_euid}/analysis-results`

## Request Rules

- Authenticate with an Atlas integration bearer token provisioned from Atlas API clients.
- Use opaque EUIDs only in the request body.
- Forward `X-Request-ID` when available so Atlas can echo and log the same request identity.
- Send `Idempotency-Key`; repeated keys must be replay-safe.
- Do not send Atlas-private Bloom IDs or Ursa UUIDs.
- Do not return results unless Ursa review state is `APPROVED`.

The request includes:

- `atlas_tenant_id`
- `atlas_trf_euid`
- `atlas_test_euid`
- `atlas_test_fulfillment_item_euid`
- `analysis_euid`
- `run_euid`
- `sequenced_library_assignment_euid`
- `flowcell_id`
- `lane`
- `library_barcode`
- `analysis_type`
- `result_status`
- `review_state`
- `result_payload`
- `artifacts[]`
- `source_system = daylily-ursa`

Each returned artifact must carry a Dewey artifact EUID. Ursa rejects return when reviewed output artifacts do not have Dewey references.

## Atlas Behavior

- Find or create the fulfillment run projection for the resolved TRF/Test/fulfillment-item context.
- Register artifact metadata through Atlas release services.
- Attach artifact EUIDs through graph-linked artifact reference objects.
- Return fulfillment run EUID, fulfillment output EUID, and artifact EUIDs.

## Ursa Behavior

- Resolve Bloom run assignment context during ingest.
- Register or resolve output artifacts through Dewey before Atlas return.
- Enforce approved review state before return.
- Persist the Atlas return response on the analysis record.
- Treat repeated idempotency keys as replay-safe success.
