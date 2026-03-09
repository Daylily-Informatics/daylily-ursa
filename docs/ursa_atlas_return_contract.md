# Ursa Atlas Return Contract

## Purpose

Atlas remains the customer-facing results authority. Ursa returns analysis completion, review state, and artifact references after resolving the canonical test fulfillment item identity through Bloom.

## Endpoint

- `POST /api/integrations/ursa/v1/fulfillment-items/{fulfillment_item_euid}/analysis-results`

## Request Rules

- authenticated with the Atlas internal API key
- request body uses opaque EUIDs only
- request includes:
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
  - `source_system` = `daylily-ursa`
- request is idempotent by `Idempotency-Key`

## Atlas Behavior

- find or create the fulfillment run projection for the resolved TRF/Test/fulfillment-item context
- register artifact metadata through Atlas release services
- attach artifact EUIDs through graph-linked artifact reference objects
- return fulfillment run EUID, fulfillment output EUID, and artifact EUIDs

## Ursa Behavior

- do not send Atlas private Bloom IDs or Ursa UUIDs
- do not return results unless review state is `APPROVED`
- persist the Atlas return response on the analysis record
- treat repeated idempotency keys as replay-safe success
