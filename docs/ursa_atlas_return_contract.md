# Ursa Atlas Return Contract

## Purpose

Atlas remains the customer-facing results authority. Ursa returns analysis completion, review state, and artifact references after resolving the test identity through Bloom.

## Endpoint

- `POST /api/integrations/ursa/v1/test-orders/{test_order_euid}/analysis-results`

## Request Rules

- authenticated with the Atlas internal API key
- request body uses opaque EUIDs only
- request includes:
  - `atlas_tenant_id`
  - `atlas_order_euid`
  - `atlas_test_order_euid`
  - `analysis_euid`
  - `run_euid`
  - `index_string`
  - `analysis_type`
  - `result_status`
  - `review_state`
  - `result_payload`
  - `artifacts[]`
  - `source_system` = `daylily-ursa`
- request is idempotent by `Idempotency-Key`

## Atlas Behavior

- find or create the assay run projection for the resolved order/test-order pair
- register artifact metadata through Atlas release services
- attach artifact EUIDs to the assay result revision
- return assay run EUID, assay result EUID, and artifact EUIDs

## Ursa Behavior

- do not send Atlas private Bloom IDs or Ursa UUIDs
- persist the Atlas return response on the analysis record
- treat repeated idempotency keys as replay-safe success
