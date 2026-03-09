# Customer Portal Runtime

This document describes the current Ursa customer portal behavior after the TapDB-first restore.

## Core Principles

- TapDB is the persistent authority for portal objects and canonical EUIDs.
- S3 buckets/files remain external resources; their links and metadata are tracked in TapDB.
- Legacy IDs may appear only as external keys in `json_addl`.
- No DynamoDB or non-TapDB SQL persistence is used for portal/file/manifest/workset restore scope.

## Object Model

- Customer/account: `actor/customer/account/1.0/`
- Linked bucket: `data/storage/s3-bucket-link/1.0/`
- Registered file: `data/file/registered/1.0/`
- File set: `data/file/fileset/1.0/`
- Manifest: `data/manifest/stage-samples/1.0/`
- Workset: `workflow/workset/analysis-request/1.0/`

All object ownership/scoping is represented through TapDB lineage (`owns`, `contains`, `uses_manifest`, etc.).

## Onboarding Flow

`ensure_customer_onboarding()` runs during auth callback:

1. Ensures a customer/account exists.
2. Reuses existing linked primary bucket when present.
3. If identity already has `s3_bucket`, links it as primary without provisioning.
4. If no bucket exists, provisions a region-correct bucket, tags it, and links it.
5. Returns an identity payload with the resolved `s3_bucket`.

## API Surfaces

- Portal UI: `/portal/...`
- Files: `/api/files/...`
- Manifests: `/api/customers/{customer_id}/manifests...`
- Worksets: `/api/customers/{customer_id}/worksets...`

Compatibility aliases are provided for high-use legacy file routes (for example `/api/files/list`, `/api/files/{file_id}`, `/api/files/{file_id}/tags`, `/api/files/{file_id}/manifest`) and are thin wrappers over TapDB-backed services.

## Manifest + Workset Behavior

- Manifest template download: `GET /api/files/manifest/template`
- Save/list/get/download customer manifests via `/api/customers/{customer_id}/manifests...`
- Workset creation supports:
  - `manifest_id` (link existing manifest)
  - `manifest_tsv_content` (create manifest, then link)

## Bucket Mutation Guardrails

Mutating operations enforce:

- bucket exists and belongs to customer
- `read_only` and `can_write` constraints
- `prefix_restriction` constraints
- delete-file rejection for already registered files
