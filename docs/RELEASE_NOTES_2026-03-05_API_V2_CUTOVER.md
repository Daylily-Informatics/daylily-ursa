# Release Notes: API v2 Hard Cutover

Date: 2026-03-05

## Summary

This release removes API v1 and all unversioned API routes, and standardizes API access on `/api/v2/*`.

## Breaking Changes

- Removed all `/api/v1/*` endpoints.
- Removed `/v1/*` mirrored route registration.
- Removed unversioned API endpoints under `/api/*`.
- Removed root API-like paths:
  - `/worksets*`
  - `/queue/stats`
  - `/scheduler/stats`
- Canonical API contract is now strictly `/api/v2/*`.

## GUI Cleanup Included

- Removed stale links from workset detail page:
  - `/portal/worksets/{workset_id}/results/vcf`
  - `/portal/worksets/{workset_id}/results/bam`
  - `/portal/worksets/{workset_id}/results/qc`
  - `/portal/worksets/{workset_id}/results/all`
- Removed stale subject detail navigation:
  - `/portal/biospecimen/subjects/{subject_id}`

## Required Client Action

- Migrate all API consumers to `/api/v2/*` using the mapping in:
  - `docs/API_V1_TO_V2_MIGRATION.md`
