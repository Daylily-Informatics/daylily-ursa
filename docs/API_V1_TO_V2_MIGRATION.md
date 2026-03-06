# API v1 to v2 Migration Map

Date: 2026-03-05

This document defines the hard-cutover mapping used in the v2-only release.
All legacy `/api/v1/*`, unversioned `/api/*`, and root API-like paths are removed.

## Global Path Mapping Rules

| Legacy path pattern | New path pattern |
|---|---|
| `/api/v1/{rest}` | `/api/v2/{rest}` |
| `/api/{rest}` | `/api/v2/{rest}` |
| `/customers{rest}` | `/api/v2/customers{rest}` |
| `/worksets{rest}` | `/api/v2/worksets{rest}` |
| `/queue/stats` | `/api/v2/queue/stats` |
| `/scheduler/stats` | `/api/v2/scheduler/stats` |

## High-Impact Endpoint Examples

| Legacy endpoint | New endpoint |
|---|---|
| `POST /api/v1/auth/change-password` | `POST /api/v2/auth/change-password` |
| `GET /api/v1/auth/tokens` | `GET /api/v2/auth/tokens` |
| `POST /api/v1/auth/tokens` | `POST /api/v2/auth/tokens` |
| `DELETE /api/v1/auth/tokens/{token_id}` | `DELETE /api/v2/auth/tokens/{token_id}` |
| `PATCH /api/v1/customers/{customer_id}` | `PATCH /api/v2/customers/{customer_id}` |
| `GET /api/spot-market/status` | `GET /api/v2/spot-market/status` |
| `POST /api/estimate-cost` | `POST /api/v2/estimate-cost` |
| `POST /worksets/validate` | `POST /api/v2/worksets/validate` |
| `POST /worksets/generate-yaml` | `POST /api/v2/worksets/generate-yaml` |
| `PATCH /api/files/{file_id}` | `PATCH /api/v2/files/{file_id}` |
| `GET /api/portal/search` | `GET /api/v2/portal/search` |
| `GET /api/monitor/status` | `GET /api/v2/monitor/status` |
| `GET /api/monitor/logs` | `GET /api/v2/monitor/logs` |

## Caller Migration Checklist

- Update all frontend `fetch()`/XHR calls to `/api/v2/*`.
- Update API client wrappers in `static/js/api.js`.
- Update tests to use `/api/v2/*`.
- Verify no references remain to `/api/v1/*`, `/api/*` (unversioned), `/worksets*`, `/queue/stats`, or `/scheduler/stats`.
