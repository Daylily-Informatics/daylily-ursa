# Ursa Mounted TapDB Admin Execution Plan

## Goal

Expose TapDB admin surfaces inside Ursa's FastAPI process under `/admin/tapdb`,
gated solely by Ursa's authenticated admin session, without a separate TapDB
auth flow.

## Implementation Steps

1. Add a dedicated Ursa integration module to:
   - lazy load TapDB admin app (`admin.main:app`)
   - force mounted-mode TapDB auth bypass env vars
   - enforce Ursa admin session gate before forwarding requests
2. Mount TapDB from `daylib_ursa.workset_api.create_app` after Ursa routes are
   composed.
3. Add minimal settings:
   - `URSA_TAPDB_MOUNT_ENABLED` (default true)
   - `URSA_TAPDB_MOUNT_PATH` (default `/admin/tapdb`)
4. Fail fast when mount is enabled and TapDB admin app cannot be imported.
5. Add tests for:
   - route existence
   - unauthenticated redirect
   - non-admin 403
   - admin access
   - mounted-mode TapDB local-auth bypass
   - startup failure behavior
6. Update docs (`README.md`, `docs/INTEGRATION_GUIDE.md`) to describe mounted
   behavior and standalone vs mounted mode.

## Acceptance Criteria

- Ursa starts one FastAPI app containing `/admin/tapdb`.
- Mounted TapDB routes are inaccessible without Ursa admin session.
- TapDB local auth flow is bypassed in mounted mode.
- Tests pass for mounted admin-only behavior and startup policy.
