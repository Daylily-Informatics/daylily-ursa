# Ursa TapDB/Postgres Refactor Design

## 1. Scope and decision summary
Recommended path (single endorsed option):
- Migrate Ursa workflow state from TapDB to a dedicated Ursa-owned Postgres database, using domain-specific relational workflow tables, with TapDB conventions (EUID, audit discipline, explicit transaction ownership), and a finite phased cutover (no indefinite dual-write).

Explicit assumptions:
- Ursa keeps an independent Postgres instance and does not share tables or database instances with Atlas or Bloom.
- Cross-system integration is API and/or event driven only.
- Atlas remains the human authentication surface and Ursa aligns RBAC/tenant semantics to Atlas patterns.
- Atlas and TapDB repos are locally available and inspected in this design.
- Existing S3-based workflow side effects remain in scope during migration and require explicit idempotency and outbox coordination.

Explicit non-goals:
- Do not force Ursa runtime workflow state into TapDB generic tables (`generic_template`, `generic_instance`, `generic_instance_lineage`).
- Do not implement semantic/vector search or unrelated feature work in this migration.
- Do not introduce indefinite dual-write between TapDB and Postgres.
- Do not introduce direct DB-table sharing between Ursa and Atlas/Bloom.

## 2. Evidence and repo inspection
Repos inspected locally:
- Ursa: `/Users/jmajor/projects/daylily/daylily-ursa`
- Atlas: `/Users/jmajor/projects/lsmc/lsmc-atlas`
- TapDB: `/Users/jmajor/projects/daylily/daylily-tapdb`

Key files inspected (representative):
- Ursa app assembly and routes:
  - `daylib/workset_api.py` (`create_app`)
  - `daylib/workset_api_cli.py` (`main`)
  - `daylib/routes/worksets.py` (`create_worksets_router`)
  - `daylib/routes/customer_worksets.py` (`create_customer_worksets_router`)
  - `daylib/routes/portal.py` (`create_portal_router`, `/portal/login`, `/auth/callback`, `/portal/register`, `/portal/search`, `/api/portal/search`)
  - `daylib/routes/monitoring.py` (`create_monitoring_router`)
- Ursa persistence/lifecycle:
  - `daylib/workset_state_db.py` (`WorksetState`, `WorksetStateDB.register_workset`, `acquire_lock`, `release_lock`, `update_state`, `record_failure`, `get_retryable_worksets`, `reset_for_retry`, `list_worksets_by_customer`, `archive_workset`, `delete_workset`, `restore_workset`)
  - `daylib/file_registry.py` (`register_file`, `create_fileset`, `list_customer_files`, `find_file_by_s3_uri`, `record_file_workset_usage`)
  - `daylib/manifest_registry.py` (`save_manifest`, `list_customer_manifests`)
  - `daylib/workset_customer.py` (`CustomerManager`, `onboard_customer`, `_save_customer_config`)
  - `daylib/workset_monitor.py` (`WorksetMonitor.run`, `_attempt_acquire`, `_release_workset_lock`, `process_workset`)
  - `daylib/workset_worker.py` (`process_workset`, `main`)
  - `daylib/workset_concurrent_processor.py` (`process_cycle`, `_process_single_workset`)
  - `daylib/workset_scheduler.py` (`WorksetScheduler.get_next_workset`, `schedule_workset`, `get_scheduling_stats`)
- Ursa config/startup/auth checks:
  - `daylib/config.py` (`Settings`)
  - `daylib/ursa_config.py` (`UrsaConfig` and region/cognito accessors)
  - `daylib/cli/server.py` (`start`, `_validate_cognito_oauth_uris`, `_describe_cognito_app_client`)
  - `daylib/routes/dependencies.py` (`verify_workset_ownership`, `verify_workset_access`)
- Atlas patterns:
  - `app/auth/rbac.py` (`Role`, `Permission`, `ROLE_PERMISSIONS`, `can_access_tenant`)
  - `app/auth/dependencies.py` (`CurrentUser`, `get_current_user`, `get_current_tenant`, `require_role`, `require_permission`, `get_bloom_integration_client`)
  - `app/api/routes/internal.py` (`verify_internal_api_key`, internal routes)
- TapDB patterns:
  - `daylily_tapdb/connection.py` (`TAPDBConnection.session_scope`, `_set_session_username`)
  - `daylily_tapdb/cli/db_config.py` (`_strict_namespace_enabled`, `get_db_config_for_env`)
  - `schema/tapdb_schema.sql` (EUID triggers, `audit_log`, soft-delete triggers)

Atlas/TapDB local availability:
- Available and inspected locally.

Blocked/unverified areas:
- No current Ursa implementation of Atlas status reporting for `lsmc_work_order` found in local Ursa sources (`rg` across `daylib`, `tests`, `docs` found no `lsmc_work_order` symbol).
- No current Ursa internal endpoint/service-auth contract for Atlas-triggered direct analysis found; this is design-required and must be introduced.

## 3. Current Ursa architecture
### Framework/app shape
- FastAPI app factory is in `daylib/workset_api.py:create_app`.
- CLI startup path creates dependencies then app in `daylib/workset_api_cli.py:main` and serves via `uvicorn.run`.
- Operational server command path is `daylib/cli/server.py:start` (HTTPS defaults, auth preflight, process management).

### Route and API surface
- Core workset API surface: `daylib/routes/worksets.py:create_worksets_router` (`/worksets`, `/worksets/{id}`, `/worksets/{id}/lock`, `/queue/stats`, `/scheduler/stats`, `/worksets/next`).
- Customer-scoped lifecycle API: `daylib/routes/customer_worksets.py:create_customer_worksets_router` (`/api/customers/{customer_id}/worksets`, retry/archive/delete/restore/logs/performance routes).
- Portal/auth/API mix: `daylib/routes/portal.py:create_portal_router` (`/portal/login`, `/auth/callback`, `/portal/register`, `/portal/search`, `/api/portal/search`, admin/monitor pages).
- Monitoring/admin route module: `daylib/routes/monitoring.py:create_monitoring_router`.
- Versioned mirror route assembly under `/v1` in `daylib/workset_api.py:create_app`.

### Current persistence model
- Primary workflow state in TapDB via `daylib/workset_state_db.py:WorksetStateDB`.
- Customer records in TapDB via `daylib/workset_customer.py:CustomerManager` (`daylily-customers`).
- File registry and filesets in TapDB via `daylib/file_registry.py:FileRegistry`.
- Manifests in TapDB via `daylib/manifest_registry.py:ManifestRegistry`.
- Current config surface still describes TapDB-first tables in `daylib/config.py:Settings` (`workset_table_name`, `customer_table_name`, `daylily_manifest_table`, `daylily_linked_buckets_table`).

### State machine and lifecycle model
- State enum includes `ready`, `in_progress`, `error`, `complete`, `retrying`, `failed`, `canceled`, `archived`, `deleted` in `daylib/workset_state_db.py:WorksetState`.
- `register_workset` initializes `state_history` inline and uses conditional create (`ConditionExpression="attribute_not_exists(workset_id)"`).
- `update_state` mutates current state and appends to inline `state_history` via `list_append`.
- Retry logic is in `record_failure`, `get_retryable_worksets`, and `reset_for_retry`.
- Archive/delete semantics are implemented by `archive_workset`, `delete_workset` (soft/hard), `restore_workset`.

### Scheduler/worker/monitor behavior
- Scheduler (`daylib/workset_scheduler.py:WorksetScheduler`) pulls ready queue from `state_db`, chooses a workset/cluster, and reports queue stats.
- Worker loop (`daylib/workset_worker.py:main`) claims ready worksets, calls `process_workset`, updates state, and releases locks.
- Concurrent processor (`daylib/workset_concurrent_processor.py:_process_single_workset`) acquires lock, updates state, executes, records failure/complete, then releases lock.
- Monitor daemon (`daylib/workset_monitor.py:WorksetMonitor.run`) uses S3 sentinel files plus TapDB locking/state updates and performs side effects (pipeline invocation, S3 exports, optional notifications).

### Auth model
- Ursa uses optional Cognito auth via daylily-cognito imports and combined auth dependency in `daylib/workset_api.py:create_app` (`get_current_user` supports session, JWT bearer, API key).
- Portal hosted-UI login and callback logic is in `daylib/routes/portal.py` (`portal_login`, `portal_auth_callback`).
- Startup preflight checks Cognito OAuth app-client name/URIs in `daylib/cli/server.py:_validate_cognito_oauth_uris` and `start`.

### Tenant model
- Current tenant boundary is customer-centric (`customer_id` string), not Atlas UUID tenant model.
- Ownership checks are customer-based in `daylib/routes/dependencies.py:verify_workset_ownership` and `verify_workset_access`.
- Route-level filtering in customer workset endpoints uses `verify_workset_ownership`.

### Integration surfaces
- S3 sentinel and artifact interactions are pervasive in `daylib/workset_monitor.py` and `daylib/routes/portal.py`/`daylib/routes/customer_worksets.py`.
- Optional SNS/Linear notifications in `daylib/workset_notifications.py`.
- Cluster orchestration side effects use `pcluster` subprocess invocations in `daylib/workset_monitor.py`.
- No current Atlas status outbox/event sync implementation found in Ursa source.

### Current testing coverage
- State/persistence tests: `tests/test_workset_state_db.py`, `tests/test_integration.py`, `tests/test_workset_concurrent_processor.py`.
- Customer/tenant filtering tests: `tests/test_workset_customer.py`, `tests/test_portal_authz_matrix.py`.
- Auth/portal tests: `tests/test_optional_auth.py`, `tests/test_portal_account_security.py`, `tests/test_cli_cognito.py`.
- Registry tests: `tests/test_file_registry.py`, `tests/test_manifest_registry.py`.
- Monitor/notification tests: `tests/test_monitor_dashboard.py`, `tests/test_workset_notifications.py`.

## 4. Atlas patterns to mirror
Tenant scoping and RBAC:
- Atlas role/permission graph: `app/auth/rbac.py` (`Role`, `Permission`, `ROLE_PERMISSIONS`).
- Cross-tenant gating: `app/auth/rbac.py:can_access_tenant` and `Permission.CROSS_TENANT_READ`.
- User context carries tenant and roles: `app/auth/dependencies.py:CurrentUser` (`tenant_id`, `roles`).

Auth dependency style:
- Route dependencies enforce role/permission checks (`require_role`, `require_permission`, `require_internal[_api]`, `require_org_admin[_api]` in `app/auth/dependencies.py`).
- Atlas supports session and token modes in auth dependency layer (`get_current_user`, `get_current_user_or_token`).

ID conventions:
- Atlas internal request models use UUID tenant and resource ids in `app/api/routes/internal.py` (`tenant_id: uuid.UUID`, `order_id`, `test_order_id`, etc.).

Error/API response conventions:
- Standard FastAPI `HTTPException` with explicit 401/403/404/400 responses in `app/auth/dependencies.py` and `app/api/routes/internal.py`.
- Typed request models with explicit fields and validation patterns in `app/api/routes/internal.py`.

Service-to-service auth:
- Machine auth pattern exists via `verify_internal_api_key` in `app/api/routes/internal.py`.
- Scoped integration-client validation pattern exists in `app/auth/dependencies.py:get_bloom_integration_client` (permission and allowed endpoint checks).

## 5. TapDB conventions to adopt
Audit and transaction ownership:
- Explicit transaction ownership and commit behavior in `daylily_tapdb/connection.py:TAPDBConnection.session_scope` (`commit` flag).
- Per-transaction audit attribution with `SET LOCAL session.current_username` in `_set_session_username`.

EUID strategy:
- Config-driven prefix table and trigger-generated EUIDs in `schema/tapdb_schema.sql` (`tapdb_identity_prefix_config`, `set_generic_*_euid`, `set_audit_log_euid`, trigger bindings).

Soft-delete strategy:
- Soft-delete trigger model sets `is_deleted` in schema trigger functions and `BEFORE DELETE` triggers (`schema/tapdb_schema.sql`).

Migration/bootstrap and multi-DB stance:
- Namespace-strict config and environment resolution in `daylily_tapdb/cli/db_config.py` (`_strict_namespace_enabled`, `get_db_config_for_env`).
- Configuration enforces explicit namespace context and scoped config files.

Adoption boundary for Ursa:
- Adopt conventions (audit fields, EUID generation policy, explicit session ownership, soft-delete semantics) without reusing TapDB generic runtime workflow tables for Ursa orchestration.

## 6. Recommended target architecture
Core decisions:
- Dedicated Ursa Postgres instance: yes.
- Shared DB with Atlas or Bloom: no.
- Cross-system interaction: API/events only.

Target architecture:
- Ursa API + workers read/write Ursa-owned relational workflow tables in Postgres.
- Ursa emits domain events into a transactional outbox table.
- Outbox delivery worker publishes Atlas status updates (including `lsmc_work_order` status contract) and other external notifications.
- Atlas-triggered direct analysis requests arrive through an Ursa internal API surface with machine auth + delegated identity context.

Tenant boundary model:
- Canonical tenant key in Ursa target: `tenant_id` (UUID-compatible with Atlas), with temporary bridge mapping from existing `customer_id` during migration.
- Every tenant-owned table carries `tenant_id` and enforces tenant-scoped uniqueness.

RBAC enforcement approach:
- Align role/permission semantics with Atlas role model (`INTERNAL_USER`, `EXTERNAL_USER_ADMIN`, `ADMIN`, `CROSS_TENANT_READ` semantics).
- Enforce role and permission checks in Ursa dependencies and service-layer guards, not only in route code.

Service-to-service auth pattern:
- Atlas-to-Ursa calls authenticate machine principal (internal API key/service token) and carry delegated actor context.
- Ursa validates machine credentials and delegated tenant/role context before creating or mutating workflow state.

Status reporting to Atlas for `lsmc_work_order`:
- Outbox event type `atlas.work_order.status_changed` emitted on workflow state transitions mapped to Atlas status vocabulary.
- Delivery worker retries with idempotency key, then DLQ table on repeated failure.

Atlas-triggered direct analysis jobs:
- New internal endpoint accepts Atlas-originated request with idempotency key + delegated actor claims.
- Endpoint creates `analysis_request` + initial `workset` record transactionally.

Internal cross-tenant roles:
- Internal and admin roles can cross tenant boundaries only via explicit permission checks (`cross_tenant:read` equivalent and write-scope controls).

Failure isolation model:
- State transition + lease update in one DB transaction.
- Side effects (Atlas status sync, optional SNS, other external calls) isolated via outbox and retry processors.

## 7. Proposed schema shape
Proposed core tables and semantics:

### `analysis_request`
- Purpose: durable intake record for upstream requests (portal/manual/API/internal Atlas-triggered).
- Tenant ownership: `tenant_id` required.
- Key uniqueness rules: unique `request_id` (EUID), optional unique `(tenant_id, upstream_source, upstream_request_id)`.
- Mutability: mutable status summary fields; immutable request payload snapshot column.
- Likely indexes: `(tenant_id, created_at DESC)`, `(tenant_id, status)`, unique upstream idempotency index.
- External references: links to Atlas work order identifier where applicable.

### `workset`
- Purpose: canonical workflow entity formerly represented by TapDB workset item.
- Tenant ownership: `tenant_id` required.
- Key uniqueness rules: unique `workset_id` (stable external id), unique internal `id` (UUID/EUID).
- Mutability: mutable current fields (state, lock, progress, pointers).
- Likely indexes: `(tenant_id, state, priority)`, `(state, priority, updated_at)`, `(lock_expires_at)`.
- External references: `analysis_request_id`, optional `atlas_work_order_id`.

### `workset_current` (or `workset_state_projection`)
- Purpose: optimized mutable projection for worker claims and portal listing.
- Tenant ownership: `tenant_id` required.
- Key uniqueness rules: unique FK to `workset.id`.
- Mutability: mutable by design; one row per workset.
- Likely indexes: `(tenant_id, state, updated_at)`, `(state, lock_expires_at)`, `(preferred_cluster)`.
- External references: FK to `workset`.

### `execution_attempt`
- Purpose: one row per processing attempt/retry execution window.
- Tenant ownership: `tenant_id` required.
- Key uniqueness rules: unique `(workset_id, attempt_number)`.
- Mutability: append-only per attempt lifecycle (status fields mutable until terminal, then frozen).
- Likely indexes: `(workset_id, attempt_number DESC)`, `(tenant_id, started_at DESC)`.
- External references: cluster/run metadata, tmux session id, results URI.

### `workset_event`
- Purpose: immutable event history replacing inline TapDB `state_history` append lists.
- Tenant ownership: `tenant_id` required.
- Key uniqueness rules: unique event id; monotonic sequence per workset (`workset_event_seq`).
- Mutability: append-only immutable.
- Likely indexes: `(workset_id, sequence)`, `(tenant_id, created_at DESC)`, `(event_type)`.
- External references: actor context id, correlated attempt id.

### `idempotency_key`
- Purpose: deduplicate create/retry/internal trigger requests and external callback handling.
- Tenant ownership: tenant-scoped for tenant requests; global scope for machine-only operations where needed.
- Key uniqueness rules: unique `(scope, key)` and optional `(tenant_id, operation, key)`.
- Mutability: mostly immutable; status/result pointer updatable.
- Likely indexes: unique indexes above + `(expires_at)` for cleanup.
- External references: links to created `analysis_request`/`workset`.

### `outbox_event`
- Purpose: transactional outbox for Atlas status sync and other external side effects.
- Tenant ownership: nullable for system events, populated for tenant events.
- Key uniqueness rules: unique `event_id`; dedupe key unique per destination where required.
- Mutability: mutable delivery status (`pending`, `in_flight`, `sent`, `dead_letter`).
- Likely indexes: `(status, next_attempt_at)`, `(destination, status)`, `(tenant_id, created_at)`.
- External references: payload points to `workset`/`execution_attempt`/Atlas identifiers.

### `artifact_output`
- Purpose: normalized metadata for results/artifacts (S3 URIs, checksums, type, size).
- Tenant ownership: `tenant_id` required.
- Key uniqueness rules: unique `(tenant_id, workset_id, artifact_type, uri)`.
- Mutability: append-only preferred (new revisions) or immutable rows with superseded flag.
- Likely indexes: `(tenant_id, workset_id)`, `(tenant_id, created_at DESC)`.
- External references: workset/execution attempt foreign keys.

### `actor_context`
- Purpose: normalized attribution for human/machine/delegated caller context.
- Tenant ownership: nullable for system-level events, otherwise tenant-bound.
- Key uniqueness rules: unique context id; optional dedupe on request correlation id.
- Mutability: immutable snapshot rows.
- Likely indexes: `(tenant_id, actor_type)`, `(external_subject_id)`.
- External references: referenced by `workset_event`, `analysis_request`, `outbox_event`.

## 8. TapDB to Postgres semantic mapping
| TapDB behavior/semantic | Where it exists today | Why it is risky | Proposed Postgres mapping | Required constraints/locks/indexes | Test cases needed |
|---|---|---|---|---|---|
| Conditional create (`attribute_not_exists`) | `WorksetStateDB.register_workset`, `FileRegistry.register_file`, `ManifestRegistry.save_manifest` | Duplicate creates under concurrency can split workflow identity | `INSERT ... ON CONFLICT DO NOTHING` with deterministic business keys | Unique index on external ids (`workset_id`, `file_id`, manifest id/upstream id) | Create race with N concurrent requests, assert single row + deterministic response |
| Conditional lock acquisition | `WorksetStateDB.acquire_lock` conditional `update_item` | Current state and lock updates are multi-call and not transactionally coupled to subsequent state transition | Single transaction: claim candidate row with `FOR UPDATE SKIP LOCKED`, set lease owner/expiry and state atomically | Index on `(state, lock_expires_at)`, check constraints for valid lease fields | Competing workers claim same workset; only one succeeds |
| Stale lease takeover | `acquire_lock` with `lock_expires_at`/fallback to `lock_acquired_at` | Clock skew and non-atomic stale checks can allow double-claim windows | Transactional takeover condition `WHERE lock_expires_at < now()` during claim | Partial index on expired leases; transaction isolation at READ COMMITTED + row lock | Expired lease takeover by second worker; first worker cannot continue transition |
| Inline history append (`state_history = list_append`) | `WorksetStateDB.update_state`, `record_failure` | Mutable document list can drift or exceed practical payload expectations | Immutable `workset_event` append with monotonic sequence; `workset_current` updated in same transaction | Unique `(workset_id, sequence)`, FK to workset | Sequence monotonicity and no missing events across retries/errors |
| Retry clone/retry attempt semantics | `customer_worksets.retry_customer_workset`, `record_failure`, `reset_for_retry` | Retry state and cloned workset relations are split across APIs and metadata fields | First-class `execution_attempt` rows plus optional `retries_from_workset_id` relation | Unique `(workset_id, attempt_number)`; FK consistency | Retry increments attempt number exactly once under concurrent triggers |
| Archive/delete semantics | `archive_workset`, `delete_workset` (soft/hard), `restore_workset` | Mixed soft/hard semantics can break lineage/audit | Soft delete + archive flags on core tables; hard delete restricted to retention jobs | `is_deleted`/`archived_at` indexed, FK with `ON DELETE RESTRICT` | Archive, restore, soft-delete, retention purge flows preserve audit |
| Mutable progress updates | `update_progress`, `update_progress_step` | Best-effort updates may overwrite critical state if unordered | Progress updates versioned by `updated_at` + optimistic guard (`version`) | Version column + check constraint on legal state/progress combos | Out-of-order progress writes are rejected or ignored deterministically |
| Query/index assumptions (GSI + scan fallback) | `list_worksets_by_customer` uses `customer-id-state-index` else scan | Scan fallback causes latency/cost and potential consistency surprises | Native relational indexes on `(tenant_id,state,updated_at)` and pagination by keyset | Composite indexes + covering indexes for common list patterns | Query parity for customer+state+limit pagination |
| Eventual-consistency assumptions between S3 sentinels and DB state | `WorksetMonitor._sync_workset_state` and sentinel logic | Divergent source-of-truth can cause duplicate or stuck worksets | DB becomes authoritative for lifecycle; S3 sentinels become derived/diagnostic | DB transition transaction + reconciliation job constraints | Simulate sentinel/DB divergence and verify deterministic recovery |
| Idempotent external calls | File URI dedupe in `FileRegistry.find_file_by_s3_uri`; no formal outbox for Atlas | Retries can duplicate external side effects (status pushes, notifications) | `idempotency_key` + `outbox_event` with destination dedupe keys | Unique `(destination, dedupe_key)` on outbox | Replayed delivery attempts yield one external side effect |

## 9. Tenant boundary and RBAC
Canonical tenant identifier:
- Target canonical key: `tenant_id` UUID aligned to Atlas (`app/auth/dependencies.py:CurrentUser.tenant_id`).
- Migration bridge: maintain mapping from legacy Ursa `customer_id` to canonical `tenant_id` during phased cutover.

Where `tenant_id` must live:
- Required on all tenant-owned workflow tables: `analysis_request`, `workset`, `workset_current`, `execution_attempt`, `workset_event`, `artifact_output`, tenant-scoped `idempotency_key`, tenant-scoped `outbox_event`.

Internal cross-tenant access:
- Allow only through explicit role/permission checks mirroring Atlas semantics (`Permission.CROSS_TENANT_READ`, internal/admin roles from `app/auth/rbac.py`).

Atlas role semantics mapping into Ursa checks:
- Ursa dependency layer should evolve from customer checks (`verify_workset_ownership`, `verify_workset_access`) to role/permission+tenant checks similar to Atlas `require_permission`/`require_role`.

How local customer-centric auth must be replaced/bridged:
- Bridge period: translate existing session/JWT customer identifiers into `tenant_id` through mapping table/service.
- End state: route/service checks key off `tenant_id` + role permissions, not bucket/customer-name heuristics.

## 10. Service-to-service auth and delegated identity
How Atlas calls Ursa:
- Machine-authenticated internal Ursa endpoint(s), analogous to Atlas `verify_internal_api_key` pattern.

How Ursa validates caller:
- Validate machine credential (API key/service token) and enforce endpoint allowlist and permission scope, analogous to Atlas `get_bloom_integration_client`.

How delegated end-user context is conveyed:
- Include delegated actor claims (`actor_sub`, `actor_email`, `actor_roles`, `actor_tenant_id`, `trace_id`) in signed token or signed headers.
- Persist delegated identity snapshot in `actor_context` and reference in workflow events.

How scopes/roles/tenant are represented:
- Machine token scope controls max privileges; delegated roles are intersected with machine scope before authorization.

Replay/expiry expectations:
- Require short-lived credentials, nonce/jti, and idempotency key on mutating internal requests.
- Store recent token IDs / request IDs for replay detection window.

How machine auth differs from browser/session auth:
- Browser/session remains for human portal flows.
- Machine auth required for Atlas-triggered analysis and status callback channels; no session cookies in service-to-service path.

## 11. Idempotency and concurrency
Idempotency-key strategy:
- Require idempotency keys for all create/retry/internal-trigger POST endpoints.
- Persist key + operation + tenant scope + resulting resource reference.

Business uniqueness constraints:
- Unique `workset_id` per tenant/domain.
- Unique upstream request key per source (`tenant_id + upstream_source + upstream_request_id`).
- Unique outbox destination dedupe keys.

Row-lock strategy:
- Worker claim with `SELECT ... FOR UPDATE SKIP LOCKED` on ready rows.

Advisory-lock strategy:
- Use PG advisory locks only for coarse-grained operations (single-workset retry orchestration) where row lock coverage is insufficient.

Worker claim strategy:
- One transaction performs claim + lease fields + state transition to in-progress + attempt row insert.

State transition + lease update:
- Must be one transaction; never split claim and transition into separate best-effort calls.

What should never rely on best-effort updates:
- State transition ordering.
- Lease ownership changes.
- External status sync intent recording (must be transactional via outbox row).

## 12. Auditability and side-effect isolation
DB audit model:
- Per-table `created_at`, `updated_at`, `created_by`, `updated_by`, plus optional soft-delete fields.
- Trigger-backed audit rows or append-only domain events for all lifecycle changes.

Domain event model:
- Immutable `workset_event` records every state/progress/attempt transition with actor context.

Outbox model:
- Transactional outbox (`outbox_event`) written in same transaction as state changes requiring external effects.

Atlas status sync model:
- Dedicated outbox consumer maps internal state to Atlas `lsmc_work_order` status and retries with dedupe keys.

Retry and dead-letter behavior:
- Exponential backoff with bounded attempts.
- Dead-letter rows retained for operator replay tooling.

Actor attribution:
- All mutating operations capture machine/human/delegated actor in `actor_context` or audit columns.

## 13. Migration and cutover strategy
Endorsed path:
- Finite phased cutover with one-time backfill, bounded verification period, and no indefinite dual-write.

Cutover vs dual write decision:
- Choose cutover (with temporary validation reads), not dual-write.

Backfill approach:
- Snapshot TapDB tables (`worksets`, `customers`, files/manifests as needed).
- Transform to relational model with deterministic id mapping and event-history reconstruction.
- Validate counts, key uniqueness, and sampled record equivalence.

Verification approach:
- Run parity harness comparing key semantics (create/lock/retry/archive/list filters/idempotency) against both stores in non-production rehearsal and staging.

Cutover sequencing:
1. Deploy schema + repositories + feature flags off.
2. Run backfill and verification.
3. Enable Postgres writes for non-critical paths in staging; run parity suite.
4. Freeze TapDB mutating paths briefly for production cutover window.
5. Enable Postgres primary writes and reads.
6. Keep TapDB read-only fallback toggle for bounded rollback window.
7. De-scope TapDB access after acceptance gates.

Rollback approach:
- Feature-flag route back to TapDB primary within bounded window.
- Preserve outbox/event records for replay after rollback.
- Avoid bidirectional dual-write during rollback; use one store as source of truth at a time.

Feature-flag strategy:
- Flags for repository backend selection, worker claim backend, outbox delivery, and Atlas integration endpoints.

Handling in-flight worksets at cutover:
- Either drain active worksets before cutover or migrate in-flight leases/attempts with explicit reconciliation job.
- Cutover gate should require no ambiguous in-progress ownership between stores.

## 14. Environment and infrastructure changes
Environment variables (new/changed likely):
- New DB connectivity: `URSA_DB_URL` (or split host/port/db/user/password), `URSA_DB_POOL_SIZE`, `URSA_DB_MAX_OVERFLOW`.
- New migration/runtime toggles: `URSA_WORKFLOW_BACKEND` (`tapdb|postgres`), `URSA_PARITY_MODE`, `URSA_OUTBOX_ENABLED`.
- New Atlas integration: `URSA_ATLAS_BASE_URL`, `URSA_ATLAS_SERVICE_TOKEN` (or API key), `URSA_ATLAS_TIMEOUT_SECONDS`.
- Existing settings likely retained temporarily: TapDB table names and auth/cognito settings in `daylib/config.py:Settings`.

Secrets:
- Postgres credentials/DSN.
- Atlas service auth secret.
- Signing secret/key for delegated identity tokens if JWT-based.

Postgres/TapDB bootstrap:
- Add Ursa-specific schema bootstrap and migrations.
- Optionally adopt TapDB-style prefix config table if EUID trigger generation is implemented in Ursa schema.

Migration tooling:
- Add migration runner (e.g., Alembic or equivalent) and backfill/verifier CLI commands.

Deployment manifests:
- API/worker deployments need DB env injection and outbox consumer process.

Background workers:
- Split workflow worker and outbox delivery worker processes.

Monitoring/alerts:
- Add DB lock contention, outbox lag, dead-letter count, status-sync failure metrics.

Decommissioning/narrowing TapDB usage:
- Phase to read-only fallback then remove runtime writes and finally remove TapDB dependencies from primary workflow path.

S3/SNS changes:
- Keep S3 pipeline/result side effects.
- SNS can remain optional notification channel but should be triggered via outbox/event handling for idempotency.

Atlas integration configuration:
- Configure allowed endpoint scopes and service principal identity for Atlas→Ursa and Ursa→Atlas calls.

## 15. Missing tests and required new tests
Unit tests needed:
- Repository transition legality and state machine guards.
- RBAC permission mapping and tenant access decisions aligned to Atlas role semantics.
- Idempotency key acceptance/replay behavior.

Persistence tests needed:
- Unique constraint behavior under duplicate create.
- Row-lock claim semantics (`FOR UPDATE SKIP LOCKED`) under concurrency.
- Event append ordering and projection updates in one transaction.

Migration/backfill tests needed:
- Deterministic mapping from TapDB items to relational rows/events.
- Checksum/count parity per tenant/state.
- Backfill restartability/idempotency.

Concurrency tests needed:
- Competing worker claims, stale lease takeover, retry race conditions.
- In-flight cutover rehearsal tests.

Auth/RBAC tests needed:
- Session + token + machine auth paths.
- Cross-tenant checks for internal roles vs external roles.
- Delegated identity scope intersection behavior.

Atlas-Ursa integration tests needed:
- Atlas-triggered direct analysis request contract.
- Ursa outbox status delivery to Atlas `lsmc_work_order` with retry/dedupe.

Cutover/rollback tests needed:
- Feature-flag flip safety.
- Rollback to TapDB without split-brain ownership.

Current gaps (repo-observed):
- No existing Ursa tests for Atlas service-auth contract or Atlas status outbox flows.
- No Postgres persistence/concurrency suite exists yet in Ursa test tree.

## 16. Risks, open questions, and explicit non-assumptions
Top risks:
- Concurrency regressions when replacing TapDB conditional semantics with SQL transactions.
- Split-brain ownership during cutover if TapDB and Postgres both mutate workflow state.
- Tenant/RBAC drift if customer-centric checks are only partially replaced.
- External side-effect duplication without strict outbox + idempotency controls.
- In-flight workset migration complexity (locks, retries, sentinel reconciliation).

Open questions:
- Exact Atlas `lsmc_work_order` status API contract (endpoint, payload shape, auth scheme) is not present in local Ursa repo and requires integration contract confirmation.
- Canonical mapping source for legacy `customer_id` to Atlas `tenant_id` needs explicit ownership and migration policy.
- Final EUID prefix scheme for Ursa workflow entities requires product/ops decision.

Explicit non-assumptions:
- Do not assume TapDB generic tables are suitable for Ursa workflow orchestration.
- Do not assume existing Ursa portal session model is sufficient for Atlas machine-triggered job submission.
- Do not assume current S3 sentinel state should remain primary source of truth post-migration.

## 17. Exact files likely to change in implementation
Current files likely to change:
- `daylib/config.py`
- `daylib/ursa_config.py`
- `daylib/workset_api.py`
- `daylib/workset_api_cli.py`
- `daylib/cli/server.py`
- `daylib/routes/worksets.py`
- `daylib/routes/customer_worksets.py`
- `daylib/routes/portal.py`
- `daylib/routes/dependencies.py`
- `daylib/routes/monitoring.py`
- `daylib/workset_state_db.py`
- `daylib/workset_scheduler.py`
- `daylib/workset_worker.py`
- `daylib/workset_concurrent_processor.py`
- `daylib/workset_monitor.py`
- `daylib/workset_customer.py`
- `daylib/file_registry.py`
- `daylib/manifest_registry.py`
- `tests/test_workset_state_db.py`
- `tests/test_workset_concurrent_processor.py`
- `tests/test_workset_customer.py`
- `tests/test_portal_authz_matrix.py`
- `tests/test_optional_auth.py`

Likely new files/directories:
- `daylib/db/engine.py`
- `daylib/db/session.py`
- `daylib/db/models/workflow.py`
- `daylib/db/repositories/workset_repository.py`
- `daylib/db/repositories/analysis_request_repository.py`
- `daylib/db/repositories/outbox_repository.py`
- `daylib/services/rbac.py`
- `daylib/services/tenant_mapping.py`
- `daylib/services/atlas_client.py`
- `daylib/routes/internal_atlas.py`
- `daylib/workers/outbox_dispatcher.py`
- `daylib/migrations/` (migration framework + revisions)
- `scripts/backfill_tapdb_to_postgres.py`
- `scripts/verify_tapdb_postgres_parity.py`
- `scripts/cutover_rehearsal.py`
- `tests/persistence/test_postgres_workset_repository.py`
- `tests/concurrency/test_worker_claims.py`
- `tests/integration/test_atlas_status_sync.py`
- `tests/migration/test_backfill_parity.py`
