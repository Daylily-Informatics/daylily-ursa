# Ursa Refactor PR Plan

## 1. Planning assumptions
Repo-verified assumptions:
- Ursa API assembly and route composition are in `daylib/workset_api.py:create_app` with route modules in `daylib/routes/*`.
- Current workflow persistence is DynamoDB-centric via `daylib/workset_state_db.py:WorksetStateDB`, `daylib/workset_customer.py:CustomerManager`, `daylib/file_registry.py:FileRegistry`, and `daylib/manifest_registry.py:ManifestRegistry`.
- Current worker/scheduler/monitor lifecycle mutation paths are in `daylib/workset_worker.py`, `daylib/workset_concurrent_processor.py`, `daylib/workset_monitor.py`, and `daylib/workset_scheduler.py`.
- Current auth/session/Cognito startup behavior is in `daylib/routes/portal.py`, `daylib/workset_api.py` combined dependency, and `daylib/cli/server.py` preflight.
- Atlas local patterns for tenant/RBAC/service auth are available in `app/auth/rbac.py`, `app/auth/dependencies.py`, `app/api/routes/internal.py`.
- TapDB local conventions for explicit session ownership, audit attribution, EUID/soft-delete triggers, and namespace-aware config are available in `daylily_tapdb/connection.py`, `daylily_tapdb/cli/db_config.py`, `schema/tapdb_schema.sql`.

Unverified assumptions:
- Exact Atlas `lsmc_work_order` status API payload, endpoint path, and machine-auth requirements are not defined in Ursa repo and require contract confirmation.
- Exact delegated identity transport format (JWT vs signed headers) for Atlas-triggered Ursa jobs requires cross-team agreement.
- Production tenancy mapping source from Ursa `customer_id` to Atlas `tenant_id` requires explicit ownership decision.

## 2. PR sequence overview
1. Add migration contract, architecture ADR, and feature-flag scaffold for backend selection.
2. Introduce Postgres foundation and Ursa-owned workflow schema migration scaffolding.
3. Add Postgres repositories and transactional state-transition primitives behind flags.
4. Build DynamoDB semantic parity harness and concurrency/idempotency comparison tests.
5. Align Ursa auth/RBAC/tenant dependencies to Atlas semantics with bridge mapping.
6. Rewrite worker/scheduler/monitor claim/state transitions to transactional Postgres primitives.
7. Add transactional outbox and Atlas `lsmc_work_order` status sync worker with retries/DLQ.
8. Add Atlas-triggered direct analysis internal endpoint with machine auth and delegated identity.
9. Deliver backfill/cutover/rollback tooling, rehearsals, and Dynamo de-scope toggles.

## 3. Detailed PRs
### PR 1 - Migration contract and feature flags
Objective:
- Establish migration contract, acceptance criteria, and runtime feature flags without behavior change.

What changes in this PR:
- Add migration ADR docs and backend-selection feature flags in config plumbing.
- Add no-op interface boundaries for workflow repository selection.

Why this PR exists:
- Creates a controlled rollout surface and reviewable contract before schema/runtime refactors.

What risk it isolates:
- Prevents unbounded hidden behavior changes while introducing migration scaffolding.

Exact files/directories to add or modify:
- Modify: `daylib/config.py`, `daylib/workset_api.py`, `daylib/workset_api_cli.py`
- Add: `docs/adr/0001-ursa-postgres-migration.md`
- Add: `daylib/services/workflow_backend.py`
- Add: `tests/test_migration_flags.py`

Schema or migration changes:
- None.

Feature flags:
- Add `URSA_WORKFLOW_BACKEND` (`dynamo` default, `postgres` disabled by default).
- Add `URSA_PARITY_MODE` (`off` default).

Tests:
- Add/modify: `tests/test_migration_flags.py`, `tests/test_optional_auth.py` (ensure no auth regression from config changes).

Rollout:
- Deploy with defaults (`dynamo`, parity off); no runtime path change.

Rollback:
- Revert PR or unset new env vars; no data-plane impact.

Dependencies:
- None.

Acceptance criteria:
- App boots with unchanged behavior when flags are default.
- Flags appear in settings and are unit-tested.

### PR 2 - Postgres foundation and schema scaffolding
Objective:
- Introduce Ursa Postgres connection/session foundation and initial migration framework for Ursa-owned workflow tables.

What changes in this PR:
- Add DB engine/session modules.
- Add migration framework bootstrap and first schema revision for core tables.

Why this PR exists:
- Establishes durable relational storage substrate before repository adoption.

What risk it isolates:
- Isolates DB/bootstrap risk from application behavior changes.

Exact files/directories to add or modify:
- Add: `daylib/db/engine.py`
- Add: `daylib/db/session.py`
- Add: `daylib/db/models/workflow.py`
- Add: `daylib/migrations/README.md`
- Add: `daylib/migrations/env.py`
- Add: `daylib/migrations/versions/0001_initial_workflow_schema.py`
- Modify: `pyproject.toml` (migration/runtime deps if needed)
- Add: `tests/persistence/test_db_bootstrap.py`

Schema or migration changes:
- Introduce tables: `analysis_request`, `workset`, `workset_current`, `execution_attempt`, `workset_event`, `idempotency_key`, `outbox_event`, `artifact_output`, `actor_context`.
- Introduce indexes/uniques for tenant/state/claim/idempotency paths.

Feature flags:
- No functional toggle change; still `URSA_WORKFLOW_BACKEND=dynamo` by default.

Tests:
- Add: `tests/persistence/test_db_bootstrap.py`.
- Add: `tests/persistence/test_schema_constraints_smoke.py`.

Rollout:
- Run migrations in non-prod first; production migration can be applied while backend remains Dynamo.

Rollback:
- Roll back migration revision if not yet used by runtime; backend flag remains Dynamo.

Dependencies:
- PR 1.

Acceptance criteria:
- Migration can apply/rollback cleanly in CI.
- No production code path switched to Postgres yet.

### PR 3 - Postgres repositories and transactional primitives
Objective:
- Implement Postgres repository layer and transactional state/lease APIs equivalent to current Dynamo semantics.

What changes in this PR:
- Add repository interfaces/implementations.
- Add service layer that performs claim + transition + event append atomically.
- Add backend switch wiring for selected API paths behind flag.

Why this PR exists:
- Introduces transactional semantics needed before worker migration.

What risk it isolates:
- Isolates core data logic correctness from scheduler/worker complexity.

Exact files/directories to add or modify:
- Add: `daylib/db/repositories/workset_repository.py`
- Add: `daylib/db/repositories/analysis_request_repository.py`
- Add: `daylib/db/repositories/outbox_repository.py`
- Add: `daylib/services/workflow_state_service.py`
- Modify: `daylib/workset_api.py`, `daylib/routes/worksets.py`, `daylib/routes/customer_worksets.py`
- Add: `tests/persistence/test_postgres_workset_repository.py`
- Add: `tests/persistence/test_postgres_state_transitions.py`

Schema or migration changes:
- Add/adjust constraints for atomic claim and event sequence (if discovered missing in PR 2).

Feature flags:
- Use `URSA_WORKFLOW_BACKEND=postgres` for opted-in paths in test/staging only.

Tests:
- Add repository unit + integration tests for create, lock claim, release, state transition, retry reset.

Rollout:
- Enable postgres backend for low-risk read/list endpoints in staging first.

Rollback:
- Set backend flag back to `dynamo`; repositories stay dormant.

Dependencies:
- PR 2.

Acceptance criteria:
- Postgres repository parity for core CRUD/lock/state operations proven in tests.
- Flag-off path remains Dynamo.

### PR 4 - Dynamo semantic parity harness and comparison tests
Objective:
- Provide an explicit semantic comparison harness proving Postgres behavior matches required Dynamo semantics.

What changes in this PR:
- Add dual-backend test harness and scenario matrix (conditional create, lock contention, stale takeover, retry/archive/delete).
- Add CI job target for parity suite.

Why this PR exists:
- Makes migration correctness measurable and reviewable.

What risk it isolates:
- Detects silent semantic drift before worker cutover.

Exact files/directories to add or modify:
- Add: `tests/parity/test_dynamo_vs_postgres_semantics.py`
- Add: `tests/parity/test_lock_contention.py`
- Add: `tests/parity/test_retry_archive_delete_semantics.py`
- Add: `scripts/verify_dynamo_postgres_parity.py`
- Modify: `tests/conftest.py`

Schema or migration changes:
- None expected; only test fixtures.

Feature flags:
- Introduce `URSA_PARITY_MODE=compare` for harness execution.

Tests:
- New parity/concurrency suite above.

Rollout:
- Run parity suite in CI and staging rehearsal before any production backend flip.

Rollback:
- Disable parity mode; no runtime impact.

Dependencies:
- PR 3.

Acceptance criteria:
- Parity suite passes for required Dynamo semantics.
- Failures produce deterministic diff output tied to scenario IDs.

### PR 5 - Auth/RBAC alignment with Atlas
Objective:
- Align Ursa auth/tenant/RBAC checks to Atlas role and tenant semantics while preserving portal behavior.

What changes in this PR:
- Add Ursa auth dependency layer with role/permission checks patterned after Atlas.
- Add tenant mapping bridge from `customer_id` to `tenant_id`.
- Update route guards to use new dependency checks.

Why this PR exists:
- Required for Atlas-consistent authorization and internal cross-tenant role support.

What risk it isolates:
- Prevents privilege escalation and tenant leakage during migration.

Exact files/directories to add or modify:
- Add: `daylib/auth/rbac.py`
- Add: `daylib/auth/dependencies.py`
- Add: `daylib/services/tenant_mapping.py`
- Modify: `daylib/routes/dependencies.py`
- Modify: `daylib/routes/worksets.py`
- Modify: `daylib/routes/customer_worksets.py`
- Modify: `daylib/routes/portal.py`
- Add: `tests/auth/test_rbac_permissions.py`
- Add: `tests/auth/test_tenant_scope.py`
- Modify: `tests/test_portal_authz_matrix.py`

Schema or migration changes:
- Add tenant-mapping table if needed (`tenant_bridge_map`) via migration.

Feature flags:
- Add `URSA_ATLAS_RBAC_MODE` (`legacy` default, `atlas_aligned` opt-in).

Tests:
- Add RBAC/tenant scope tests including internal cross-tenant role cases.

Rollout:
- Enable in staging with Atlas-like test identities first.

Rollback:
- Flip `URSA_ATLAS_RBAC_MODE=legacy`.

Dependencies:
- PR 1 (flag scaffold), PR 3 (service integration points).

Acceptance criteria:
- Non-admin cross-tenant access blocked.
- Internal/admin cross-tenant flows require explicit permissions.

### PR 6 - Worker/scheduler/monitor transactional rewrite
Objective:
- Migrate lifecycle mutation loops to Postgres transactional claim/state primitives.

What changes in this PR:
- Replace separate claim/update/release call chains with one transactional workflow service call.
- Update worker and monitor loops to use DB-authoritative state transitions.

Why this PR exists:
- Removes race-prone multi-call state mutations.

What risk it isolates:
- Isolates concurrency correctness in background processing paths.

Exact files/directories to add or modify:
- Modify: `daylib/workset_worker.py`
- Modify: `daylib/workset_concurrent_processor.py`
- Modify: `daylib/workset_monitor.py`
- Modify: `daylib/workset_scheduler.py`
- Modify: `daylib/workset_api.py`
- Add: `tests/concurrency/test_worker_claims.py`
- Add: `tests/concurrency/test_stale_lease_takeover.py`
- Modify: `tests/test_workset_concurrent_processor.py`

Schema or migration changes:
- Add/adjust indexes on claim path (`state`, `lock_expires_at`, `priority`) if required by perf tests.

Feature flags:
- Add `URSA_WORKER_BACKEND` (`dynamo` default, `postgres` opt-in).

Tests:
- Concurrency tests for competing workers and stale leases.

Rollout:
- Enable postgres worker backend in staging with synthetic load; then canary in production.

Rollback:
- Flip worker backend flag back to `dynamo`.

Dependencies:
- PR 3, PR 4.

Acceptance criteria:
- No duplicate claims under concurrency tests.
- Worker failures preserve consistent attempt/event history.

### PR 7 - Outbox and Atlas status sync for lsmc_work_order
Objective:
- Implement transactional outbox and reliable Atlas status reporting for `lsmc_work_order`.

What changes in this PR:
- Add outbox event emission on key state transitions.
- Add outbox dispatcher worker and Atlas client integration with retry/backoff/DLQ.

Why this PR exists:
- Decouples external side effects from core state transaction and ensures reliable status propagation.

What risk it isolates:
- Prevents duplicate/missed Atlas status updates under retries/failures.

Exact files/directories to add or modify:
- Add: `daylib/services/outbox.py`
- Add: `daylib/services/atlas_status_sync.py`
- Add: `daylib/workers/outbox_dispatcher.py`
- Add: `daylib/clients/atlas_client.py`
- Modify: `daylib/services/workflow_state_service.py`
- Modify: `daylib/config.py`
- Add: `tests/integration/test_atlas_status_sync.py`
- Add: `tests/unit/test_outbox_retry_dedup.py`

Schema or migration changes:
- Add DLQ/status columns and dedupe indexes to `outbox_event`.

Feature flags:
- Add `URSA_OUTBOX_ENABLED` (off default), `URSA_ATLAS_STATUS_SYNC_ENABLED` (off default).

Tests:
- Contract-style tests for payload mapping and idempotent retries.

Rollout:
- Enable outbox emission first (dispatcher off), then enable dispatcher in staging, then production.

Rollback:
- Disable dispatcher/status-sync flags; outbox rows remain for replay.

Dependencies:
- PR 3, PR 4, PR 5.

Acceptance criteria:
- Atlas status events are delivered once-per-dedupe-key or land in DLQ with audit trail.

### PR 8 - Atlas-triggered direct analysis endpoint and delegated identity
Objective:
- Add internal API surface for Atlas-triggered analysis jobs that bypass Bloom, with delegated identity handling.

What changes in this PR:
- New internal route(s) for direct analysis request creation.
- Machine-auth validation and delegated actor context ingestion.
- Idempotent request handling tied to `analysis_request` and `workset` creation.

Why this PR exists:
- Satisfies required Atlas->Ursa direct trigger capability under secure auth semantics.

What risk it isolates:
- Isolates contract/auth risks before full cutover.

Exact files/directories to add or modify:
- Add: `daylib/routes/internal_atlas.py`
- Add: `daylib/auth/service_auth.py`
- Add: `daylib/schemas/internal_atlas.py`
- Modify: `daylib/workset_api.py`
- Modify: `daylib/config.py`
- Add: `tests/contract/test_internal_atlas_trigger.py`
- Add: `tests/auth/test_service_auth_delegation.py`

Schema or migration changes:
- Add any missing columns for delegated actor attribution in `actor_context` / `analysis_request`.

Feature flags:
- Add `URSA_ATLAS_INTERNAL_API_ENABLED` (off default).

Tests:
- Contract tests for auth, scopes, tenant binding, idempotent replays.

Rollout:
- Enable endpoint in staging with Atlas test client only, then limited production allowlist.

Rollback:
- Disable `URSA_ATLAS_INTERNAL_API_ENABLED`.

Dependencies:
- PR 5, PR 7.

Acceptance criteria:
- Valid machine + delegated requests create exactly one analysis request/workset per idempotency key.
- Unauthorized or mismatched tenant requests are rejected.

### PR 9 - Backfill, cutover, rollback tooling and Dynamo de-scope
Objective:
- Deliver production-grade backfill, cutover, and rollback playbooks/tooling with rehearsals.

What changes in this PR:
- Add backfill tool, reconciliation/verifier, cutover command runner, rollback command runner.
- Add operational runbooks and acceptance gates.
- Narrow Dynamo usage behind explicit kill-switches after cutover success.

Why this PR exists:
- Makes migration executable and reversible with controlled operational risk.

What risk it isolates:
- Isolates cutover execution risk and split-brain risk.

Exact files/directories to add or modify:
- Add: `scripts/backfill_dynamo_to_postgres.py`
- Add: `scripts/reconcile_postgres_vs_dynamo.py`
- Add: `scripts/cutover_rehearsal.py`
- Add: `scripts/rollback_to_dynamo.py`
- Add: `docs/runbooks/ursa_postgres_cutover.md`
- Add: `docs/runbooks/ursa_postgres_rollback.md`
- Modify: `daylib/config.py`
- Modify: `daylib/workset_api.py`
- Modify: `daylib/workset_monitor.py`
- Add: `tests/migration/test_backfill_parity.py`
- Add: `tests/migration/test_cutover_rehearsal.py`
- Add: `tests/migration/test_rollback_rehearsal.py`

Schema or migration changes:
- Optional migration adjustments discovered during rehearsal (indexes/constraints only).

Feature flags:
- Add `URSA_DYNAMO_WRITE_ENABLED` (on default pre-cutover), `URSA_DYNAMO_READ_FALLBACK_ENABLED` (off default pre-cutover, on only during bounded rollback window).

Tests:
- Backfill determinism, in-flight workset cutover, rollback no-split-brain checks.

Rollout:
- Execute rehearsals in staging, then production cutover with approval gates and bounded rollback window.

Rollback:
- Use rollback script + backend flags to restore Dynamo as sole writer; replay deferred outbox events after stabilization.

Dependencies:
- PR 1 through PR 8.

Acceptance criteria:
- Rehearsal passes defined SLO gates.
- Production cutover runbook can complete with verified parity and reversible fallback.

## 4. Cross-PR test strategy
Overall integration coverage:
- Maintain existing Ursa suites while adding Postgres parity/concurrency suites.
- Run both backend modes in CI matrix where practical (`dynamo`, `postgres`, `compare`).

Atlas-Ursa contract testing:
- Add explicit contract tests for internal trigger endpoint auth, tenant scope, role constraints, and status callback payloads.

Backfill verification:
- Automated record-count and keyset parity per tenant/state.
- Sampled deep-compare for metadata/history consistency.

Cutover rehearsal:
- Staging rehearsal includes in-flight worksets, worker restarts, and outbox backlog replay.

Rollback rehearsal:
- Validate reversion to Dynamo as sole writer with no concurrent Postgres writes and no duplicate external side effects.

## 5. Cutover sequence
1. Apply schema migrations in production with backend flags still set to Dynamo.
2. Run full backfill from Dynamo to Postgres and generate parity report.
3. Enable `URSA_PARITY_MODE=compare` and execute smoke traffic validation.
4. Pause/safeguard new workset submissions briefly or queue them behind idempotent intake.
5. Enable `URSA_WORKFLOW_BACKEND=postgres` for API writes/reads.
6. Enable `URSA_WORKER_BACKEND=postgres` for workers/monitor.
7. Enable outbox dispatcher and Atlas status sync.
8. Verify gates: state transition integrity, outbox lag, Atlas status success rate, tenant isolation checks.
9. Keep bounded rollback window with `URSA_DYNAMO_READ_FALLBACK_ENABLED` only.
10. Decommission Dynamo write paths (`URSA_DYNAMO_WRITE_ENABLED=off`) and close rollback window after acceptance period.

## 6. Highest-risk areas
- Transactional parity for lock/lease/state transitions under high contention.
- Tenant/RBAC bridge correctness while both `customer_id` and `tenant_id` exist.
- Atlas contract ambiguity for `lsmc_work_order` status payload/auth until finalized.
- In-flight workset cutover without split-brain ownership between stores.
- External side-effect idempotency (Atlas sync + optional notifications) during retries and rollback.
