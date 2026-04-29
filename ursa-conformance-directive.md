# Ursa Conformance Audit

> Target repo: `daylily/daylily-ursa`
> Current checkout reviewed: `codex/feature-from-main`
> Purpose: current status against the Atlas/Bloom-aligned Ursa service direction

## Guiding Rule

Ursa should remain a peer service to Atlas and Bloom. Auth, tenant isolation, database runtime, config, and CLI behavior should follow the shared platform patterns unless Ursa has an explicit service-owned reason to differ.

## Current Status

| Area | Status | Current State |
| --- | --- | --- |
| Activation | Done | `source ./activate <deploy-name>` creates/activates one deployment-scoped conda env and performs one editable install only on first create. |
| Packaging boundary | Done | `environment.yaml` holds system/runtime packages; Python dependencies live in `pyproject.toml`; no optional repo install groups are declared. |
| CLI root | Done | `daylib_ursa.cli` builds a `cli-core-yo==2.1.1` `CliSpec` with XDG config, runtime guards, prereqs, JSON policy, and explicit plugins. |
| CLI callbacks | Partially done | Command modules register through `cli-core-yo`, but several callbacks still use `typer` annotations and exits internally. |
| TapDB runtime | Done | Ursa uses `daylily-tapdb==6.0.8`, `TapdbClientBundle`, `TAPDBConnection`, `TemplateManager`, `InstanceFactory`, explicit config paths, and namespace-aware runtime env derivation. |
| Zebra dependency | Done | Ursa declares `zebra_day==6.0.1` as a direct Python dependency in the single package install set. |
| TapDB backend | Done | `TapDBBackend` writes Ursa resource templates through the TapDB 6.x runtime surface and enforces tenant-aware instance creation/listing patterns. |
| TapDB admin mount | Done | `/admin/tapdb` mounts the TapDB admin app inside Ursa when enabled, with explicit TapDB context forwarding and `X-API-Key` gating. |
| Auth package shape | Done | Auth lives under `daylib_ursa/auth/` with `CurrentUser`, web/API auth dependencies, JWT/session handling, user tokens, and user directory integration. |
| RBAC | Done | `Role`, `Permission`, `ROLE_PERMISSIONS`, and helper functions live in `daylib_ursa/auth/rbac.py`. |
| Tenant isolation | Partially done | Most API routes check tenant ownership and admin access at route/service boundaries; continued review should focus on new routes and TapDB queries as they are added. |
| Manifest contract | Done | Ursa generates `metadata.analysis_samples_manifest` from editor rows or S3 references using the installed `daylily-ephemeral-cluster==2.1.11` template. |
| Staging jobs | Done | `/api/v1/staging-jobs` defines, runs, reads, and returns logs for `daylily-ec samples stage` jobs. |
| Analysis jobs | Done | `/api/v1/analysis-jobs` can create jobs from a manifest and either stage directly from a reference bucket or reuse a completed `staging_job_euid`. |
| Atlas return | Done | Result return requires approved review state, Dewey artifact EUIDs, opaque Atlas/Bloom EUIDs, idempotency, and persisted Atlas response metadata. |

## Remaining Gaps

- Finish removing residual `typer` usage from CLI subcommand callback signatures and exits if the target remains zero direct `typer` imports under `daylib_ursa/cli/`.
- Continue auditing tenant isolation when adding routes: dependency layer, service/resource layer, and TapDB query layer should all agree.
- Keep conformance tests close to the behavior that matters: activation/package boundaries, CLI runtime guards, TapDB runtime selection, auth/RBAC, generated manifest rules, staging jobs, and analysis-job reuse of completed staging.

## Current Verification Commands

```bash
pytest tests/test_activation_metadata.py tests/test_cli_registry_v2.py tests/test_console_scripts.py -q
pytest tests/test_auth_dependencies.py tests/test_tapdb_backend.py tests/test_tapdb_mount.py tests/test_staging_jobs.py tests/test_worksets_api.py -q
```

## Non-Goals

- Do not broaden behavior outside current explicit contracts unless a current task requires it.
- Do not move Python dependencies out of `pyproject.toml`.
- Do not broaden `activate` beyond environment creation, activation, and the first editable install.
- Do not bypass `ursa`, `tapdb`, `daycog`, or `daylily-ec` with raw tools when the intended CLI path is the surface under test.
