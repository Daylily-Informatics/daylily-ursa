# Ursa Conformance Directive — Align to Atlas/Bloom Standards

> **Target repo:** `daylily/daylily-ursa` (`daylib_ursa/`)
> **Reference repo:** `lsmc/lsmc-atlas` (`app/`)
> **Secondary reference:** `daylily/bloom` (`bloom_lims/`)
> **Scope:** Database layer, auth, tenant isolation, RBAC, CLI framework, config

---

## 0. Guiding Principle

Ursa is a **peer service** to Atlas and Bloom. It MUST NOT invent its own patterns for auth, database, users, tenants, permissions, or CLI. When in doubt, copy Atlas verbatim and rename.

**The rule is simple:** if Atlas does X one way, Ursa does X the same way. No exceptions. No "Ursa-specific" alternatives.

---

## 1. DATABASE: TapDB 3.x Only — Kill the 2.x Surface

### Current Problem
`daylib_ursa/tapdb_graph/backend.py` imports `UrsaTapdbRepository` from `daylily_tapdb` 2.x. It has a version guard that **rejects** TapDB 3.x (`_MAXIMUM_TAPDB_MAJOR = 3`). The remote deployment has a manual 350-line patch to work around this. This is the single biggest source of drift.

### Required State

**File: `daylib_ursa/tapdb_graph/backend.py`**

1. **Delete** the `UrsaTapdbRepository` import and its fallback stub entirely.
2. **Delete** `_MINIMUM_TAPDB_VERSION`, `_MAXIMUM_TAPDB_MAJOR`, `_validate_tapdb_version()`.
3. **Model after Atlas's `app/integrations/tapdb_runtime.py`:**
   - Pin `TAPDB_REQUIRED_VERSION = "3.0.2"` (or current 3.x release).
   - Use `ensure_tapdb_version()` with the same semver + local-override logic Atlas uses.
   - Import from `daylily_tapdb` 3.x surface: `TAPDBConnection`, `TemplateManager`, `InstanceFactory`.
   - Implement a `TapdbClientBundle` dataclass holding `connection`, `template_manager`, `instance_factory`.
   - Provide `get_tapdb_bundle() -> TapdbClientBundle` as the single entry point.

4. **Namespace isolation:** Use `TAPDB_DATABASE_NAME = "daylily-ursa"`, `TAPDB_CLIENT_ID = "ursa"`.

5. **Connection string derivation:** Copy Atlas's `_build_sqlalchemy_url()`, `_get_tapdb_db_config_for_env()`, `_resolve_runtime_env()`, `export_database_url_for_target()` — rename Atlas→Ursa constants.

6. **No other database.** No SQLite, no raw Postgres, no in-memory stores. TapDB is the only persistence layer.

### Verbatim Pattern from Atlas

```python
# app/integrations/tapdb_runtime.py — Atlas reference
TAPDB_REQUIRED_VERSION = "3.0.2"
DEFAULT_TAPDB_CLIENT_ID = "atlas"        # Ursa: "ursa"
DEFAULT_TAPDB_DATABASE_NAME = "lsmc-atlas"  # Ursa: "daylily-ursa"

def ensure_tapdb_version(required_version=TAPDB_REQUIRED_VERSION) -> str:
    # exact pin for wheel, >= for local editable override
    ...
```

---

## 2. AUTH: Replace ActorContext with CurrentUser

### Current Problem
`daylib_ursa/auth.py` defines `ActorContext` — a bespoke auth context with non-standard role names (`admin_markers = {"admin", "atlas_admin", "ursa_admin", ...}`), external HTTP calls to Atlas for identity resolution, and a custom token system (`urs_` prefix). None of this matches Atlas.

### Required State

**File: `daylib_ursa/auth/dependencies.py`** (new — replace `daylib_ursa/auth.py`)

1. **Copy Atlas's `CurrentUser` dataclass verbatim** from `app/auth/dependencies.py` (lines 25-88).
   - Fields: `sub`, `email`, `name`, `tenant_id` (UUID), `roles` (list[str]).
   - Properties: `id`, `user_id`, `sub_uuid`, `has_role()`, `has_permission()`, `is_internal`, `is_admin`, `is_org_admin`, `can_write`.

2. **Copy Atlas's `get_current_user()` function** — session-based + Bearer token auth.
   - Web routes: `get_current_user_web()` → raises `WebAuthRedirect`.
   - API routes: `get_current_user()` → raises HTTP 401.

3. **Copy Atlas's `get_current_tenant()` dependency:**
   ```python
   async def get_current_tenant(
       current_user: Annotated[CurrentUser, Depends(get_current_user)],
   ) -> uuid.UUID:
       return current_user.tenant_id
   ```

4. **Copy Atlas's `require_role()` and `require_permission()` dependency factories.**

5. **Copy Atlas's Annotated shortcuts:**
   ```python
   RequireAuth = Annotated[CurrentUser, Depends(get_current_user)]
   RequireInternal = Annotated[CurrentUser, Depends(require_role(Role.INTERNAL_USER, Role.ADMIN))]
   RequireAdmin = Annotated[CurrentUser, Depends(require_role(Role.ADMIN))]
   ```

6. **Delete** `ActorContext`, `AtlasIdentityClient`, `_coerce_actor`, `_normalize_roles`.
7. **Delete** `UserTokenService` or migrate it to use `CurrentUser` instead of `ActorContext`.

### What Ursa Can Keep
- Its own token prefix (`urs_`) for Ursa-specific API tokens — but the token validation must return a `CurrentUser`, not an `ActorContext`.
- The `UserTokenService` class can remain if refactored to accept `CurrentUser` where it currently takes `ActorContext`.

---

## 3. RBAC: Use Atlas's Role and Permission Enums

### Current Problem
Ursa has ad-hoc role checking: `admin_markers = {"admin", "atlas_admin", "ursa_admin", "tenant_admin", "org_admin"}`. No `Permission` enum. No `ROLE_PERMISSIONS` mapping.

### Required State

**File: `daylib_ursa/auth/rbac.py`** (new)

1. **Import or copy Atlas's `Role` enum** from `app/auth/rbac.py`:
   ```python
   class Role(StrEnum):
       READ_ONLY = "READ_ONLY"
       READ_WRITE = "READ_WRITE"
       EXTERNAL_USER = "EXTERNAL_USER"
       EXTERNAL_USER_ADMIN = "EXTERNAL_USER_ADMIN"
       INTERNAL_USER = "INTERNAL_USER"
       ADMIN = "ADMIN"
   ```

2. **Define Ursa-specific permissions** following Atlas's `Permission(StrEnum)` pattern:
   ```python
   class Permission(StrEnum):
       WORKSET_CREATE = "workset:create"
       WORKSET_READ = "workset:read"
       WORKSET_UPDATE = "workset:update"
       ANALYSIS_SUBMIT = "analysis:submit"
       ANALYSIS_READ = "analysis:read"
       CLUSTER_READ = "cluster:read"
       CLUSTER_MANAGE = "cluster:manage"
       COST_READ = "cost:read"
       CROSS_TENANT_READ = "cross_tenant:read"
   ```

3. **Define `ROLE_PERMISSIONS` mapping** — same structure as Atlas's.

4. **Copy** `has_permission()`, `has_role()`, `is_internal()`, `is_admin()`, `is_read_only()`, `can_write()`, `can_access_tenant()` — verbatim from Atlas.

---

## 4. TENANT ISOLATION: Triple-Layer Enforcement

### Required Pattern (from Atlas)

Every data-touching operation must enforce tenant isolation at three layers:

1. **FastAPI dependency layer:**
   ```python
   @router.get("/worksets")
   async def list_worksets(
       tenant_id: Annotated[uuid.UUID, Depends(get_current_tenant)],
       ...
   ):
   ```

2. **Service constructor layer:**
   ```python
   class WorksetService:
       def __init__(self, backend: TapDBBackend, tenant_id: uuid.UUID):
           self.backend = backend
           self.tenant_id = tenant_id
   ```

3. **Query layer:**
   All TapDB queries must include `tenant_id` filtering. No query returns data across tenants unless the user has `CROSS_TENANT_READ` permission and the code explicitly opts in.

---

## 5. CLI: Migrate from Typer to cli-core-yo

### Current Problem
`daylib_ursa/cli/__init__.py` uses `typer.Typer()`. Atlas uses `cli-core-yo` with `CliSpec` + `create_app()`.

### Required State

**File: `daylib_ursa/cli/__init__.py`**

Replace the entire file with Atlas's pattern:

```python
from cli_core_yo.app import create_app, run
from cli_core_yo.spec import CliSpec, ConfigSpec, EnvSpec, PluginSpec, XdgSpec

spec = CliSpec(
    prog_name="ursa",
    app_display_name="Daylily Ursa",
    dist_name="daylily-ursa",
    root_help="Daylily Ursa — HPC analysis service CLI.",
    xdg=XdgSpec(app_dir_name="ursa"),
    config=ConfigSpec(
        primary_filename="ursa-config.yaml",
        template_resource=("daylib_ursa", "etc/ursa-config-template.yaml"),
        validator=_validate_ursa_config,
    ),
    env=EnvSpec(
        active_env_var="URSA_ACTIVE",
        project_root_env_var="URSA_PROJECT_ROOT",
        activate_script_name="ursa_activate",
        deactivate_script_name="ursa_deactivate",
    ),
    plugins=PluginSpec(
        explicit=[
            "daylib_ursa.cli.server.register",
            "daylib_ursa.cli.env.register",
            "daylib_ursa.cli.test.register",
            "daylib_ursa.cli.quality.register",
        ],
    ),
    info_hooks=[_ursa_info_hook],
)

app = create_app(spec)

def main() -> None:
    raise SystemExit(run(spec))
```

**Delete:** All `typer.Typer()`, `typer.Option()`, `typer.Context` usage.
**Migrate:** Each sub-command module (`server.py`, `env.py`, `test.py`, `quality.py`) to expose a `register(group)` function per cli-core-yo convention.

---

## 6. CONFIG: Align Settings with Atlas Pattern

### Current Problem
`daylib_ursa/config.py` has duplicate field definitions (two `ursa_cost_monitor_regions`, two `ursa_cost_monitor_interval_hours`, two `ursa_cost_monitor_config_path`). It also does not integrate with TapDB 3.x for database URL derivation.

### Required State

1. **Remove duplicate fields** — keep only one of each.
2. **Add `database` section** matching Atlas's pattern:
   ```python
   database_backend: str = Field(default="tapdb", description="Database backend: tapdb only")
   database_target: str = Field(default="local", description="Database target: local or aurora")
   ```
3. **Add `tapdb_*` fields** matching Atlas's env var pattern:
   ```python
   tapdb_client_id: str = Field(default="ursa")
   tapdb_database_name: str = Field(default="daylily-ursa")
   tapdb_env: Optional[str] = Field(default=None)
   ```
4. **DATABASE_URL derivation** must go through `export_database_url_for_target()` — same as Atlas.

---

## 7. FILE STRUCTURE: Match Atlas Layout

### Current Ursa layout:
```
daylib_ursa/
├── auth.py              ← monolithic, has ActorContext + tokens + Atlas HTTP client
├── config.py
├── cli/__init__.py      ← typer
├── tapdb_graph/
│   └── backend.py       ← 2.x imports
```

### Required layout:
```
daylib_ursa/
├── auth/
│   ├── __init__.py
│   ├── dependencies.py  ← CurrentUser, get_current_user, get_current_tenant
│   ├── rbac.py          ← Role, Permission, ROLE_PERMISSIONS
│   └── tokens.py        ← UserTokenService (refactored to use CurrentUser)
├── config.py            ← deduplicated, tapdb-aware
├── cli/
│   └── __init__.py      ← cli-core-yo
├── integrations/
│   └── tapdb_runtime.py ← TapDB 3.x connection, version guard, URL derivation
├── domain_access.py     ← copy from Atlas (approved domains, CORS helpers)
```

---

## 8. SANDBOX EUID PREFIX

### Required Implementation

1. Read `MERIDIAN_SANDBOX_PREFIX` env var (set from `/etc/dayhoff/sandbox-prefix`).
2. All EUID minting in Ursa must prepend the prefix when non-empty.
3. Format: `X:WST-9R` (prefix + colon + category + dash + body).
4. This requires a change in the TapDB minting layer — coordinate with `daylily-tapdb`.

---

## 9. EXECUTION CHECKLIST

| # | Task | Acceptance Criteria |
|---|------|-------------------|
| 1 | Replace `backend.py` TapDB 2.x → 3.x | No `UrsaTapdbRepository` import. Uses `TAPDBConnection`/`TemplateManager`/`InstanceFactory`. Version guard matches Atlas. |
| 2 | Replace `auth.py` → `auth/` package | `CurrentUser` dataclass identical to Atlas. No `ActorContext`. |
| 3 | Create `auth/rbac.py` | `Role` and `Permission` enums. `ROLE_PERMISSIONS` dict. Helper functions copied from Atlas. |
| 4 | Add triple-layer tenant isolation | Every router endpoint uses `Depends(get_current_tenant)`. Every service takes `tenant_id`. Every query filters by tenant. |
| 5 | Migrate CLI typer → cli-core-yo | `CliSpec` + `create_app()`. No typer imports anywhere. |
| 6 | Deduplicate config.py | No duplicate fields. TapDB config fields added. |
| 7 | Create `integrations/tapdb_runtime.py` | Matches Atlas's module structure. Ursa-specific defaults. |
| 8 | Wire sandbox EUID prefix | `MERIDIAN_SANDBOX_PREFIX` consumed at mint time. |
| 9 | All tests pass | Existing test suite updated. No regressions. |
| 10 | No drift markers | `grep -r "ActorContext\|UrsaTapdbRepository\|typer\.Typer\|typer\.Option" daylib_ursa/` returns zero hits. |

---

## 10. ANTI-PATTERNS — EXPLICITLY FORBIDDEN

| Forbidden | Why | Do Instead |
|-----------|-----|-----------|
| `ActorContext` | Ursa-specific auth context | Use `CurrentUser` from Atlas |
| `UrsaTapdbRepository` | TapDB 2.x surface | Use TapDB 3.x primitives |
| `typer.Typer()` | Non-standard CLI framework | Use `cli-core-yo` `CliSpec` |
| `httpx.get(atlas_url + "/auth/me")` | External HTTP for identity | Use local TapDB user lookup or session |
| Ad-hoc role strings (`"ursa_admin"`) | Non-standard roles | Use `Role` enum from `auth/rbac.py` |
| `from daylily_tapdb import UrsaTapdbRepository` | Removed in 3.x | Import 3.x primitives |
| Duplicate config fields | Causes Pydantic validation chaos | One field per setting |
| Queries without `tenant_id` filter | Tenant leak | Always filter by tenant |
