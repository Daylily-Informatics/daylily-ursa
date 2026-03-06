You are refactoring the repo “daylily-ursa” to be TapDB-native and EUID-first, matching Bloom patterns.

Primary goal
- TapDB `euid` must become the ONLY external identifier for all domain entities and relationships.
- Remove all application-defined “custom IDs” (for example: `workset_id`, `file_id`, `fileset_id`, `manifest_id`, any `*_id` that is minted by the app and stored in `json_addl` or exposed via API).
- Do not generate UUID-derived custom IDs anywhere (no `uuid4()`-minted identifiers for domain entities).

Identity rules
- TapDB internal `uuid` (DB PK) is allowed ONLY for internal bookkeeping if already required by TapDB client internals, but must never appear in API request/response models, URLs, or user-facing logs.
- TapDB `euid` is the canonical identifier everywhere outside TapDB. All API path params and references between entities must use EUIDs.
- Do not trust or round-trip `json_addl["euid"]`. If code currently sets/overrides EUID via json_addl, remove that. EUID must come from the TapDB instance field.

Non-goals
- Do not change TapDB itself.
- Do not add new abstraction layers that increase complexity. Prefer small helper functions and straightforward changes.

Breaking change policy
- This is intentionally a breaking API change: remove custom-id endpoints and request/response fields.
- Instead of backward-compatible endpoints, provide a one-time migration utility that maps historical custom IDs to EUIDs. Runtime API should be pure EUID.

Deliverables
1) Code refactor: all endpoints, models, services, and internal references use EUID only.
2) Migration utility script: produces a mapping from old custom IDs to EUIDs from existing TapDB data and optionally strips old keys from stored metadata (if TapDB supports patch/update).
3) Updated tests to pass. Update any fixtures, sample payloads, docs, and OpenAPI examples.
4) A short “Refactor summary” in the final response: list of changed files, commands run, and grep evidence that custom IDs are gone.

Work plan (execute in this order)

0. Baseline and safety
- Create a new branch.
- Run the existing test suite (or at least the fastest subset). Record results.
- Dump current OpenAPI schema or route list for comparison.

1. Inventory: find all custom-id usage
Use ripgrep to locate all custom ID fields and UUID minting. At minimum:
- rg -n "workset_id|fileset_id|file_id|manifest_id" .
- rg -n "json_addl" .
- rg -n "uuid4\\(" .
- rg -n "\\bUUID\\b|uuid\\b" .   (then manually review to avoid confusing TapDB internal uuid with app-defined ids)

Produce a short internal list of:
- Each custom ID field name
- Where it is created/minted
- Where it is accepted in API input
- Where it is emitted in API output
- Where it is persisted (especially in TapDB json_addl)

2. Define an explicit EUID type and helpers
- Add a small shared type alias for EUIDs (for example `Euid = str`) and optionally a validation helper (lightweight, no heavy regex unless TapDB defines a strict format).
- Add helper functions that make it hard to regress:
  - `require_euid(x) -> str` that raises a clear exception if missing/empty
  - `instance_to_public_dict(instance)` that always sets `"euid": instance.euid` and never reads `"euid"` from json_addl

3. Refactor Pydantic models and API schemas first
- For every request/response model that currently includes custom IDs:
  - Remove those fields.
  - Replace with `euid: str` (or `*_euid: str` where needed, but prefer a single field name `euid` for the primary resource and `related_euid` for references).
- Update FastAPI route signatures:
  - Replace `{workset_id}` with `{workset_euid}` or simply `{euid}` depending on your routing conventions.
  - Replace all body/query params that accept custom ids with EUID equivalents.
- Update OpenAPI docs/examples accordingly.
- Ensure response payloads include EUID and do not include custom IDs or TapDB internal uuid.

4. Refactor service layer and TapDB access
- Anywhere code looks up or filters by `json_addl["*_id"]`, replace with EUID-based lookup.
- Add or use existing TapDB client methods to fetch by EUID.
  - If TapDB client currently only supports uuid or custom filter, implement a thin wrapper that queries by EUID in the supported way.
- Ensure that any object creation returns the TapDB-issued EUID and that downstream code uses it.

5. Purge custom ID minting and persistence
- Delete all uuid4-based ID generation used for worksets/files/filesets/manifests or any domain resource identity.
- Remove any code that stores custom IDs into TapDB `json_addl` for identity/addressing.
- Remove any code that expects those keys to exist (parsing, setdefault, translations, etc).
- Critical: ensure `from_json_addl()` or any serializer does NOT allow json_addl to override instance.euid.
  - The EUID in outputs must always be `instance.euid`.

6. Migration utility (one-time tool, not runtime API)
Create `scripts/migrate_custom_ids_to_euids.py` (or similar) that:
- Scans TapDB for relevant instance types.
- For each instance, if metadata contains old custom ID keys, output a mapping row:
  - {resource_type, old_custom_id_key, old_custom_id_value, euid}
- Output formats:
  - JSONL and CSV are both useful. Prefer JSONL as primary.
- Optional cleanup step:
  - If TapDB supports patch/update of instance metadata, add a `--strip-custom-ids` flag that removes those keys from json_addl and writes back.
  - Must be idempotent and safe to re-run.
- Add `--dry-run` default and explicit `--apply` for writes.
- Add logging that never prints TapDB internal uuid.

7. Update tests and fixtures
- Update all tests, fixtures, and sample data to use EUIDs.
- Ensure no tests expect custom ids.
- Add at least one test that asserts:
  - Creating a resource yields an EUID
  - Fetching by EUID works
  - Response does not contain custom ids and does not contain internal uuid

8. Definition of done and grep-based guardrails
At the end, ensure:
- rg -n "workset_id|fileset_id|file_id|manifest_id" . returns only references inside the migration script (if at all). Prefer even the migration script to treat these as strings and not as stable API fields.
- rg -n "uuid4\\(" . returns no usages for identity. If uuid4 is used, it must be clearly non-identity (for example request correlation id) and justified in a comment.
- No FastAPI routes contain `{*_id}` for domain resources.
- No response models include custom IDs or TapDB internal uuid.
- `json_addl["euid"]` is never read to determine identity.

Output requirements (in your final response)
- A concise summary of what you changed.
- A file list of key modifications.
- The exact grep outputs (or a summarized count) proving custom IDs are removed.
- How to run:
  - tests
  - migration script (dry-run and apply)

Important constraints
- Keep the refactor straightforward. Do not preserve the old custom-id architecture.
- Prefer small commits or clearly separated steps in the diff.
- If you find ambiguous “id” fields, treat them as custom IDs unless they are clearly TapDB EUID or an external vendor identifier. When in doubt, search usage and decide based on whether it is minted locally and used for addressing.

Now implement the refactor.
Do not ask me questions unless you are completely blocked by missing TapDB client capabilities. If blocked, implement the missing TapDB client wrapper in this repo (without modifying TapDB server) as the simplest path forward.

IMPORTANT: Keep changes minimal and avoid new architecture. Match Bloom’s identity patterns.

AND:start by printing the inventory grep results before editing files, so you can sanity-check scope.