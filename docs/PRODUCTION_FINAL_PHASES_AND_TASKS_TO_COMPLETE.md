# Production Final Phases & Tasks To Complete (daylily-ursa)

Source: distilled from `docs/PRODUCTION_READINESS_REVIEW_2026-01-24.md`.


## Creds to use for testing if needed
AWS_PROFILE=lsmc
email_address=johnm@lsmc.life
ui password:C4un3y!!C4un3y!!
customer_id=lsmc-76fd9f2a
workset-manifest=augment-testing-manifest
config ~/.ursa/ursa-config.yaml
prefer clusters in us-west-2

## Assumptions
- "TODO: add to future development plan" items are **excluded** from this list (per instructions).
- This checklist targets pre-production hardening for a real customer-facing deployment.

## Trade-offs
- **Gain:** a single actionable pre-prod punch list.
- **Lose:** deep design discussion; this is execution-focused.

## DO THIS
- complete Phase 0–4, write tests for each phase as appropriate.
- add full audit logging.
- add threat modeling, chaos/failover drills to TODO: future development plan


---

## Phase 0 — Release blockers (must-fix)

### 0.1 Packaging/runtime correctness
- [ ] Fix `ursa server start` so it does **not** depend on repo-relative `bin/daylily-workset-api`.
  - Acceptance: after `pip install daylily-ursa`, `ursa server start` works and starts the API.

### 0.2 CI gates
- [ ] Add CI to run at minimum: unit tests, lint, typecheck.
  - Acceptance: CI runs on PRs and blocks merges on failure.

### 0.3 AuthZ / access control (portal)
- [ ] Ensure **worksets are only visible to the creating user/owner**.
- [ ] Ensure **monitor page** is only accessible to **admin** users.
- [ ] Ensure **cluster page cost details** are hidden for non-admin users (only show budgets they have visibility to).
  - Acceptance: verified via role-based UI + backend enforcement (not just hidden buttons). tests written and pass

### 0.4 File upload regression
- [ ] Fix `/portal/files/upload` returning **403 Forbidden** when uploading to a provisioned/linked bucket.
  - Add tests covering expected behavior and common IAM/policy failure modes.

### 0.5 Account / security UX
- [x] Enforce password minimum length: **8 characters**.
- [x] Fix "Generate API token" not working.

---

## Phase 1 — Cross-region + S3 correctness
- [x] Make all portal S3 operations consistently region-aware (no plain boto3 client in presign/download/upload paths).
  - Acceptance: BYOB buckets in a non-default region work end-to-end (browse, presign download, upload where allowed).

---

## Phase 2 — UI/UX fixes (pre-prod)

### 2.1 Usage page
- [ ] Add "Edit" button to Billing Info card.
- [ ] Fix mismatch between cost numbers in the card vs the pie chart (must derive from the same source).
- [ ] Fix "Export report" button not working.
- [ ] Present transfer cost as 3 explicit options: within-region, cross-region, internet egress.

### 2.2 Register page
- [ ] Change title branding to "Ursa" and fix the header/logo presentation.
- [ ] Confirm Resource Limits are captured in DB and visible on account page.
- [ ] If AWS Account ID is entered, make it visible/usable in the UI; if not possible, document required fixes to discover S3 resources.
- [ ] Clarify purpose of "Cost center" vs "Budget" (UI copy + backend mapping).
- [ ] Add basic Terms of Service + Privacy documents tailored for LSMC.

### 2.3 Admin tooling (portal)
- [ ] Admin tab features:
  - [ ] Add user
  - [ ] Set/remove admin privileges
  - [ ] Change user password

### 2.4 Files / buckets page
- [ ] Move "Link bucket" buttons card above linked buckets list; match width/styling.
- [ ] Remove "Discover files" card; replace with a button that redirects to the auto-discover page.
- [ ] "Edit bucket" should be a per-bucket action that opens a dialog to edit properties.

### 2.5 Biospecimen / subjects
- [ ] Rename action to "Add new biospecimen".
- [ ] Add dedicated subjects page: `/portal/subjects`.
- [ ] Move "Add New" card above list and match width.

### 2.6 Clusters page
- [ ] Add admin-only "Delete cluster" button with 2 acknowledgement prompts.
  - Must execute: `AWS_PROFILE=<profile> pcluster delete-cluster -n <cluster-name> --region <region>`

### 2.7 Worksets list page
- [ ] Reduce table cell padding by ~40%.
- [ ] Fix footer positioning: only at bottom of page, not forced visible.
- [ ] Fix user menu hover text overflow (upper-right user link tooltip/text).
- [ ] Fix "Samples" column in Recent Worksets always showing 0.
- [ ] Fix account details "Rename" button: rename to "Edit" and allow editing fields.

### 2.8 File registration
- [ ] Fix bulk import in `/portal/files/register` and add tests.

---

## Phase 3 — Operational behavior (monitor / worksets)
- [ ] Validate monitor restart semantics around in-flight SSH commands:
  - if monitor dies mid-SSH, ensure outcomes are reconciled on restart.
- [ ] Workset retry semantics:
  - retries should create a **new** workset (same name prefix + new datetime suffix) cloned from the original, leaving the original unchanged.
- [ ] Add admin-only tooling to filter/grep monitor logs to show **remote SSH + AWS CLI commands** executed for a specified workset, so admins can replay manually.

---

## Phase 4 — Documentation quick wins
- [ ] Fix port inconsistencies in `docs/QUICKSTART_WORKSET_MONITOR.md`.
- [ ] Ensure docs reference packaged entry points (console scripts), not repo-relative scripts.

