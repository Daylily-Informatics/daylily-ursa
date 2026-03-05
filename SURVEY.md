# Daylily Ecosystem — Architectural Survey & Refactoring Report

**Date**: 2026-02-28  
**Scope**: daylily-ursa, daylily-ephemeral-cluster, daylily-omics-analysis, daylily-omics-references, daylily-cognito  
**Conducted by**: Forge (Augment Agent) with Major (project owner)

---

## Executive Summary

A full architectural review of the Daylily genomics ecosystem was conducted, covering 5 repositories. The review produced a detailed assessment of strengths, risks, and improvement opportunities — then immediately executed on the highest-priority refactoring items. **6 PRs were merged** in a single session, addressing code quality, test reliability, API structure, and operational tooling.

### Headline Results

| Metric | Before | After |
|--------|--------|-------|
| `workset_api.py` (monolith) | 3,972 lines | 810 lines (80% reduction) |
| Route modules | 4 files | 14 files |
| Test suite | 18 failures + 7 collection errors | **785 passed, 0 failed** |
| API versioning | None | `/v1/` prefix on all API routes |
| WorksetState enum | 15 states (5 unused) | 10 states (all active) |
| Cross-repo version tracking | None | `config/ecosystem-versions.json` + CLI |

---

## PRs Merged

| PR | Title | Impact |
|----|-------|--------|
| [#6](https://github.com/Daylily-Informatics/daylily-ursa/pull/6) | Trim WorksetState enum to 10 actively-used states | Removed 5 unused states (`LOCKED`, `QUEUED`, `PAUSED`, `PENDING_REVIEW`, `BILLING_HOLD`) |
| [#7](https://github.com/Daylily-Informatics/daylily-ursa/pull/7) | Cleanup temp artifacts + update workset state diagram | Updated `WORKSET_STATE_DIAGRAM.md` to all 10 states; cleaned `.gitignore` |
| [#8](https://github.com/Daylily-Informatics/daylily-ursa/pull/8) | Extract customer-workset/dashboard/billing routes + version matrix | `workset_api.py`: 3,972 → 2,287 lines; added `ecosystem-versions.json` |
| [#9](https://github.com/Daylily-Informatics/daylily-ursa/pull/9) | Extract remaining routes (Part 2/2) | `workset_api.py`: 2,287 → 770 lines; 6 new router modules |
| [#10](https://github.com/Daylily-Informatics/daylily-ursa/pull/10) | Fix billing test date drift | Fixed hardcoded dates outside 30-day billing window |
| [#11](https://github.com/Daylily-Informatics/daylily-ursa/pull/11) | Add `/v1/` prefix to all API routes | All API routes available under `/v1/` with backward compat |

---

## Architectural Assessment

### Strengths

1. **Ephemeral-by-design infrastructure** — Clusters are cattle, not pets. All state lives in durable stores (S3, TapDB). This is the single most important architectural decision.
2. **Cost-as-a-first-class metric** — Custom Snakemake telemetry captures EC2 instance type, spot price, and actual cost per task. No other public bioinformatics framework does this.
3. **Clean separation: infrastructure vs. analysis** — `daylily-ephemeral-cluster` (infra) and `daylily-omics-analysis` (workflows) are independent. The cluster layer is workflow-manager-agnostic.
4. **FSx-as-shared-scratch with S3 mirror** — Solves distributed storage without requiring users to think about object storage during analysis.
5. **Spot fleet diversity** — Heterogeneous instance partitions (i8–i192) with per-partition spot price caps. Sophisticated spot market engineering.
6. **TapDB-backed workset state machine** — Conditional writes for locking, well-defined state enum, audit trail.
7. **Auth as a shared library** — `daylily-cognito` is a standalone library any FastAPI service can consume.

### Novel/Publishable Contributions

- **Custom Snakemake cost-performance telemetry** — The ability to say "Tool X costs $Y per genome on instance type Z with accuracy A" is a publishable contribution.
- **Hybrid multi-platform variant calling** — Illumina + ONT, Illumina + PacBio, Ultima + ONT, Roche SBX duplex support rivals commercial LIMS systems.
- **Spot fleet engineering for genomics** — Heterogeneous fleet with per-partition max-spot-price caps.

### Identified Risks (Remaining)

| Risk | Severity | Status |
|------|----------|--------|
| `daylily_ursa` namespace collision across 3 repos | 🔴 High | Open — needs rename strategy |
| SSH-based headnode interrogation | 🟡 Medium | Open — works for v1, won't scale |
| IAM policy uses `Resource: "*"` for FSx | 🟡 Medium | Open — acceptable if documented |
| DAYOA has no `pyproject.toml` | 🟡 Medium | Open |
| Multiple conda envs with unclear boundaries | 🟢 Low | Documented in conda matrix appendix |

---

## API Decomposition Detail

The 3,972-line `workset_api.py` monolith was decomposed into 14 focused router modules:

| Module | Routes | Purpose |
|--------|--------|---------|
| `routes/customer_worksets.py` | 11 | Customer workset CRUD, cancel, retry, archive, delete, restore |
| `routes/dashboard.py` | 3 | Activity feed, cost history, usage stats |
| `routes/billing.py` | 3 | Summary, invoice, per-workset billing |
| `routes/customers.py` | 5 | Customer CRUD + usage |
| `routes/files.py` | 6 | File upload, preview, download, delete |
| `routes/manifests.py` | 4 | Sample manifest management |
| `routes/s3.py` | 5 | Bucket discovery, validation, IAM policy |
| `routes/clusters.py` | 2 | Cluster listing + deletion |
| `routes/monitoring.py` | 4 | Queue stats, scheduler, command logs |
| `routes/worksets.py` | 7 | Core workset CRUD + validation |
| `routes/portal.py` | ~20 | Web portal (HTML pages) |
| `routes/utilities.py` | 2 | Cost estimation, YAML generation |
| `routes/dependencies.py` | — | Shared dependency injection |
| `routes/__init__.py` | — | Router registration |

What remains in `workset_api.py` (~810 lines): app factory, middleware, health endpoint, static files, router wiring.

---

## Cross-Repo Version Matrix

Created `config/ecosystem-versions.json` and `ursa version --ecosystem` CLI command:

| Component | Version | Repository |
|-----------|---------|------------|
| daylily-ursa | 0.1.7 | Daylily-Informatics/daylily-ursa |
| daylily-ephemeral-cluster | 0.7.605 | Daylily-Informatics/daylily-ephemeral-cluster |
| daylily-omics-analysis | 0.7.602 | Daylily-Informatics/daylily-omics-analysis |
| daylily-cognito | 0.1.24 | Daylily-Informatics/daylily-cognito |
| daylily-omics-references | 0.3.1 | Daylily-Informatics/daylily-omics-references |

---

## Conda Environment Matrix

Three environments exist across the ecosystem:

| Environment | Repo | Where It Runs | Activation |
|---|---|---|---|
| **URSA** | daylily-ursa | macOS (dev), any server | `source ./ursa_activate` |
| **DAY-EC** | daylily-ephemeral-cluster | macOS + headnode | `bin/init_dayec` |
| **DAYOA** | daylily-omics-analysis | Headnode only | `source dyoainit` |

**Key finding**: URSA and DAY-EC conda specs are ~95% identical. They could be unified into a shared `daylily-base-env.yaml`. DAYOA is fundamentally different (bioinformatics tools, headnode-only) and should stay separate.

---

## Remaining Work (Not Addressed)

| Item | Notes |
|------|-------|
| `daylily_ursa` namespace collision | Needs rename: `daylily_ec`, `daylily_oa`, `daylily_ursa` (or shared `daylily-core`) |
| URSA/DAY-EC conda env unification | Low-hanging fruit per matrix research; deferred by owner |
| Headnode agent (push-based monitoring) | Replaces SSH; architecture-level change |
| DAYOA `pyproject.toml` migration | Legacy `setup.py` with hardcoded version |

---

## Recommendations for Publication

**Lead with**:
1. Ephemeral infrastructure philosophy (clusters as cattle)
2. Cost-as-a-metric telemetry (the killer feature)
3. Concrete numbers: "$5/genome WGS at 30x"
4. Multi-platform support breadth

**Fix before publishing**:
1. `daylily_ursa` namespace collision
2. Temporary files in omics-analysis/ephemeral-cluster repos
3. Document cost guardrails prominently

---

*Report generated during architectural survey session on 2026-02-28.*
*6 PRs merged. 785 tests passing. Zero failures.*
