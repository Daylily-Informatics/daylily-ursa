# daylily-ursa Docs

Use the root [README](../README.md) as the current repo overview and operational entry point. This index separates current guides from archived background notes.

## Current Guides

- [Ursa-Atlas return contract](ursa_atlas_return_contract.md): canonical result-return contract from Ursa into Atlas.
- [Google OAuth default](GOOGLE_OAUTH_DEFAULT.md): default Cognito plus Google Hosted UI setup helper and ownership boundaries.
- [TapDB admin mount status](tapdb_mount_execplan.md): current mounted TapDB admin behavior inside the Ursa FastAPI app.
- [Conformance audit](../ursa-conformance-directive.md): current status of the Atlas/Bloom alignment work.
- [Auth E2E README](../tests/e2e/README.md): Playwright auth browser test prerequisites and scope.

## Archived Background

These files are retained for context. They describe older workset-monitor and cluster-management workflows and should not override the root README or current code:

- [IAM setup guide](IAM_SETUP_GUIDE.md)
- [Multi-region cluster discovery](MULTI_REGION.md)
- [Workset state diagram](WORKSET_STATE_DIAGRAM.md)

For current runtime operations, start from `source ./activate <deploy-name>` and use `ursa ...` first. Use `tapdb ...`, `daycog ...`, or `daylily-ec ...` only where Ursa explicitly delegates that lifecycle.
