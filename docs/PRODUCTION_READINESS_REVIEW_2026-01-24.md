# Production Readiness Review (daylily-ursa) — 2026-01-24

Author: Forge

## Scope
Read-only review of code + docs focused on “what remains” for production readiness.

## Assumptions
- Target is AWS-hosted service running multiple replicas behind a load balancer.
- Handles genomics-adjacent workflows; treat logs/metadata as potentially sensitive.
- “Production-ready” includes security hardening, CI, observability, and runbooks.

## Constraints / What I did NOT inspect
- Per core directives, I did not inspect any directory named `.ignore/`.

## Current strengths
- Solid packaging baseline via `pyproject.toml` (extras: `auth`, `dev`, `cluster`).
- Clear entry points defined (`ursa`, `daylily-workset-api`, `daylily-workset-monitor`, etc.).
- Multi-region primitives exist (`daylib/workset_multi_region.py`) + documentation.
- Cross-region S3 risk is actively addressed via `RegionAwareS3Client`.
- Guardrails exist in settings (e.g., wildcard CORS forbidden in production; demo mode validation).

## Production blockers (fix before go-live)
1. **CLI server launcher is not production-package-safe**
   - `daylib/cli/server.py` shells out to `bin/daylily-workset-api` relative to repo root.
   - After `pip install daylily-ursa`, that `bin/` file likely won’t exist.
   - This will break `ursa server start` in real deployments.
   - TODO: make the executable scripts pip available console scripts instead of only repo-relative paths

2. **No visible CI configuration in-repo**
   - At review time, `.github/` was not present (so no GitHub Actions workflows were found).
   - Production requires automated: unit tests, lint, type checks, and basic security checks.
   - TODO: add some CI to the repo

## High-risk gaps (strongly recommended)
- **Rate limiting is not distributed**
  - `daylib/rate_limiting.py` implements in-memory token buckets.
  - In production (multiple pods/instances), rate limiting must be shared (e.g., Redis) or enforced at the edge.
  - TODO: add to new document 'FUTURE_DEVELOPMENT_PLAN.md`

- **Secrets and env hardening**
  - `Settings.session_secret_key` defaults to a dev value.
  - Production should fail fast when secrets are unset/unsafe (don’t “warn and continue”).
  - TODO: MAke tis change to fail fast

- **Portal S3 access consistency**
  - `daylib/routes/portal.py` imports `RegionAwareS3Client`, but some routes still create a plain boto3 S3 client for presigned URLs.
  - For BYOB/cross-region buckets this will reintroduce 301/region mismatch failures.
  - TODO: make sure all S3 access is region-aware and consistent

- **Observability baseline**
  - CloudWatch metric emission exists in `WorksetStateDB`, but you still need:
    - consistent request/trace IDs
    - structured logs (JSON) with redaction policy
    - dashboards/alarms and on-call runbook
    - TODO: add to new document 'FUTURE_DEVELOPMENT_PLAN.md`

## Documentation correctness (quick wins)
- `docs/QUICKSTART_WORKSET_MONITOR.md` appears inconsistent about ports (example uses `--port 8000` but references `http://localhost:8001`).
- Ensure docs reflect the packaged entry points in `pyproject.toml` (avoid referencing repo-relative scripts).
- TODO: review all documentation vs the actual state of the database and update accordingly

### Moonshot (hardening for regulated / multi-tenant scale)
- Threat model + security review (Cognito, IAM least privilege, bucket policies for BYOB).
- Full audit logging + data classification + automated secret scanning.
- End-to-end integration tests with mocked AWS (moto) plus a staging environment smoke test.
- Formal multi-region failover tests (chaos-style) and explicit consistency guarantees.
- TODO: add to new document 'FUTURE_DEVELOPMENT_PLAN.md`

# Add to new document 'FUTURE_DEVELOPMENT_PLAN.md`
- actual customer billing and accounting with AWS billing integration or stripe or someting
- PRoject concept
- moving to use tapdb?
- moving to use a single source of truth for Source, biospecimen, etc.... with portal and lims
- test apis with token authentication, etc

# Manual Notes From UI Debugging (to be digested in to clear prompt actions)
Modifications

http://localhost:8001/portal/usage
billing info card should have edit button.

http://localhost:8001/portal/register

Change the title page name to Ursa and change the flower to the constellationConfirm the Resource Limits specified are captured int he database. Make them visible on the account page if not already.If an aws account ID is entered, make sure we can see it… if we can not, propose to the user what fixes need to be in place to see the s3 resources .?What is cost center here for?  Is it same as budget?Please add a legal, very basic TOS and Privacy document which is modified for the company LSMC.Admin users should see an ‘admin’ tab, with features like:
- Add user
- Set/remove admin privileges
- Change user password

Only admin users should see the monitor page and cost details on the cluster page. All costs presented to non-admin users should only be for the budgets they have visibility to.On http://localhost:8001/portal/files/bucketsThe link buttons card should be moved above the linked buckets list, and styled to be as wide as the linked buckets list.

Remove discover files card and simply have a button that redirects to the auot-discover files page.
Edit bucket should just be a button on the listed linked bucket that opens a dialog to edit the bucket properties.http://localhost:8001/portal/biospecimen
Should be named ‘add new biospecimen’, subjects should have their own http://localhost:8001/portal/subjectsmove the Add New card above the list and style it so it is as wide as the list card.
http://localhost:8001/portal/clustersAdmin users should be provided a button to delete cluster, which should have 2 acknowledgement prompts. It should run AWS_PROFILE=<profile> pcluster delete-cluster -n <cluster-name> —region <region>Logs

http://localhost:8001/portal/files/uploadFails when trying to upload files to the provisioned and linked bucket
INFO:     127.0.0.1:57432 - "GET /static/css/main.css?v=1769239044 HTTP/1.1" 200 OK
INFO:     127.0.0.1:57430 - "GET /static/js/main.js?v=1769239044 HTTP/1.1" 200 OK
2026-01-23 23:17:54,824 - daylily.routes.portal - INFO - Upload request from johnm@lsmc.life: RIH0_ANA0-HG002_DBC0_0.R1.fastq.gz to bucket lb-33db76e4e0f0948a
INFO:     127.0.0.1:57439 - "POST /portal/files/upload HTTP/1.1" 403 Forbidden
2026-01-23 23:17:54,897 - daylily.routes.portal - INFO - Upload request from johnm@lsmc.life: RIH0_ANA0-HG002_DBC0_0.R2.fastq.gz to bucket lb-33db76e4e0f0948a
INFO:     127.0.0.1:57439 - "POST /portal/files/upload HTTP/1.1" 403 Forbidden
INFO:     127.0.0.1:57439 - "GET /api/monitor/logs?lines=100 HTTP/1.1" 200 OK
INFO:     127.0.0.1:57439 - "GET /api/monitor/status HTTP/1.1" 200 OK
2026-01-23 23:18:56,447 - daylily.routes.portal - INFO - Upload request from johnm@lsmc.life: RIH0_ANA0-HG002_DBC0_0.R1.fastq.gz to bucket lb-33db76e4e0f0948a
INFO:     127.0.0.1:57447 - "POST /portal/files/upload HTTP/1.1" 403 Forbidden
2026-01-23 23:18:56,525 - daylily.routes.portal - INFO - Upload request from johnm@lsmc.life: RIH0_ANA0-HG002_DBC0_0.R2.fastq.gz to bucket lb-33db76e4e0f0948a
INFO:     127.0.0.1:57447 - "POST /portal/files/upload HTTP/1.1" 403 Forbidden
INFO:     127.0.0.1:57447 - "GET /api/monitor/logs?lines=100 HTTP/1.1" 200 OK
INFO:     127.0.0.1:57447 - "GET /api/monitor/status HTTP/1.1" 200 OK
# WORKSETS 
- Should only be visible to the creating user owning it.
- Tighten up the talbe cells with less padding by 40%
- If ssh commands are running and the monitor is shut down, do the ssh commands complete in the bg>? Does the monitor gracefully restart knowing the outcome of these ssh jobs?
- If worksets are retried, the selected worksets to retry should not be edited, but left in their current status,… a new workset with the same name prefix, but new date time suffix should be cloned from it and registered as ready.


http://localhost:8001/portal/worksets
The footer should not be forced to always be visible , but truly only appear at the bottom of the page (this applies to all pages)# User link upper right (when hovering, the text displayed is off the visible page)http://localhost:8001/portal- The Samples column in Recent Worksets is always zero and this is incorrect.
The rename button for account details does not work, should be renamed to ‘edit’ and allow editing fieldshttp://localhost:8001/portal/files/register- bulk import does not work. Fix and add tests
Password min length:  set to minimum EIGHT chars.Generate API token does not work



# For pipeline page
http://localhost:8001/portal/clusters
run every 20 minutes the ssh commands on each headnode to gather info about the compute fleet:
## Best single sinfo
sinfo -N -o "%.20N %.9P %.6t %.10c %.10m %.10G %.20R" | head -n 5
            NODELIST PARTITION  STATE       CPUS     MEMORY       GRES            PARTITION
      i8-dy-r6gb64-1       i8*  idle~          8      62259     (null)                   i8
      i8-dy-r7gb64-1       i8*  idle~          8      62259     (null)                   i8
   i128-dy-c6gb256-1      i128  idle~        128     249036     (null)                 i128
   i128-dy-c6gb256-2      i128  idle~        128     249036     (null)                 i128
## Why nodes are unhappy
sinfo -N -r

sinfo -N -r | head -n 5
NODELIST              NODES  PARTITION STATE
i8-dy-r6gb64-1            1        i8* idle~
i8-dy-r7gb64-1            1        i8* idle~
i128-dy-c6gb256-1         1       i128 idle~
i128-dy-c6gb256-2         1       i128 idle~

add this as a section to each cluster card and note the datetime generated. allow manual force refresh

### Super detailed info (only on demand from cluter detail view)
scontrol show nodes| head -n 50                 [28/1934]
NodeName=i8-dy-r6gb64-1 Arch=x86_64 CoresPerSocket=1
   CPUAlloc=0 CPUEfctv=8 CPUTot=8 CPULoad=0.00
   AvailableFeatures=dynamic,r6i.2xlarge,r6gb64
   ActiveFeatures=dynamic,r6i.2xlarge,r6gb64
   Gres=(null)
   NodeAddr=i8-dy-r6gb64-1 NodeHostName=i8-dy-r6gb64-1 Version=24.05.8
   OS=Linux 6.8.0-1029-aws #31~22.04.1-Ubuntu SMP Thu Apr 24 21:16:18 UTC 2025
   RealMemory=62259 AllocMem=0 FreeMem=54627 Sockets=8 Boards=1
   State=IDLE+CLOUD+POWERED_DOWN ThreadsPerCore=1 TmpDisk=0 Weight=1000 Owner=N/A MCS_label=N/A
   Partitions=i8
   BootTime=2026-01-23T13:37:33 SlurmdStartTime=2026-01-23T13:42:09
   LastBusyTime=Unknown ResumeAfterTime=None
   CfgTRES=cpu=8,mem=62259M,billing=8
   AllocTRES=
   CurrentWatts=0 AveWatts=0

   Reason=Scheduler health check failed [root@2026-01-23T14:13:13]


NodeName=i8-dy-r7gb64-1 Arch=x86_64 CoresPerSocket=1
   CPUAlloc=0 CPUEfctv=8 CPUTot=8 CPULoad=0.00
   AvailableFeatures=dynamic,r7i.2xlarge,r7gb64
   ActiveFeatures=dynamic,r7i.2xlarge,r7gb64
   Gres=(null)
   NodeAddr=i8-dy-r7gb64-1 NodeHostName=i8-dy-r7gb64-1 Version=24.05.8
   OS=Linux 6.8.0-1029-aws #31~22.04.1-Ubuntu SMP Thu Apr 24 21:16:18 UTC 2025
   RealMemory=62259 AllocMem=0 FreeMem=53982 Sockets=8 Boards=1
   State=IDLE+CLOUD+POWERED_DOWN ThreadsPerCore=1 TmpDisk=0 Weight=1000 Owner=N/A MCS_label=N/A
   Partitions=i8
   BootTime=2026-01-24T07:29:07 SlurmdStartTime=2026-01-24T07:35:06
   LastBusyTime=Unknown ResumeAfterTime=None
   CfgTRES=cpu=8,mem=62259M,billing=8
   AllocTRES=
   CurrentWatts=0 AveWatts=0


NodeName=i128-dy-c6gb256-1 CoresPerSocket=1
   CPUAlloc=0 CPUEfctv=128 CPUTot=128 CPULoad=0.00
   AvailableFeatures=dynamic,c6gb256
   ActiveFeatures=dynamic,c6gb256
   Gres=(null)
   NodeAddr=i128-dy-c6gb256-1 NodeHostName=i128-dy-c6gb256-1
   RealMemory=249036 AllocMem=0 FreeMem=N/A Sockets=128 Boards=1
   State=IDLE+CLOUD+POWERED_DOWN ThreadsPerCore=1 TmpDisk=0 Weight=1000 Owner=N/A MCS_label=N/A
   Partitions=i128
   BootTime=None SlurmdStartTime=None
   LastBusyTime=Unknown ResumeAfterTime=None
   CfgTRES=cpu=128,mem=249036M,billing=128
   AllocTRES=
   CurrentWatts=0 AveWatts=0

# Cost caclulatons
The cost calculation per-workset should be triggered after the fst->s3 export completes, so it populates the cost info per sample and workset properly.

# Usage page
in http://localhost:8001/portal/usage, the card cost numbers do not match the circle pie chart cost numbers. These should draw from the same chart.

- export report button not working

- transfer cost should always be presented as 3 options: w/in region, across regions, back to internet

# Monitor
- should be only accessible to admin users.
- there should be a filter to grep out all of the remote ssh commands and aws cli commands that are executed by the monitor for a specified workset, so the admin can manutally try rerunning the sequence of commands if needed.