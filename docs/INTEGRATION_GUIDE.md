# Ursa Integration Guide

This guide describes how the Ursa UI/API layer integrates with the processing layer, and how persistence works with TapDB graph objects (Postgres).

## Architecture Overview

- **UI/API layer**
  - Portal (browser) + FastAPI server
  - Reads/writes workflow state via TapDB graph persistence
- **Processing layer**
  - Workset monitor/worker process
  - Observes S3 workset directories (sentinels) and executes pipelines
  - Updates workflow state in TapDB

```mermaid
flowchart TB
    subgraph UI["UI/API Layer"]
        Portal[Portal]
        API[FastAPI Server]
    end

    subgraph Persistence["Persistence"]
        TapDB[(TapDB Graph<br/>(Postgres))]
    end

    subgraph Processing["Processing Layer"]
        Monitor[Workset Monitor / Worker]
        S3[(S3 Workset Paths<br/>+ Sentinels)]
        Compute[Compute Engine]
    end

    Portal --> API
    API --> TapDB
    API --> S3
    Monitor --> S3
    Monitor --> TapDB
    Monitor --> Compute
```

## TapDB Configuration (Strict Namespace)

Ursa uses TapDB in strict namespace mode. Configure TapDB via environment variables:

```bash
export TAPDB_STRICT_NAMESPACE=1
export TAPDB_CLIENT_ID=local
export TAPDB_DATABASE_NAME=ursa
export TAPDB_ENV=dev   # dev|test|prod
```

Then bootstrap TapDB (preferred):

```bash
tapdb config init --client-id local --database-name ursa --env dev
tapdb bootstrap local
```

Ursa can also bootstrap the required templates:

```bash
ursa aws setup
ursa aws status
```

## API Server

Start the API server using the packaged entry points:

```bash
export AWS_PROFILE=lsmc
ursa server start
```

Or run directly:

```bash
daylily-workset-api --host 0.0.0.0 --port 8914 --bootstrap-tapdb
```

### Mounted TapDB Admin Surface

Ursa mounts TapDB admin inside the same FastAPI process (no separate TapDB web
server required) under:

- `/admin/tapdb`

Access control is enforced by Ursa before requests reach TapDB:

- unauthenticated -> `307` redirect to `/portal/login`
- authenticated non-admin -> `403` JSON
- authenticated admin -> allowed

Mounted mode explicitly bypasses TapDB-local auth by setting:

- `TAPDB_ADMIN_DISABLE_AUTH=true`
- `TAPDB_ADMIN_DISABLED_USER_ROLE=admin`

This preserves Ursa as the sole auth/session gate while keeping standalone
TapDB usage unchanged.

Mount configuration:

```bash
URSA_TAPDB_MOUNT_ENABLED=true
URSA_TAPDB_MOUNT_PATH=/admin/tapdb
```

## Workset Monitor

The workset monitor reads a YAML config file and runs continuously (or `--once` for a single iteration).

Example:

```bash
daylily-workset-monitor ~/.config/ursa/workset-monitor-config.yaml --enable-tapdb
```

TapDB configuration is read from `TAPDB_*` environment variables. There is no per-service “table name” configuration.

## Troubleshooting

### Worksets Not Being Discovered

1. Confirm TapDB is configured (`TAPDB_CLIENT_ID`, `TAPDB_DATABASE_NAME`, `TAPDB_ENV`).
2. Confirm templates are present:
   - `ursa aws status`
3. Verify S3 permissions and bucket/prefix values in your monitor config.
4. Check monitor logs.

### Template Bootstrap Errors

If you see errors mentioning missing templates, bootstrap:

```bash
ursa aws setup
```
