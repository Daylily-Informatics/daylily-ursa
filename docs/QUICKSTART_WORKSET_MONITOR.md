# Quick Start: Workset Monitor + API

This quick start gets an Ursa API + workset monitor running locally.

## Prerequisites

- Python 3.10+
- AWS credentials configured (typically via `AWS_PROFILE`)
- TapDB configured (strict namespace)

## 1) Configure TapDB (Strict Namespace)

```bash
export TAPDB_STRICT_NAMESPACE=1
export TAPDB_CLIENT_ID=local
export TAPDB_DATABASE_NAME=ursa
export TAPDB_ENV=dev
```

Bootstrap TapDB (preferred):

```bash
tapdb config init --client-id local --database-name ursa --env dev
tapdb bootstrap local
```

Or bootstrap templates from Ursa:

```bash
ursa aws setup
```

## 2) Register Your First Workset

```python
from daylib.workset_state_db import WorksetPriority, WorksetStateDB

db = WorksetStateDB()
db.bootstrap()

db.register_workset(
    workset_id="my-first-workset",
    bucket="my-s3-bucket",
    prefix="worksets/my-first-workset/",
    priority=WorksetPriority.NORMAL,
    metadata={"samples": [{"sample_id": "S1"}]},
    customer_id="cust-demo",
)
```

## 3) Start the API

```bash
export AWS_PROFILE=lsmc
ursa server start --foreground
```

Or run directly:

```bash
daylily-workset-api --host 0.0.0.0 --port 8914 --bootstrap-tapdb --verbose
```

The API will be available at `https://localhost:8914` if you start via `ursa server start` (it configures TLS cert paths).

## 4) Run the Monitor

```bash
daylily-workset-monitor ~/.config/ursa/workset-monitor-config.yaml --enable-tapdb
```

## Basic Usage (Python)

```python
from daylib.workset_state_db import WorksetState, WorksetStateDB

db = WorksetStateDB()

workset = db.get_workset("my-first-workset")
print(workset["state"])

db.update_state(
    workset_id="my-first-workset",
    new_state=WorksetState.IN_PROGRESS,
    reason="Processing started",
)
```

