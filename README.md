# daylily-ursa

**Daylily Workset Management API** - Automated analysis workset manager for genomics pipelines.

## Overview

Daylily Ursa provides a comprehensive workset management system for orchestrating genomics analysis pipelines. It handles:

- **Workset Lifecycle Management** - Create, monitor, and manage analysis worksets through DynamoDB-backed state machine
- **File Registry** - Track and validate input/output files across S3 buckets
- **Customer Portal** - Web-based interface for customers to submit and monitor worksets
- **Biospecimen Management** - Track samples, manifests, and metadata
- **Multi-Region Support** - Coordinate worksets across AWS regions
- **Notifications** - SNS-based alerts for workset state changes

## Installation

```bash
pip install daylily-ursa

# With authentication support
pip install daylily-ursa[auth]

# For development
pip install daylily-ursa[dev]
```

## Quick Start

```bash
# Start the API server
daylily-workset-api --host 0.0.0.0 --port 8000

# Start the workset monitor
daylily-workset-monitor --config config/workset-monitor-config.yaml
```

## Architecture

```
daylib/
├── workset_api.py          # FastAPI application
├── workset_state_db.py     # DynamoDB state management
├── workset_monitor.py      # S3 workset monitoring
├── workset_integration.py  # DynamoDB/S3 integration layer
├── file_registry.py        # File tracking and validation
├── biospecimen.py          # Sample/manifest management
├── routes/                 # API route modules
│   ├── portal.py           # Customer portal routes
│   ├── worksets.py         # Workset CRUD routes
│   └── utilities.py        # Utility endpoints
└── ...
```

## Configuration

Set environment variables or use a `.env` file:

```bash
AWS_REGION=us-east-1
DAYLILY_CONTROL_BUCKET=my-control-bucket
DAYLILY_CONTROL_PREFIX=worksets/
DYNAMODB_TABLE_NAME=daylily-worksets
```

## Related Projects

- [daylily-ephemeral-cluster](https://github.com/Daylily-Informatics/daylily-ephemeral-cluster) - AWS ParallelCluster infrastructure for running genomics pipelines

## License

MIT
