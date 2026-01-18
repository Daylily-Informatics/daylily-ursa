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

## Quick Start (Development)

```bash
# Activate the development environment (creates conda env if needed)
source ./dayu_activate

# Check system status
dayu status

# Run tests
dayu test

# Start the API server
dayu server
```

## CLI Tools

### `dayu_activate` - Environment Setup

Source this script to set up the development environment:

```bash
source ./dayu_activate
```

This will:
1. Create the `DAYU` conda environment from `config/dayu_env.yaml` (if not exists)
2. Activate the conda environment
3. Install the package in development mode
4. Add CLI tools to PATH

### `dayu` - Management CLI

The main CLI tool for managing the project:

```bash
dayu <command> [args]
```

**Testing & Quality:**
- `dayu test` - Run the complete test suite
- `dayu test-cov` - Run tests with coverage report
- `dayu lint` - Run ruff linter
- `dayu format` - Format code with ruff
- `dayu typecheck` - Run mypy type checker

**Server Commands:**
- `dayu server` - Start the FastAPI server
- `dayu server-dev` - Start server with auto-reload

**AWS Resource Management:**
- `dayu setup-aws` - Create required AWS resources (DynamoDB tables)
- `dayu teardown-aws` - Delete all AWS resources
- `dayu aws-status` - Check status of AWS resources

**Environment:**
- `dayu env` - Generate `.env` file template
- `dayu status` - Check system status
- `dayu clean` - Remove cached files and build artifacts
- `dayu version` - Show version information

## Installation (Production)

```bash
pip install daylily-ursa

# With authentication support
pip install daylily-ursa[auth]

# For development
pip install daylily-ursa[dev]
```

## Alternative Quick Start

```bash
# Start the API server directly
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
