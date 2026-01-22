# daylily-ursa

**Daylily Workset Management API** — Automated analysis workset manager for genomics pipelines.

## Overview

Daylily Ursa provides a comprehensive workset management system for orchestrating genomics analysis pipelines. It handles:

- **Workset Lifecycle Management** — Create, monitor, and manage analysis worksets through DynamoDB-backed state machine
- **File Registry** — Track and validate input/output files across S3 buckets
- **Customer Portal** — Web-based interface for customers to submit and monitor worksets
- **Biospecimen Management** — Track samples, manifests, and metadata
- **Multi-Region Support** — Coordinate worksets across AWS regions
- **Notifications** — SNS-based alerts for workset state changes
- **Storage Metrics** — Track and display workset directory sizes and storage consumption
- **Cognito Authentication** — Optional AWS Cognito integration for secure multi-tenant access

## Quick Start (Development)

```bash
# Activate the development environment (creates conda env if needed)
source ./ursa_activate

# Check system status
ursa info

# Run tests
ursa test run

# Start the API server (no auth, development mode)
ursa server start
```

## CLI Tools

### `ursa_activate` — Environment Setup

Source this script to set up the development environment:

```bash
source ./ursa_activate
```

This will:
1. Create the `URSA` conda environment from `config/ursa_env.yaml` (if not exists)
2. Activate the conda environment
3. Install the package in development mode
4. Add CLI tools to PATH
5. Enable tab completion for the `ursa` CLI

### `ursa` — Management CLI

The main CLI tool for managing the project. Uses Typer with subcommand groups:

```bash
ursa <group> <command> [args]
```

**Command Groups:**

| Group | Description |
|-------|-------------|
| `ursa server` | API server management (start, stop, status, logs) |
| `ursa monitor` | Workset monitor daemon (start, stop, status, logs) |
| `ursa aws` | AWS resource management (setup, status, teardown) |
| `ursa cognito` | Cognito authentication (setup, status, set-admin, set-password) |
| `ursa test` | Testing and code quality (run, cov, lint, format, typecheck) |
| `ursa env` | Environment and configuration (status, generate, clean) |

**Top-Level Commands:**
- `ursa version` — Show version information
- `ursa info` — Show system status and configuration
- `ursa --help` — Show all available commands

**Examples:**

```bash
# Server management
ursa server start              # Start API server as daemon
ursa server start --no-daemon  # Start in foreground
ursa server stop               # Stop the server
ursa server status             # Check server status
ursa server logs               # Tail server logs

# Testing
ursa test run                  # Run test suite
ursa test cov                  # Run with coverage
ursa test lint                 # Run ruff linter
ursa test format               # Format code

# AWS resources
ursa aws setup                 # Create DynamoDB tables
ursa aws status                # Check resource status
ursa aws teardown              # Delete all resources (destructive)

# Cognito authentication
ursa cognito setup             # Create Cognito User Pool
ursa cognito status            # Check Cognito configuration
ursa cognito set-admin         # Grant/revoke admin status
```

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
daylily-workset-api --host 0.0.0.0 --port 8001

# Start the workset monitor
daylily-workset-monitor config/workset-monitor-config.yaml
```

## Architecture

```
daylib/
├── workset_api.py          # FastAPI application entry point
├── workset_state_db.py     # DynamoDB state management
├── workset_monitor.py      # S3 workset monitoring daemon
├── workset_integration.py  # DynamoDB/S3 integration layer
├── workset_metrics.py      # Storage and performance metrics
├── workset_customer.py     # Customer/tenant management
├── workset_auth.py         # Authentication utilities
├── workset_multi_region.py # Multi-region coordination
├── file_registry.py        # File tracking and validation
├── biospecimen.py          # Sample/manifest management
├── config.py               # Pydantic settings (env vars)
├── cli/                    # Typer CLI modules
│   ├── __init__.py         # Main CLI app
│   ├── server.py           # Server commands
│   ├── monitor.py          # Monitor commands
│   ├── aws.py              # AWS resource commands
│   ├── cognito.py          # Cognito commands
│   ├── test.py             # Test commands
│   └── env.py              # Environment commands
└── routes/                 # FastAPI route modules
    ├── portal.py           # Customer portal routes
    ├── worksets.py         # Workset CRUD routes
    ├── utilities.py        # Utility endpoints
    └── dependencies.py     # Shared dependencies
```

## Configuration

Configuration is managed via environment variables or a `.env` file. Generate a template:

```bash
ursa env generate
```

**Key Environment Variables:**

```bash
# AWS Configuration (required)
AWS_PROFILE=your-profile-name
AWS_DEFAULT_REGION=us-west-2

# S3 Configuration
# NOTE: S3 buckets are discovered from cluster tags (aws-parallelcluster-monitor-bucket)
# No bucket environment variables are required. See ~/.ursa/ursa.yaml for region config.

# DynamoDB Tables (auto-created if missing)
WORKSET_TABLE_NAME=daylily-worksets
CUSTOMER_TABLE_NAME=daylily-customers
DAYLILY_FILE_REGISTRY_TABLE=daylily-file-registry

# Authentication (optional)
ENABLE_AUTH=false
COGNITO_USER_POOL_ID=us-west-2_xxxxxxxx
COGNITO_CLIENT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxx
SESSION_SECRET_KEY=change-this-in-production
WHITELIST_DOMAINS=all  # or comma-separated: company.com,partner.org

# Server
URSA_HOST=0.0.0.0
URSA_PORT=8001

# Multi-Region (optional)
DAYLILY_MULTI_REGION=false
DAYLILY_PRIMARY_REGION=us-west-2
```

See `docs/AUTHENTICATION_SETUP.md` and `docs/MULTI_REGION.md` for detailed configuration guides.

## Features

### Customer Portal

Web-based interface at `/portal/` providing:
- Dashboard with workset overview and storage metrics
- Workset list with status, progress, and directory sizes
- Workset detail view with resources, samples, and logs
- File registry for managing input files
- Usage tracking and storage breakdown
- Cluster management (admin only)

### Storage Metrics

Workset directory sizes are automatically calculated during the pre-export phase and displayed throughout the UI:
- **Dashboard**: Total storage across all worksets
- **Workset List**: Per-workset directory size column
- **Workset Detail**: Storage in the Resources card
- **Usage Page**: Storage breakdown by workset

### Authentication Modes

1. **No Auth** (development): `ursa server start`
2. **Cognito Auth** (production): Set `ENABLE_AUTH=true` with Cognito configuration

See `docs/AUTHENTICATION_SETUP.md` for setup instructions.

## Documentation

Detailed guides are available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| `AUTHENTICATION_SETUP.md` | Cognito authentication configuration |
| `CUSTOMER_PORTAL.md` | Portal features and multi-tenant support |
| `MULTI_REGION.md` | Multi-region deployment guide |
| `BILLING_INTEGRATION.md` | AWS billing and cost allocation |
| `IAM_SETUP_GUIDE.md` | Required IAM permissions |
| `QUICKSTART_WORKSET_MONITOR.md` | Monitor daemon setup |
| `WORKSET_STATE_DIAGRAM.md` | Workset state machine reference |

## Related Projects

- [daylily-ephemeral-cluster](https://github.com/Daylily-Informatics/daylily-ephemeral-cluster) — AWS ParallelCluster infrastructure for running genomics pipelines

## License

MIT
