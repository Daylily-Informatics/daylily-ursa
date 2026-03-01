# Authentication Setup Guide

This guide explains how to run the Workset Monitor API with or without authentication.

## Overview

The Workset Monitor API supports two modes:

1. **Without Authentication** (default) - No authentication required, suitable for:
   - Development and testing
   - Internal deployments behind VPN/firewall
   - Environments with network-level security

2. **With Authentication** - AWS Cognito JWT-based authentication, suitable for:
   - Production deployments
   - Multi-tenant environments
   - Public-facing APIs

## Running Without Authentication

### Installation

```bash
# Install base dependencies (no authentication)
pip install -e .
```

### Running the Server

```bash
# Using the example script
python examples/run_api_without_auth.py

# Or using uvicorn directly
uvicorn daylib.workset_api:app --host 0.0.0.0 --port 8914
```

### Python Code Example

```python
from daylib.workset_api import create_app
from daylib.workset_state_db import WorksetStateDB

# Initialize components
state_db = WorksetStateDB("daylily-worksets", "us-west-2")

# Create app WITHOUT authentication
app = create_app(
    state_db=state_db,
    enable_auth=False,  # Disable authentication
)

# Run with uvicorn
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8914)
```

### Testing Without Authentication

```bash
# All endpoints work without authentication
curl http://localhost:8914/worksets
curl http://localhost:8914/queue/stats
curl -X POST http://localhost:8914/customers \
  -H "Content-Type: application/json" \
  -d '{"customer_name": "Test", "email": "test@example.com"}'
```

## Running With Authentication

### Installation

```bash
# Install with authentication support
pip install -e ".[auth]"

# Or install python-jose separately
pip install 'python-jose[cryptography]'
```

### Cognito Setup via `daycog` (Required)

Use the `daylily-cognito` operational CLI rather than direct AWS commands:

```bash
source ../daylily-cognito/daycog_activate
daycog setup --name daylily-workset-users --port 8914 --profile <aws-profile> --region us-west-2
```

This writes/updates `~/.config/daycog/default.env` with:
- `COGNITO_REGION`
- `COGNITO_USER_POOL_ID`
- `COGNITO_APP_CLIENT_ID`

Create a test user:

```bash
daycog add-user test@example.com --password 'YourPassword123!'
```

### Running the Server

```bash
# Set environment variables
export COGNITO_USER_POOL_ID=us-west-2_XXXXXXXXX
export COGNITO_APP_CLIENT_ID=XXXXXXXXXXXXXXXXXXXXXXXXXX

# Run the server
python examples/run_api_with_auth.py
```

### Python Code Example

```python
from daylib.workset_api import create_app
from daylib.workset_state_db import WorksetStateDB
from daylily_cognito.auth import CognitoAuth

# Initialize components
state_db = WorksetStateDB("daylily-worksets", "us-west-2")

# Initialize Cognito authentication
cognito_auth = CognitoAuth(
    region="us-west-2",
    user_pool_id="us-west-2_XXXXXXXXX",
    app_client_id="XXXXXXXXXXXXXXXXXXXXXXXXXX",
)

# Create app WITH authentication
app = create_app(
    state_db=state_db,
    cognito_auth=cognito_auth,
    enable_auth=True,  # Enable authentication
)

# Run with uvicorn
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8914)
```

### Getting a JWT Token

```bash
# Authenticate via daylily-cognito library and print an ID token
python - <<'PY'
import os
from daylily_cognito.auth import CognitoAuth

auth = CognitoAuth(
    region=os.environ["COGNITO_REGION"],
    user_pool_id=os.environ["COGNITO_USER_POOL_ID"],
    app_client_id=os.environ.get("COGNITO_CLIENT_ID") or os.environ["COGNITO_APP_CLIENT_ID"],
)
tokens = auth.authenticate("test@example.com", "YourPassword123!")
print(tokens["id_token"])
PY
```

Save the token for API requests.

### Testing With Authentication

```bash
# Set token variable
TOKEN="eyJraWQiOiJ..."

# Make authenticated requests
curl http://localhost:8914/worksets \
  -H "Authorization: Bearer $TOKEN"

curl -X POST http://localhost:8914/customers \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"customer_name": "Test", "email": "test@example.com"}'
```

## Configuration Comparison

| Feature | Without Auth | With Auth |
|---------|-------------|-----------|
| Installation | `pip install -e .` | `pip install -e ".[auth]"` |
| Dependencies | Base only | Base + python-jose |
| AWS Cognito | Not required | Required |
| JWT Tokens | Not required | Required |
| User Management | Not available | Available |
| Multi-tenant | Basic | Full support |
| Production Ready | Internal only | Yes |

## Security Considerations

### Without Authentication

**Pros:**
- Simple setup
- No external dependencies
- Fast development

**Cons:**
- No user tracking
- No access control
- Requires network-level security

**Best Practices:**
- Deploy behind VPN or firewall
- Use network security groups
- Enable CloudWatch logging
- Monitor API access

### With Authentication

**Pros:**
- User tracking and audit logs
- Fine-grained access control
- Multi-tenant isolation
- Production-ready security

**Cons:**
- More complex setup
- Additional AWS costs
- Token management required

**Best Practices:**
- Enable MFA for users
- Rotate credentials regularly
- Use HTTPS in production
- Monitor failed auth attempts
- Set token expiration appropriately

## Switching Between Modes

You can switch between authentication modes by changing the `enable_auth` parameter:

```python
# Development: No auth
app = create_app(state_db=state_db, enable_auth=False)

# Production: With auth
app = create_app(
    state_db=state_db,
    cognito_auth=cognito_auth,
    enable_auth=True,
)
```

## Troubleshooting

### "Incompatible 'jose' package found" Warning

**Problem:** Wrong `jose` package installed (need `python-jose`).

**Symptoms:**
```
SyntaxError: Missing parentheses in call to 'print'
```
or
```
Incompatible 'jose' package found. Please uninstall it and install 'python-jose' instead.
```

**Solution:**
```bash
# Uninstall the incompatible jose package
pip uninstall jose

# Install the correct python-jose package
pip install 'python-jose[cryptography]'
```

**Note:** The API will work without authentication even with the incompatible `jose` package installed. The warning is only relevant if you want to enable authentication.

### "No module named 'jose'" Error

**Problem:** Authentication enabled but python-jose not installed.

**Solution:**
```bash
pip install 'python-jose[cryptography]'
# Or install with auth extras
pip install -e ".[auth]"
```

### "Authentication requires python-jose" Error

**Problem:** Trying to create CognitoAuth without python-jose.

**Solution:** Either install python-jose or disable authentication:
```python
# Option 1: Install python-jose
pip install 'python-jose[cryptography]'

# Option 2: Disable authentication
app = create_app(state_db=state_db, enable_auth=False)
```

### "Invalid token" Error

**Problem:** JWT token is expired or invalid.

**Solution:** Get a new token via the shared library:
```bash
python - <<'PY'
import os
from daylily_cognito.auth import CognitoAuth

auth = CognitoAuth(
    region=os.environ["COGNITO_REGION"],
    user_pool_id=os.environ["COGNITO_USER_POOL_ID"],
    app_client_id=os.environ.get("COGNITO_CLIENT_ID") or os.environ["COGNITO_APP_CLIENT_ID"],
)
print(auth.authenticate("user@example.com", "password")["id_token"])
PY
```

### "User pool not found" Error

**Problem:** Cognito User Pool ID is incorrect.

**Solution:** Verify pool/app status using daycog:
```bash
daycog status
daycog list-pools --profile <aws-profile> --region us-west-2
```

## Environment Variables

```bash
# AWS Configuration
export AWS_REGION=us-west-2
export AWS_PROFILE=my-profile

# DynamoDB Tables
export WORKSET_TABLE_NAME=daylily-worksets
export CUSTOMER_TABLE_NAME=daylily-customers

# Cognito Configuration (only if using authentication)
export COGNITO_USER_POOL_ID=us-west-2_XXXXXXXXX
export COGNITO_APP_CLIENT_ID=XXXXXXXXXXXXXXXXXXXXXXXXXX

# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8914
export ENABLE_AUTH=false  # or true
```

## See Also

- [Customer Portal Guide](CUSTOMER_PORTAL.md)
- [Quick Reference](QUICK_REFERENCE.md)
