# Ursa Playwright Auth E2E

These tests cover the authenticated Ursa GUI login/logout browser flow. They require a running Ursa server, a reachable Cognito Hosted UI configuration, Playwright browser binaries, and a real test user password.

Run from an activated Ursa environment:

```bash
source ./activate <deploy-name>
E2E_USER_PASSWORD=... pytest tests/e2e/test_auth_e2e.py -m e2e
```

Defaults:

- `E2E_USER_EMAIL=johnm+test@lsmc.com`
- `URSA_BASE_URL=https://localhost:18913`

Override `URSA_BASE_URL` when the server is listening elsewhere. These tests do not exercise staging jobs, analysis jobs, cluster operations, or Atlas return.
