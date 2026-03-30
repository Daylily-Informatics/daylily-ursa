## Ursa Playwright Auth E2E

Run:

```bash
E2E_USER_PASSWORD=... pytest tests/e2e/test_auth_e2e.py -m e2e
```

Defaults:
- `E2E_USER_EMAIL=johnm+test@lsmc.com`
- `URSA_BASE_URL=https://localhost:18913`

These tests cover only the Ursa GUI login/logout browser flow.
