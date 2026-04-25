# Ursa Mounted TapDB Admin Status

## Current Behavior

Ursa mounts the TapDB admin ASGI app inside the Ursa FastAPI process when `ursa_tapdb_mount_enabled` is true. The default mount path is `/admin/tapdb`.

Mounted mode:

- loads `admin.main:app` lazily from the installed TapDB package
- requires explicit `tapdb_env` and `tapdb_config_path`
- forwards Ursa's TapDB env, config path, client ID, and namespace into the embedded app
- gates access with `X-API-Key` matching `ursa_internal_api_key`
- injects an embedded TapDB admin identity into the forwarded ASGI scope
- does not mutate TapDB admin auth environment variables
- fails application startup when enabled and TapDB admin import/configuration fails

Set `ursa_tapdb_mount_enabled: false` to skip importing the TapDB admin app.

## Settings

- `ursa_tapdb_mount_enabled`: enables or disables the mount
- `ursa_tapdb_mount_path`: mount path, default `/admin/tapdb`
- `ursa_internal_api_key`: API key required by the mounted gate
- `tapdb_env`: TapDB environment selector
- `tapdb_config_path`: explicit TapDB config path
- `tapdb_client_id`: TapDB client ID
- `tapdb_database_name`: TapDB namespace/database name

## Verification

Current tests cover:

- mounted route existence
- valid API key access
- missing or wrong API key denial
- no mutation of TapDB admin auth environment variables
- fail-fast startup when the TapDB admin app cannot be imported
- disabled mount skipping TapDB import
- explicit TapDB context forwarding

Run:

```bash
pytest tests/test_tapdb_mount.py -q
```
