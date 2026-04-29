# Google OAuth Default For Ursa

Ursa uses `daylily-auth-cognito==2.1.5` for browser/session helpers and Hosted UI token exchange. Shared user-pool and app-client lifecycle belongs to `daycog`; this repo's helper script only applies Ursa-specific Google OAuth defaults on top of that lifecycle.

## Helper Script

From an activated Ursa environment where `daycog` is on `PATH`:

```bash
source ./activate <deploy-name>
./scripts/setup_cognito_google_default.sh
```

The helper reads these defaults from the script:

- `POOL_NAME=daylily-ursa-users`
- `CLIENT_NAME=ursa`
- `AWS_PROFILE=lsmc`
- `AWS_REGION=us-west-2`
- `GOOGLE_CLIENT_JSON=$HOME/.config/google_oauth/client_secret_2_95843944781-d1831sfs0ic2ggmp6t404b958v1nqn40.apps.googleusercontent.com.json`

Override values explicitly when the deployment differs:

```bash
AWS_PROFILE=lsmc AWS_REGION=us-west-2 \
POOL_NAME=daylily-ursa-users CLIENT_NAME=ursa PORT=8913 \
GOOGLE_CLIENT_JSON=/path/to/client_secret.json \
./scripts/setup_cognito_google_default.sh
```

## Required Ursa Config

Authenticated GUI startup reads Cognito settings from the Ursa YAML config, not from ad hoc shell-only values. Populate these fields through the normal Ursa config path:

- `cognito_user_pool_id`
- `cognito_app_client_id`
- `cognito_region`
- `cognito_domain`
- `cognito_callback_url`
- `cognito_logout_url`

Use `daycog` for Cognito pool/client lifecycle and `ursa config ...` for Ursa-owned configuration.

## Google Redirect URI

The Google Cloud OAuth client must include the Cognito Hosted UI IdP response URI for the configured domain:

```text
https://<cognito-domain>/oauth2/idpresponse
```
