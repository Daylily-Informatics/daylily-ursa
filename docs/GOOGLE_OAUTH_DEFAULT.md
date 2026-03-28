# Google OAuth Default For Ursa

Ursa defaults to Google-enabled Cognito setup using:

- OAuth client JSON:
  `~/.config/google_oauth/client_secret_2_95843944781-d1831sfs0ic2ggmp6t404b958v1nqn40.apps.googleusercontent.com.json`
- Cognito pool/client:
  `daylily-ursa-users` / `ursa`
- App port:
  `8914`

## One-command setup

```bash
source ../daylily-cognito/activate
./scripts/setup_cognito_google_default.sh
```

## Overriding defaults

```bash
AWS_PROFILE=lsmc AWS_REGION=us-west-2 \
POOL_NAME=daylily-ursa-users CLIENT_NAME=ursa PORT=8914 \
GOOGLE_CLIENT_JSON=/path/to/client_secret.json \
./scripts/setup_cognito_google_default.sh
```

## Required Google Redirect URI

In Google Cloud Console, the OAuth client must include this URI:

`https://daylily-ursa-5r8giqv5p.auth.us-west-2.amazoncognito.com/oauth2/idpresponse`
