#!/usr/bin/env bash
set -euo pipefail

# Default Google OAuth client JSON for Ursa Cognito setup.
DEFAULT_GOOGLE_CLIENT_JSON="${HOME}/.config/google_oauth/client_secret_2_95843944781-d1831sfs0ic2ggmp6t404b958v1nqn40.apps.googleusercontent.com.json"

POOL_NAME="${POOL_NAME:-daylily-ursa-users}"
CLIENT_NAME="${CLIENT_NAME:-ursa}"
PORT="${PORT:-8914}"
AWS_PROFILE_VALUE="${AWS_PROFILE:-lsmc}"
AWS_REGION_VALUE="${AWS_REGION:-us-west-2}"
GOOGLE_CLIENT_JSON="${GOOGLE_CLIENT_JSON:-$DEFAULT_GOOGLE_CLIENT_JSON}"

if ! command -v daycog >/dev/null 2>&1; then
  echo "daycog is not installed or not on PATH"
  echo "Run: source ../daylily-auth-cognito/activate"
  exit 1
fi

if [[ ! -f "$GOOGLE_CLIENT_JSON" ]]; then
  echo "Google OAuth client JSON not found:"
  echo "  $GOOGLE_CLIENT_JSON"
  echo "Override with: GOOGLE_CLIENT_JSON=/path/to/client_secret.json $0"
  exit 1
fi

echo "Configuring Cognito + Google OAuth for Ursa..."
echo "  pool:    $POOL_NAME"
echo "  client:  $CLIENT_NAME"
echo "  port:    $PORT"
echo "  profile: $AWS_PROFILE_VALUE"
echo "  region:  $AWS_REGION_VALUE"
echo "  google:  $GOOGLE_CLIENT_JSON"

AWS_PROFILE="$AWS_PROFILE_VALUE" AWS_REGION="$AWS_REGION_VALUE" \
daycog setup-with-google \
  --name "$POOL_NAME" \
  --client-name "$CLIENT_NAME" \
  --profile "$AWS_PROFILE_VALUE" \
  --region "$AWS_REGION_VALUE" \
  --port "$PORT" \
  --google-client-json "$GOOGLE_CLIENT_JSON"

echo
echo "Done. Recommended next step:"
echo "  cp ~/.config/daycog/${POOL_NAME}.${AWS_REGION_VALUE}.${CLIENT_NAME}.env ~/.config/daycog/default.env"
