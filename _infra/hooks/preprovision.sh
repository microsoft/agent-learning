#!/usr/bin/env sh
# =====================================================================
# Pre-provision hook (POSIX / sh)
# 1. Ensures AZURE_LOCATION has a sensible default.
# 2. Ensures an SSH public key exists for the VM administrator, generating
#    one under ~/.ssh if the azd env doesn't already carry AZURE_SSH_PUBLIC_KEY.
# =====================================================================
set -e

# 1) Default region.
LOCATION="${AZURE_LOCATION:-eastus2}"
azd env set AZURE_LOCATION "$LOCATION" >/dev/null
echo "AZURE_LOCATION=$LOCATION"

# 2) Ensure an SSH public key for the VM administrator.
EXISTING="$(azd env get-values | sed -n 's/^AZURE_SSH_PUBLIC_KEY="\{0,1\}\([^"]*\)"\{0,1\}$/\1/p')"
if [ -n "$EXISTING" ]; then
  echo 'AZURE_SSH_PUBLIC_KEY already set; skipping key generation.'
  exit 0
fi

ENV_NAME="${AZURE_ENV_NAME:-agent-learning}"
KEY_PATH="$HOME/.ssh/agent-learning-$ENV_NAME"
mkdir -p "$HOME/.ssh"

if [ ! -f "$KEY_PATH.pub" ]; then
  echo "Generating SSH key pair at $KEY_PATH ..."
  ssh-keygen -t rsa -b 4096 -f "$KEY_PATH" -N "" -C 'agent-learning-vm' >/dev/null
fi

PUB="$(cat "$KEY_PATH.pub")"
azd env set AZURE_SSH_PUBLIC_KEY "$PUB" >/dev/null
echo "SSH public key stored in azd env. Private key kept at: $KEY_PATH"
