#!/usr/bin/env bash
# Usage: source ./scripts/load_env.sh

set -euo pipefail

if [[ ! -f .env ]]; then
  echo ".env file not found" >&2
  return 1 2>/dev/null || exit 1
fi

set -a
source .env
set +a