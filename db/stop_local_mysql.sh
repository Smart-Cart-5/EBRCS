#!/usr/bin/env bash
# Stop local MySQL Docker container used for EBRCS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.mysql.yml"
PURGE="false"

if [ "${1:-}" = "--purge" ]; then
    PURGE="true"
fi

compose_cmd() {
    if docker compose version >/dev/null 2>&1; then
        echo "docker compose"
        return
    fi
    if command -v docker-compose >/dev/null 2>&1; then
        echo "docker-compose"
        return
    fi
    echo ""
}

COMPOSE="$(compose_cmd)"
if [ -z "$COMPOSE" ]; then
    echo "❌ docker compose/docker-compose not found."
    exit 1
fi

if [ "$PURGE" = "true" ]; then
    echo "Stopping local MySQL and removing volume..."
    # shellcheck disable=SC2086
    $COMPOSE -f "$COMPOSE_FILE" down -v
else
    echo "Stopping local MySQL (volume kept)..."
    # shellcheck disable=SC2086
    $COMPOSE -f "$COMPOSE_FILE" down
fi

echo "✅ Done."
