#!/usr/bin/env bash
# Start local MySQL via Docker and print DATABASE_URL example.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.mysql.yml"

MYSQL_PORT="${MYSQL_PORT:-3307}"
MYSQL_DATABASE="${MYSQL_DATABASE:-item_db}"
MYSQL_USER="${MYSQL_USER:-ebrcs_app}"
MYSQL_PASSWORD="${MYSQL_PASSWORD:-ebrcs_pass}"
MYSQL_ROOT_PASSWORD="${MYSQL_ROOT_PASSWORD:-root1234}"

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

if ! command -v docker >/dev/null 2>&1; then
    echo "❌ docker not found."
    exit 1
fi

export MYSQL_PORT MYSQL_DATABASE MYSQL_USER MYSQL_PASSWORD MYSQL_ROOT_PASSWORD

echo "Starting local MySQL container..."
# shellcheck disable=SC2086
$COMPOSE -f "$COMPOSE_FILE" up -d

echo "Waiting for MySQL readiness..."
for _ in $(seq 1 60); do
    # shellcheck disable=SC2086
    if MYSQL_PWD="$MYSQL_ROOT_PASSWORD" $COMPOSE -f "$COMPOSE_FILE" exec -T mysql \
        mysqladmin ping -h 127.0.0.1 -uroot --silent >/dev/null 2>&1; then
        echo "✅ Local MySQL is ready."
        echo ""
        echo "Use this in .env:"
        echo "DATABASE_URL=mysql+pymysql://$MYSQL_USER:$MYSQL_PASSWORD@127.0.0.1:$MYSQL_PORT/$MYSQL_DATABASE"
        exit 0
    fi
    sleep 2
done

echo "❌ MySQL container did not become ready in time."
echo "Check logs:"
echo "  $COMPOSE -f \"$COMPOSE_FILE\" logs mysql"
exit 1
