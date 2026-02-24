#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

get_env_value() {
    local key="$1"
    local value
    value="$(awk -v k="$key" -F= '$1==k {v=$2} END {print v}' "$ENV_FILE")"
    value="${value%%#*}"
    value="$(printf '%s' "$value" | sed 's/[[:space:]]*$//')"
    printf '%s' "$value"
}

expect_status() {
    local url="$1"
    local expected="$2"
    local auth="${3:-}"
    local code

    if [[ -n "$auth" ]]; then
        code="$(curl -k -s -o /dev/null -w '%{http_code}' -u "$auth" "$url")"
    else
        code="$(curl -k -s -o /dev/null -w '%{http_code}' "$url")"
    fi

    if [[ "$code" != "$expected" ]]; then
        echo "FAIL status $url expected=$expected actual=$code" >&2
        exit 1
    fi
}

echo "[runtime-smoke] health check"
curl -fsS "http://127.0.0.1:8000/api/health" >/dev/null

echo "[runtime-smoke] db-viewer auth gate"
expect_status "https://127.0.0.1/api/db-viewer" "401"

echo "[runtime-smoke] adminer auth gate"
expect_status "https://127.0.0.1/adminer/" "401"

adminer_user="$(get_env_value ADMINER_BASIC_USER)"
adminer_password="$(get_env_value ADMINER_BASIC_PASSWORD)"
if [[ -z "$adminer_user" || -z "$adminer_password" ]]; then
    echo "FAIL missing ADMINER_BASIC_USER/PASSWORD in .env" >&2
    exit 1
fi

echo "[runtime-smoke] adminer login page reachable with basic auth"
expect_status "https://127.0.0.1/adminer/" "200" "${adminer_user}:${adminer_password}"

echo "[runtime-smoke] php mysql extensions loaded"
php -m | grep -qi '^mysqli$'
php -m | grep -qi '^pdo_mysql$'

echo "PASS runtime_smoke.sh"
