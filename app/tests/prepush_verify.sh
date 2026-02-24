#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_DIR="$ROOT_DIR/app"
FRONTEND_DIR="$APP_DIR/frontend"

echo "[1/6] shell syntax checks"
bash -n "$APP_DIR/setup_adminer.sh"
bash -n "$APP_DIR/run_web_production.sh"
bash -n "$APP_DIR/setup_systemd.sh"

echo "[2/6] setup_adminer env parser test"
bash "$APP_DIR/tests/test_setup_adminer_env.sh"

echo "[3/6] frontend build (typecheck + bundle)"
(
    cd "$FRONTEND_DIR"
    npm run build
)

echo "[4/6] frontend unit tests"
(
    cd "$FRONTEND_DIR"
    npm test
)

echo "[5/6] runtime smoke tests"
bash "$APP_DIR/tests/runtime_smoke.sh"

echo "[6/6] git status snapshot"
git -C "$ROOT_DIR" status --short

echo "PASS prepush_verify.sh"
