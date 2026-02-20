#!/usr/bin/env bash
# Bootstrap/check DB schema used by EBRCS web app.
# Usage:
#   ./setup_db.sh          # create missing tables
#   ./setup_db.sh --check  # check only

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$APP_DIR")"

if [ ! -d "$APP_DIR/backend/.venv" ]; then
    echo "❌ Backend virtual environment not found. Run ./setup_venv.sh first."
    exit 1
fi

source "$APP_DIR/backend/.venv/bin/activate"
export PYTHONPATH="$APP_DIR:$PROJECT_ROOT"

python -m backend.db_bootstrap "$@"
