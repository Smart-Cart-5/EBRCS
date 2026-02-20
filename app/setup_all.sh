#!/usr/bin/env bash
# One-shot web app setup: venv + dependencies + DB bootstrap.

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$APP_DIR"
./setup_venv.sh
./setup_db.sh

echo ""
echo "✅ All setup steps completed."
echo "Next:"
echo "  ./run_web.sh               # local dev"
echo "  ./run_web_production.sh    # production mode"
