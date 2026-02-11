#!/usr/bin/env bash
# Run backend (FastAPI) and frontend (Vite dev) concurrently for local development.
# Usage: ./run_web.sh

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$APP_DIR")"

# Load .env if exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment from .env..."
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Activate backend virtual environment
if [ -d "$APP_DIR/backend/.venv" ]; then
    echo "Activating backend virtual environment..."
    source "$APP_DIR/backend/.venv/bin/activate"
else
    echo "⚠️  Backend virtual environment not found. Run ./setup_venv.sh first."
    exit 1
fi

# Fix OpenMP duplicate library issue on macOS
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== EBRCS Web App (Local Dev) ==="
echo "Backend : http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""

# Start backend (PYTHONPATH includes app dir for backend module, project root for checkout_core)
cd "$APP_DIR"
PYTHONPATH="$APP_DIR:$PROJECT_ROOT" uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start frontend dev server
cd "$APP_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM EXIT

wait
