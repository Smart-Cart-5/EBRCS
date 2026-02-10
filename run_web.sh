#!/usr/bin/env bash
# Run backend (FastAPI) and frontend (Vite dev) concurrently for local development.
# Usage: ./run_web.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if exists
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "Loading environment from .env..."
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# Fix OpenMP duplicate library issue on macOS
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== EBRCS Web App (Local Dev) ==="
echo "Backend : http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""

# Start backend
PYTHONPATH="$PROJECT_DIR" uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start frontend dev server
cd "$PROJECT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM EXIT

wait
