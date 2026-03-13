#!/usr/bin/env bash
# One-shot full setup: local Docker MySQL + venv + dependencies + DB bootstrap + seed import.
#
# Usage:
#   cd app && ./setup_all.sh
#
# DB 모드:
#   기본값 = 로컬 Docker MySQL 자동 기동 + 시드 import
#   공용 EC2 DB를 쓰려면 .env의 DATABASE_URL을 EC2 주소로 설정하고
#   --skip-db 옵션을 사용하세요.
#
# Options:
#   --skip-db    Docker MySQL 기동 및 시드 import 건너뜀 (EC2 공용 DB 사용 시)

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$APP_DIR")"
SKIP_DB="false"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --skip-db) SKIP_DB="true"; shift ;;
        *) echo "❌ Unknown option: $1"; exit 1 ;;
    esac
done

# ── Step 0: Docker Desktop 확인 및 설치 ──────────────────────────────────────
echo "▶ Step 0: Checking Docker..."
if ! command -v docker >/dev/null 2>&1; then
    echo "  Docker not found."
    if command -v brew >/dev/null 2>&1; then
        echo "  Installing Docker Desktop via Homebrew..."
        brew install --cask docker
    else
        echo "  ❌ Homebrew not found. Please install Docker Desktop manually:"
        echo "     https://www.docker.com/products/docker-desktop/"
        exit 1
    fi
fi

# Docker Desktop이 설치됐지만 데몬이 안 켜져 있으면 자동 시작
if ! docker info >/dev/null 2>&1; then
    echo "  Docker daemon not running. Starting Docker Desktop..."
    open -a Docker 2>/dev/null || true
    echo "  Waiting for Docker daemon (up to 60s)..."
    for i in $(seq 1 30); do
        if docker info >/dev/null 2>&1; then
            echo "  ✅ Docker daemon is ready."
            break
        fi
        sleep 2
        if [ "$i" -eq 30 ]; then
            echo "  ❌ Docker daemon did not start in time."
            echo "     Please start Docker Desktop manually and re-run."
            exit 1
        fi
    done
else
    echo "  ✅ Docker is running."
fi
echo ""

# ── Step 1: 로컬 Docker MySQL 기동 + 스키마 + 시드 ────────────────────────────
if [ "$SKIP_DB" = "false" ]; then
    echo "▶ Step 1/3: Setting up local Docker MySQL..."
    "$PROJECT_ROOT/db/setup_local_db.sh"
    echo ""
else
    echo "▶ Step 1/3: Skipped (--skip-db). Using DATABASE_URL from .env."
    echo ""
fi

# ── Step 2: Python venv + 의존성 설치 ────────────────────────────────────────
echo "▶ Step 2/3: Setting up Python venv and Node dependencies..."
cd "$APP_DIR"
./setup_venv.sh
echo ""

# ── Step 3: DB 스키마 최종 확인 ──────────────────────────────────────────────
echo "▶ Step 3/3: Verifying DB schema..."
./setup_db.sh
echo ""

echo "================================================================"
echo "  ✅ All setup steps completed."
echo ""
echo "  Next step:"
echo "    ./run_web.sh               # local dev"
echo "    ./run_web_production.sh    # production mode"
echo "================================================================"
