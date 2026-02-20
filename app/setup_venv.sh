#!/bin/bash
# 웹앱 백엔드 가상환경 설정 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MIN_NODE_VERSION="20.19.0"

version_lt() {
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" != "$1" ]
}

echo "🔧 웹앱 백엔드 가상환경 설정 시작..."

if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "❌ python3가 없습니다. Python 3.11+를 설치하세요."
    exit 1
fi

if ! command -v node >/dev/null 2>&1; then
    echo "❌ Node.js가 없습니다. Node.js ${MIN_NODE_VERSION}+를 설치하세요."
    exit 1
fi

NODE_VERSION="$(node -v | sed 's/^v//')"
if version_lt "$MIN_NODE_VERSION" "$NODE_VERSION"; then
    echo "❌ Node.js 버전이 낮습니다. 현재: v${NODE_VERSION}, 필요: v${MIN_NODE_VERSION}+"
    exit 1
fi

# backend 디렉토리로 이동
cd "$SCRIPT_DIR/backend"

# 가상환경 생성
if [ ! -d ".venv" ]; then
    echo "📦 가상환경 생성 중..."
    "$PYTHON_BIN" -m venv .venv
else
    echo "✓ 가상환경이 이미 존재합니다."
fi

# 가상환경 활성화
echo "🔌 가상환경 활성화 중..."
source .venv/bin/activate

# 의존성 설치
echo "📥 패키지 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

# Frontend npm 설치
echo ""
echo "📦 Frontend 패키지 설치 중..."
cd "$SCRIPT_DIR/frontend"
if [ -f package-lock.json ]; then
    npm ci
else
    npm install
fi

echo ""
echo "✅ 웹앱 환경 설정 완료!"
echo ""
echo "사용법:"
echo "  cd app"
echo "  ./setup_db.sh    # DB 스키마 초기화/검증"
echo "  ./run_web.sh"
