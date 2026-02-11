#!/bin/bash
# 웹앱 백엔드 가상환경 설정 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔧 웹앱 백엔드 가상환경 설정 시작..."

# backend 디렉토리로 이동
cd "$SCRIPT_DIR/backend"

# 가상환경 생성
if [ ! -d ".venv" ]; then
    echo "📦 가상환경 생성 중..."
    python3 -m venv .venv
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
npm install

echo ""
echo "✅ 웹앱 환경 설정 완료!"
echo ""
echo "사용법:"
echo "  cd app"
echo "  ./run_web.sh"
