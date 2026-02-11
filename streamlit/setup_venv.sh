#!/bin/bash
# Streamlit 가상환경 설정 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 Streamlit 가상환경 설정 시작..."

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

echo "✅ Streamlit 가상환경 설정 완료!"
echo ""
echo "사용법:"
echo "  source streamlit/.venv/bin/activate"
echo "  cd streamlit"
echo "  ./run.sh"
