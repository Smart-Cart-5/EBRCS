#!/bin/bash
# EBRCS 웹앱 프로덕션 실행 스크립트 (AWS EC2용)

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$APP_DIR")"

echo "🚀 EBRCS 웹앱 프로덕션 모드 실행"
echo "================================"
echo ""

# 가상환경 활성화 확인
if [ ! -d "$APP_DIR/backend/.venv" ]; then
    echo "❌ 가상환경이 없습니다. setup_venv.sh를 먼저 실행하세요."
    exit 1
fi

# 가상환경 활성화
source "$APP_DIR/backend/.venv/bin/activate"

# nvm 로드 (Node.js)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 환경 변수 확인
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "❌ .env 파일이 없습니다."
    exit 1
fi

# .env 로드
export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)

# 로그 디렉토리 생성
mkdir -p "$APP_DIR/logs"

# 기존 프로세스 종료
echo "🔄 기존 프로세스 종료 중..."
pkill -f "uvicorn backend.main:app" || true
pkill -f "vite" || true
sleep 2

# Frontend 빌드
echo "🔨 Frontend 빌드 중..."
cd "$APP_DIR/frontend"
npm run build
cd "$APP_DIR"

# Backend 실행 (프로덕션 모드)
echo "🚀 Backend 시작 중..."
cd "$APP_DIR"
export PYTHONPATH="$APP_DIR:$PROJECT_ROOT"
nohup uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    > "$APP_DIR/logs/backend.log" 2>&1 &

BACKEND_PID=$!
echo "  ✓ Backend PID: $BACKEND_PID"

# Frontend 실행 (Vite preview)
echo "🌐 Frontend 시작 중..."
cd "$APP_DIR/frontend"
nohup npx vite preview \
    --host 0.0.0.0 \
    --port 5173 \
    > "$APP_DIR/logs/frontend.log" 2>&1 &

FRONTEND_PID=$!
cd "$APP_DIR"
echo "  ✓ Frontend PID: $FRONTEND_PID"

# PID 저장
echo $BACKEND_PID > "$APP_DIR/logs/backend.pid"
echo $FRONTEND_PID > "$APP_DIR/logs/frontend.pid"

# 대기
sleep 3

# 상태 확인
echo ""
echo "================================"
if ps -p $BACKEND_PID > /dev/null && ps -p $FRONTEND_PID > /dev/null; then
    echo "✅ 웹앱 실행 성공!"
    echo ""
    echo "🌐 접속 주소:"
    echo "  Frontend: http://$(curl -s ifconfig.me):5173"
    echo "  Backend API: http://$(curl -s ifconfig.me):8000/api/health"
    echo ""
    echo "📊 로그 확인:"
    echo "  Backend: tail -f app/logs/backend.log"
    echo "  Frontend: tail -f app/logs/frontend.log"
    echo ""
    echo "🛑 종료 방법:"
    echo "  cd app && ./stop_web.sh"
else
    echo "❌ 실행 실패. 로그를 확인하세요:"
    echo "  cat app/logs/backend.log"
    echo "  cat app/logs/frontend.log"
    exit 1
fi
