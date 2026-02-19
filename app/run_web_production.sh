#!/bin/bash
# EBRCS 웹앱 프로덕션 실행 스크립트 (AWS EC2용)

set -e
MIN_NODE_VERSION="20.19.0"

version_lt() {
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" != "$1" ]
}

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

if ! command -v node >/dev/null 2>&1; then
    echo "❌ Node.js를 찾을 수 없습니다. setup_venv.sh를 먼저 실행하세요."
    exit 1
fi
if ! command -v lsof >/dev/null 2>&1; then
    echo "❌ lsof를 찾을 수 없습니다. (Ubuntu: sudo apt-get install -y lsof)"
    exit 1
fi
NODE_VERSION="$(node -v | sed 's/^v//')"
if version_lt "$MIN_NODE_VERSION" "$NODE_VERSION"; then
    echo "❌ Node.js 버전이 낮습니다. 현재: v${NODE_VERSION}, 필요: v${MIN_NODE_VERSION}+"
    exit 1
fi

# 환경 변수 확인
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "❌ .env 파일이 없습니다."
    exit 1
fi

# .env 로드
set -a
# shellcheck disable=SC1090
source "$PROJECT_ROOT/.env"
set +a

# 로그 디렉토리 생성
mkdir -p "$APP_DIR/logs"

# 기존 프로세스 종료
echo "🔄 기존 프로세스 종료 중..."

# PID 파일 기반 종료
if [ -f "$APP_DIR/logs/backend.pid" ]; then
    BACKEND_PID=$(cat "$APP_DIR/logs/backend.pid")
    kill $BACKEND_PID 2>/dev/null || true
    echo "  - Backend PID $BACKEND_PID 종료 시도"
fi

if [ -f "$APP_DIR/logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat "$APP_DIR/logs/frontend.pid")
    kill $FRONTEND_PID 2>/dev/null || true
    echo "  - Frontend PID $FRONTEND_PID 종료 시도"
fi

# pkill로 남은 프로세스 종료
pkill -f "uvicorn backend.main:app" || true
pkill -f "vite preview" || true
sleep 2

# 포트 8000을 사용 중인 프로세스 강제 종료
PORT_PID=$(lsof -ti:8000 || true)
if [ -n "$PORT_PID" ]; then
    echo "  - 포트 8000 점유 프로세스 (PID: $PORT_PID) 강제 종료"
    kill -9 $PORT_PID || true
    sleep 1
fi

# 포트 5173을 사용 중인 프로세스 강제 종료
PORT_PID=$(lsof -ti:5173 || true)
if [ -n "$PORT_PID" ]; then
    echo "  - 포트 5173 점유 프로세스 (PID: $PORT_PID) 강제 종료"
    kill -9 $PORT_PID || true
    sleep 1
fi

# 최종 확인
if lsof -i:8000 >/dev/null 2>&1; then
    echo "❌ 포트 8000을 해제할 수 없습니다. 수동으로 확인하세요:"
    echo "   lsof -i:8000"
    exit 1
fi

echo "  ✓ 기존 프로세스 종료 완료"

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
PUBLIC_IP="$(curl -s ifconfig.me || true)"
if [ -z "$PUBLIC_IP" ]; then
    PUBLIC_IP="YOUR_EC2_IP"
fi

# 상태 확인
echo ""
echo "================================"
if ps -p $BACKEND_PID > /dev/null && ps -p $FRONTEND_PID > /dev/null; then
    echo "✅ 웹앱 실행 성공!"
    echo ""
    echo "🌐 접속 주소 (직접 접속):"
    echo "  웹앱: http://${PUBLIC_IP}:5173"
    echo "  API 테스트: http://${PUBLIC_IP}:8000/api/health"
    echo ""
    echo "🌐 Nginx 리버스 프록시를 설정한 경우:"
    echo "  웹앱: http://${PUBLIC_IP}"
    echo "  API 테스트: http://${PUBLIC_IP}/api/health"
    echo ""
    echo "📝 내부 서비스 (localhost only):"
    echo "  Backend: http://localhost:8000"
    echo "  Frontend: http://localhost:5173"
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
