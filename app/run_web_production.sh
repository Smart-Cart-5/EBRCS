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

# SessionManager is in-memory (process-local). Keep single worker by default.
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"
if ! [[ "$UVICORN_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "⚠️  UVICORN_WORKERS 값이 올바르지 않아 1로 설정합니다: $UVICORN_WORKERS"
    UVICORN_WORKERS="1"
fi
if [ "$UVICORN_WORKERS" -gt 1 ]; then
    echo "⚠️  UVICORN_WORKERS=$UVICORN_WORKERS (세션/웹소켓은 인메모리라 멀티 워커에서 불안정할 수 있음)"
fi

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
    --workers "$UVICORN_WORKERS" \
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

# 준비 상태 확인 (모델 로딩으로 backend startup이 오래 걸릴 수 있음)
echo "⏳ 서비스 준비 상태 확인 중..."
BACKEND_READY="false"
FRONTEND_READY="false"

for _ in $(seq 1 180); do
    if ! ps -p $BACKEND_PID >/dev/null 2>&1; then
        break
    fi
    if curl -sS --max-time 2 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
        BACKEND_READY="true"
        break
    fi
    sleep 1
done

for _ in $(seq 1 30); do
    if ! ps -p $FRONTEND_PID >/dev/null 2>&1; then
        break
    fi
    if curl -sS --max-time 2 http://127.0.0.1:5173 >/dev/null 2>&1; then
        FRONTEND_READY="true"
        break
    fi
    sleep 1
done

PUBLIC_IP="$(curl -s ifconfig.me || true)"
if [ -z "$PUBLIC_IP" ]; then
    PUBLIC_IP="YOUR_EC2_IP"
fi

# HTTPS 프록시 상태 확인 (Nginx)
HTTPS_READY="false"
if command -v curl >/dev/null 2>&1; then
    if curl -k -s --max-time 3 https://127.0.0.1/ >/dev/null 2>&1; then
        HTTPS_READY="true"
    fi
fi

# 인증서 호스트 불일치 안내 (EC2 재시작으로 공인 IP가 바뀐 경우)
CERT_HINT=""
if [ -f /etc/nginx/ssl/ebrcs.crt ] && [ "$PUBLIC_IP" != "YOUR_EC2_IP" ]; then
    CERT_INFO="$(openssl x509 -in /etc/nginx/ssl/ebrcs.crt -noout -subject -ext subjectAltName 2>/dev/null || true)"
    if ! echo "$CERT_INFO" | grep -q "$PUBLIC_IP"; then
        CERT_HINT="⚠️  현재 SSL 인증서와 공인 IP가 다를 수 있습니다. (권장: sudo ./setup_https.sh ${PUBLIC_IP})"
    fi
fi

# 상태 확인
echo ""
echo "================================"
if ps -p $BACKEND_PID > /dev/null && ps -p $FRONTEND_PID > /dev/null && [ "$BACKEND_READY" = "true" ] && [ "$FRONTEND_READY" = "true" ]; then
    echo "✅ 웹앱 실행 성공!"
    echo ""
    echo "🌐 접속 주소 (카메라 사용: HTTPS 권장):"
    echo "  웹앱(HTTPS): https://${PUBLIC_IP}"
    echo "  API(HTTPS): https://${PUBLIC_IP}/api/health"
    echo ""
    if [ "$HTTPS_READY" != "true" ]; then
        echo "⚠️  현재 HTTPS 프록시(Nginx) 응답이 없습니다."
        echo "   sudo ./setup_https.sh ${PUBLIC_IP}"
        echo ""
    fi
    if [ -n "$CERT_HINT" ]; then
        echo "$CERT_HINT"
        echo ""
    fi
    echo "🌐 직접 포트 접속 (디버깅용, 카메라 비권장):"
    echo "  Frontend: http://${PUBLIC_IP}:5173"
    echo "  Backend:  http://${PUBLIC_IP}:8000/api/health"
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
    echo "  Backend ready: $BACKEND_READY"
    echo "  Frontend ready: $FRONTEND_READY"
    echo "  cat app/logs/backend.log"
    echo "  cat app/logs/frontend.log"
    exit 1
fi
