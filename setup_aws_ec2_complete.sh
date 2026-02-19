#!/bin/bash
# EBRCS 완전 자동 AWS EC2 배포 스크립트

set -e
MIN_NODE_VERSION="20.19.0"

version_lt() {
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" != "$1" ]
}

echo "🚀 EBRCS AWS EC2 완전 자동 배포"
echo "================================"
echo ""

# 1. 시스템 업데이트
echo "📦 시스템 업데이트 중..."
sudo apt-get update 2>/dev/null || true
sudo apt-get upgrade -y

# 2. Python 설치 확인 (Ubuntu 24.04는 Python 3.12가 기본)
echo "🐍 Python 확인 중..."
sudo apt-get install -y python3 python3-venv python3-pip curl ca-certificates lsof
python3 --version

# 3. 기존 Node.js 제거
echo "📗 기존 Node.js 제거 중..."
sudo apt-get remove --purge nodejs -y 2>/dev/null || true

# 4. nvm으로 Node.js 20 설치
echo "📗 Node.js 20 설치 중 (nvm)..."
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install 20
nvm use 20
nvm alias default 20
NODE_VERSION="$(node -v | sed 's/^v//')"
if version_lt "$MIN_NODE_VERSION" "$NODE_VERSION"; then
    echo "❌ Node.js 버전이 낮습니다. 현재: v${NODE_VERSION}, 필요: v${MIN_NODE_VERSION}+"
    exit 1
fi
echo "✓ Node.js v${NODE_VERSION}"

# 5. Git 설치
echo "📚 Git 설치 중..."
sudo apt-get install -y git

# 6. 시스템 패키지 (OpenCV 의존성)
echo "🔧 시스템 패키지 설치 중..."
sudo apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    pkg-config

# 7. 저장소 클론
echo "📥 GitHub 저장소 클론 중..."
if [ -d "ebrcs_streaming" ]; then
    echo "⚠️  ebrcs_streaming 폴더가 이미 존재합니다. 건너뜁니다."
else
    git clone https://github.com/Smart-Cart-5/EBRCS.git ebrcs_streaming
fi
cd ebrcs_streaming

# 8. 웹앱 의존성 설정
echo "🔨 웹앱 환경 설정 중..."
cd app
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
./setup_venv.sh
cd ..

# 12. .env 파일 생성
echo "⚙️  환경 변수 설정 중..."
if [ ! -f .env ]; then
    cat > .env <<EOF
HF_TOKEN=your_huggingface_token_here
HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here
EOF
    echo "⚠️  .env 파일이 생성되었습니다. HF_TOKEN을 나중에 설정하세요!"
fi

# 13. data 폴더 확인 및 심볼릭 링크
echo "📁 data 폴더 설정 중..."
mkdir -p data
if [ ! -L "app/data" ]; then
    ln -s ~/ebrcs_streaming/data ~/ebrcs_streaming/app/data
    echo "✓ 심볼릭 링크 생성 완료"
fi

# 14. 실행 스크립트 권한 설정
echo "🔧 실행 스크립트 권한 설정 중..."
chmod +x app/run_web.sh
chmod +x app/run_web_production.sh
chmod +x app/stop_web.sh
chmod +x streamlit/run.sh 2>/dev/null || true
chmod +x streamlit/run_mobile.sh 2>/dev/null || true

# 15. Nginx 설치 및 설정
echo "🌐 Nginx 설치 및 설정 중..."
sudo apt-get install -y nginx

# Nginx 설정 파일 생성
sudo tee /etc/nginx/sites-available/ebrcs > /dev/null << 'NGINX_EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
NGINX_EOF

# Nginx 설정 활성화
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -sf /etc/nginx/sites-available/ebrcs /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx

echo ""
echo "================================"
echo "✅ 설정 완료!"
echo ""
echo "📝 다음 단계:"
echo "  1. data 폴더에 파일 업로드:"
echo "     로컬에서: scp -i <your-key>.pem -r data/* ubuntu@<YOUR_EC2_IP>:~/ebrcs_streaming/data/"
echo ""
echo "  2. .env 파일 수정 (HF_TOKEN 설정):"
echo "     nano .env"
echo ""
echo "  3. 웹앱 실행:"
echo "     cd app && ./run_web_production.sh"
echo ""
echo "🌐 접속 주소:"
echo "  http://$(curl -s ifconfig.me) (포트 80, Nginx 리버스 프록시)"
echo ""
