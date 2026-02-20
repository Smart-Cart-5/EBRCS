#!/bin/bash
# AWS EC2 Ubuntu에서 EBRCS 웹앱 설정 스크립트

set -e
MIN_NODE_VERSION="20.19.0"

version_lt() {
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" != "$1" ]
}

echo "🚀 EBRCS 웹앱 AWS EC2 설정"
echo "=========================="
echo ""

# 1. 시스템 업데이트
echo "📦 시스템 업데이트 중..."
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y curl ca-certificates lsof

# 2. Python 3.11 설치
echo "🐍 Python 3.11 설치 중..."
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# 3. Node.js 20 설치
echo "📗 Node.js 20 설치 중..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 4. Git 설치
echo "📚 Git 설치 중..."
sudo apt-get install -y git

# 5. 시스템 패키지 (OpenCV 의존성)
echo "🔧 시스템 패키지 설치 중..."
sudo apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    pkg-config

# 6. 저장소 클론
echo "📥 GitHub 저장소 클론 중..."
read -p "GitHub 저장소 URL 입력: " REPO_URL
if [ -d "ebrcs_streaming" ]; then
    echo "⚠️  ebrcs_streaming 폴더가 이미 존재합니다. 기존 폴더를 사용합니다."
else
    git clone "$REPO_URL" ebrcs_streaming
fi
cd ebrcs_streaming

NODE_VERSION="$(node -v | sed 's/^v//')"
if version_lt "$MIN_NODE_VERSION" "$NODE_VERSION"; then
    echo "❌ Node.js 버전이 낮습니다. 현재: v${NODE_VERSION}, 필요: v${MIN_NODE_VERSION}+"
    exit 1
fi
echo "✓ Node.js v${NODE_VERSION}"
python3.11 --version

# 7. 웹앱 환경 설정
echo "🔨 웹앱 환경 설정 중..."
cd app
./setup_venv.sh
cd ..

# 10. 환경 변수 설정
echo "⚙️  환경 변수 설정 중..."
if [ ! -f .env ]; then
    cat > .env <<EOF
HF_TOKEN=your_huggingface_token_here
HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here
DATABASE_URL=mysql+pymysql://<USER>:<PASSWORD>@<HOST>:3306/item_db
EOF
    echo "⚠️  .env 파일이 생성되었습니다. HF_TOKEN, DATABASE_URL을 설정하세요!"
fi

# 11. data 폴더 확인
if [ ! -d "data" ]; then
    echo "⚠️  data/ 폴더가 없습니다. 임베딩 파일을 업로드하세요."
    mkdir -p data
fi

# 12. 실행 스크립트 권한 설정
chmod +x app/run_web.sh
chmod +x app/run_web_production.sh
chmod +x app/stop_web.sh

echo ""
echo "=========================="
echo "✅ 설정 완료!"
echo ""
echo "📝 다음 단계:"
echo "  1. .env 파일 수정: nano .env"
echo "  2. HF_TOKEN 설정"
echo "  3. data/ 폴더에 임베딩 파일 업로드"
echo "  4. 웹앱 실행:"
echo "     - 개발: cd app && ./run_web.sh"
echo "     - 프로덕션: cd app && ./run_web_production.sh"
echo ""
echo "🌐 접속 방법:"
echo "  웹앱: http://YOUR_EC2_PUBLIC_IP:5173"
echo ""
