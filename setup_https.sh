#!/bin/bash
# HTTPS 설정 자동화 스크립트
# Usage: sudo ./setup_https.sh [domain_or_ip]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOMAIN="${1:-_}"  # Default to _ (catch-all)

echo "🔒 EBRCS HTTPS 설정"
echo "=================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ 이 스크립트는 sudo로 실행해야 합니다."
    echo "Usage: sudo ./setup_https.sh [domain_or_ip]"
    exit 1
fi

# 1. Install Nginx if not installed
if ! command -v nginx &> /dev/null; then
    echo "📦 Nginx 설치 중..."
    apt-get update
    apt-get install -y nginx
fi

# 2. Create SSL directory
echo "📁 SSL 디렉토리 생성..."
mkdir -p /etc/nginx/ssl

# 3. Generate self-signed certificate
if [ ! -f /etc/nginx/ssl/ebrcs.crt ] || [ ! -f /etc/nginx/ssl/ebrcs.key ]; then
    echo "🔐 자체 서명 SSL 인증서 생성 중..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/nginx/ssl/ebrcs.key \
        -out /etc/nginx/ssl/ebrcs.crt \
        -subj "/C=US/ST=State/L=City/O=EBRCS/CN=${DOMAIN}"
    echo "  ✓ 인증서 생성 완료"
else
    echo "  ℹ️  기존 SSL 인증서 사용"
fi

# 4. Copy Nginx configuration
echo "⚙️  Nginx 설정 파일 복사 중..."
cp "$SCRIPT_DIR/nginx/ebrcs.conf" /etc/nginx/sites-available/ebrcs

# 5. Enable site
if [ ! -L /etc/nginx/sites-enabled/ebrcs ]; then
    ln -s /etc/nginx/sites-available/ebrcs /etc/nginx/sites-enabled/ebrcs
    echo "  ✓ Site 활성화 완료"
fi

# 6. Disable default site if exists
if [ -L /etc/nginx/sites-enabled/default ]; then
    rm /etc/nginx/sites-enabled/default
    echo "  ✓ 기본 site 비활성화"
fi

# 7. Test Nginx configuration
echo "🧪 Nginx 설정 테스트 중..."
nginx -t

# 8. Restart Nginx
echo "🔄 Nginx 재시작 중..."
systemctl restart nginx
systemctl enable nginx

echo ""
echo "=================="
echo "✅ HTTPS 설정 완료!"
echo ""
echo "📝 접속 방법:"
echo "  https://your-server-ip"
echo "  (브라우저 경고가 나오면 '고급 > 계속 진행' 클릭)"
echo ""
echo "🔧 Let's Encrypt 인증서로 업그레이드하려면:"
echo "  sudo snap install --classic certbot"
echo "  sudo certbot --nginx -d your-domain.com"
echo ""
