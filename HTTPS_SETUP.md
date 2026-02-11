# 🔒 HTTPS 설정 가이드

EBRCS 웹앱을 HTTPS로 설정하여 외부에서 카메라를 사용할 수 있도록 하는 가이드입니다.

## 왜 HTTPS가 필요한가?

브라우저의 보안 정책상 `getUserMedia()` (카메라 접근)는 다음 환경에서만 작동합니다:
- ✅ `localhost` 또는 `127.0.0.1`
- ✅ **HTTPS** 연결

HTTP로 외부 접속하면 카메라를 사용할 수 없습니다.

---

## 자동 설정 (권장)

### 1. 자체 서명 인증서 (테스트/개발용)

```bash
cd /path/to/ebrcs_streaming
sudo ./setup_https.sh
```

**장점:**
- 5분 안에 설정 완료
- 도메인 불필요

**단점:**
- 브라우저 경고 발생 (하지만 "고급 > 계속 진행"으로 접속 가능)

### 2. Let's Encrypt 인증서 (프로덕션용)

도메인이 있는 경우:

```bash
# 1. 도메인을 서버 IP에 연결 (Route 53, Cloudflare 등)

# 2. Certbot 설치
sudo snap install --classic certbot

# 3. Nginx에 인증서 자동 설정
sudo certbot --nginx -d your-domain.com

# 4. 자동 갱신 설정 (certbot이 자동으로 설정)
sudo certbot renew --dry-run
```

**장점:**
- 무료 정식 SSL 인증서
- 브라우저 경고 없음
- 자동 갱신

**단점:**
- 도메인 필요

---

## 수동 설정

### 1. Nginx 설치

```bash
sudo apt-get update
sudo apt-get install -y nginx
```

### 2. 자체 서명 인증서 생성

```bash
sudo mkdir -p /etc/nginx/ssl
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/ebrcs.key \
    -out /etc/nginx/ssl/ebrcs.crt \
    -subj "/C=US/ST=State/L=City/O=EBRCS/CN=your-domain-or-ip"
```

### 3. Nginx 설정 복사

```bash
sudo cp nginx/ebrcs.conf /etc/nginx/sites-available/ebrcs
sudo ln -s /etc/nginx/sites-available/ebrcs /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default  # 기본 사이트 비활성화
```

### 4. Nginx 재시작

```bash
sudo nginx -t  # 설정 테스트
sudo systemctl restart nginx
sudo systemctl enable nginx  # 부팅 시 자동 시작
```

---

## 웹앱 실행

HTTPS 설정 후 웹앱 실행:

```bash
cd app
./run_web_production.sh
```

- Nginx: 포트 80 (HTTP → HTTPS 리다이렉트), 443 (HTTPS)
- Backend: 포트 8000 (localhost only)
- Frontend: 포트 5173 (localhost only)

---

## 접속 방법

### 자체 서명 인증서 사용 시

1. 브라우저에서 접속:
   ```
   https://your-server-ip
   ```

2. 보안 경고가 나타나면:
   - **Chrome/Edge**: "고급" → "안전하지 않음(계속 진행)" 클릭
   - **Firefox**: "고급..." → "위험을 감수하고 계속" 클릭
   - **Safari**: "세부사항 보기" → "웹 사이트 방문" 클릭

3. 카메라 시작 버튼을 누르면 정상 작동합니다!

### Let's Encrypt 인증서 사용 시

브라우저 경고 없이 바로 접속 가능:
```
https://your-domain.com
```

---

## 최적화 설정

Nginx 설정에 포함된 최적화:

- **`proxy_buffering off`**: 실시간 데이터 전송
- **`proxy_request_buffering off`**: WebSocket 버퍼링 제거
- **WebSocket 지원**: `/api/ws/*` 경로 지원
- **장시간 연결**: 타임아웃 86400초 (24시간)

---

## 문제 해결

### 카메라가 작동하지 않음

1. **HTTPS로 접속했는지 확인**
   - URL이 `https://`로 시작하는지 확인

2. **브라우저 콘솔 확인** (F12)
   - 에러 메시지 확인

3. **WebSocket 연결 확인**
   - 개발자 도구 → Network → WS 탭 확인

### Nginx 에러

```bash
# Nginx 로그 확인
sudo tail -f /var/log/nginx/error.log

# Nginx 설정 테스트
sudo nginx -t

# Nginx 재시작
sudo systemctl restart nginx
```

### Backend 에러

```bash
# Backend 로그 확인
tail -f app/logs/backend.log

# Backend 재시작
cd app
./stop_web.sh
./run_web_production.sh
```

---

## AWS EC2 보안 그룹 설정

AWS EC2에서 실행하는 경우 보안 그룹에서 다음 포트를 열어야 합니다:

- **포트 80** (HTTP) - HTTPS로 리다이렉트
- **포트 443** (HTTPS) - 메인 접속 포트
- **포트 22** (SSH) - 서버 관리용

---

## 참고 자료

- [Nginx 공식 문서](https://nginx.org/en/docs/)
- [Let's Encrypt 가이드](https://letsencrypt.org/getting-started/)
- [WebSocket with Nginx](https://nginx.org/en/docs/http/websocket.html)
