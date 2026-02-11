# 🚀 AWS EC2 배포 가이드 (Git Clone 방식)

## 📋 전체 플로우

```
1. GitHub에 코드 push
2. AWS EC2 인스턴스 생성
3. SSH 접속
4. setup_aws_ec2.sh 실행
5. .env 설정 + data/ 업로드
6. run_web_production.sh 실행
7. 완료!
```

---

## 1️⃣ GitHub에 코드 Push

### A. .gitignore 확인

불필요한 파일 제외:
```bash
# .gitignore에 추가 확인
data/
venv/
node_modules/
*.log
.env
```

### B. Push

```bash
git add .
git commit -m "Prepare for AWS deployment"
git push origin main
```

---

## 2️⃣ AWS EC2 인스턴스 생성

### A. AWS 콘솔 접속

https://console.aws.amazon.com/ec2

### B. Launch Instance

**설정:**
```
Name: ebrcs-webapp

AMI: Ubuntu Server 22.04 LTS (Free tier eligible)

Instance Type:
  - CPU only: t3.medium (2 vCPU, 4GB RAM) ~$30/월
  - GPU: g4dn.xlarge (4 vCPU, 16GB RAM, T4 GPU) ~$400/월

Key pair:
  - Create new key pair
  - Name: ebrcs-key
  - Type: RSA
  - Format: .pem
  - Download!

Network settings:
  - Allow SSH (22) from My IP
  - Allow HTTP (80) from Anywhere
  - Allow HTTPS (443) from Anywhere
  - Custom TCP (5173) from Anywhere ← Frontend
  - Custom TCP (8000) from Anywhere ← Backend API

Storage: 30 GB gp3
```

### C. Launch!

**중요:** Key pair (.pem) 다운로드 후 안전하게 보관

---

## 3️⃣ SSH 접속

### A. Key 파일 권한 설정

```bash
chmod 400 ~/Downloads/ebrcs-key.pem
```

### B. SSH 접속

```bash
ssh -i ~/Downloads/ebrcs-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

**EC2 Public IP 확인:**
- AWS 콘솔 → EC2 → Instances → Public IPv4 address

---

## 4️⃣ 초기 설정 (EC2 내부)

### A. 설정 스크립트 다운로드 및 실행

```bash
# 1. 임시 디렉토리 생성
mkdir -p ~/temp && cd ~/temp

# 2. 설정 스크립트 다운로드
wget https://raw.githubusercontent.com/YOUR_USERNAME/EBRCS_streaming/main/setup_aws_ec2.sh

# 3. 실행 권한
chmod +x setup_aws_ec2.sh

# 4. 실행
./setup_aws_ec2.sh
```

**입력할 내용:**
```
GitHub 저장소 URL 입력: https://github.com/YOUR_USERNAME/EBRCS_streaming.git
```

### B. 환경 변수 설정

```bash
cd ~/ebrcs_streaming

# .env 파일 수정
nano .env
```

**입력:**
```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

저장: `Ctrl+O` → `Enter` → `Ctrl+X`

### C. data/ 폴더 업로드

**로컬 터미널에서 (새 터미널):**
```bash
# data 폴더 압축 (로컬)
cd /Users/kimminseong/Desktop/UNIV/LIKE_LION/last_project/EBRCS_streaming
tar -czf data.tar.gz data/

# EC2로 업로드
scp -i ~/Downloads/ebrcs-key.pem \
    data.tar.gz \
    ubuntu@YOUR_EC2_PUBLIC_IP:~/ebrcs_streaming/

# EC2에서 압축 해제 (EC2 SSH)
cd ~/ebrcs_streaming
tar -xzf data.tar.gz
rm data.tar.gz
```

---

## 5️⃣ 웹앱 실행

### Option A: 수동 실행 (테스트용)

```bash
cd ~/ebrcs_streaming
./run_web_production.sh
```

**접속:**
```
http://YOUR_EC2_PUBLIC_IP:5173
```

### Option B: systemd 서비스 (프로덕션 추천)

```bash
cd ~/ebrcs_streaming
./setup_systemd.sh
```

**서비스 관리:**
```bash
# 상태 확인
sudo systemctl status ebrcs

# 재시작
sudo systemctl restart ebrcs

# 로그 확인
sudo journalctl -u ebrcs -f
```

---

## 6️⃣ 도메인 연결 (선택 사항)

### A. Elastic IP 할당

1. AWS 콘솔 → EC2 → Elastic IPs
2. Allocate Elastic IP address
3. Associate Elastic IP address → 인스턴스 선택

### B. 도메인 DNS 설정

Route 53 또는 외부 DNS:
```
Type: A
Name: ebrcs.yourdomain.com
Value: YOUR_ELASTIC_IP
```

### C. Nginx 리버스 프록시 (80 포트)

```bash
# Nginx 설치
sudo apt-get install -y nginx

# 설정 파일 생성
sudo nano /etc/nginx/sites-available/ebrcs
```

**내용:**
```nginx
server {
    listen 80;
    server_name ebrcs.yourdomain.com;

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
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
    }
}
```

**활성화:**
```bash
sudo ln -s /etc/nginx/sites-available/ebrcs /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**접속:**
```
http://ebrcs.yourdomain.com
```

---

## 7️⃣ SSL/TLS 설정 (HTTPS)

```bash
# Certbot 설치
sudo apt-get install -y certbot python3-certbot-nginx

# SSL 인증서 발급
sudo certbot --nginx -d ebrcs.yourdomain.com

# 자동 갱신 확인
sudo certbot renew --dry-run
```

**접속:**
```
https://ebrcs.yourdomain.com
```

---

## 🔧 문제 해결

### 1. 포트 접근 안됨

**Security Group 확인:**
- AWS 콘솔 → EC2 → Security Groups
- Inbound rules에 5173, 8000 포트 추가

### 2. 모델 로딩 실패

**로그 확인:**
```bash
tail -f ~/ebrcs_streaming/logs/backend.log
```

**HF_TOKEN 확인:**
```bash
cat ~/ebrcs_streaming/.env
```

### 3. Frontend 빌드 실패

**Node.js 버전 확인:**
```bash
node -v  # v20.x 이상
npm -v
```

---

## 📊 비용 예상

| 항목 | CPU (t3.medium) | GPU (g4dn.xlarge) |
|------|-----------------|-------------------|
| EC2 인스턴스 | ~$30/월 | ~$400/월 |
| Elastic IP | 무료 (사용 중) | 무료 (사용 중) |
| Storage (30GB) | ~$3/월 | ~$3/월 |
| 총 | **~$33/월** | **~$403/월** |

**절약 팁:**
- Reserved Instance (1년): ~40% 할인
- Spot Instance: ~70% 할인 (중단 가능성 있음)

---

## 🚀 업데이트 방법

### 코드 업데이트

```bash
# EC2 SSH
cd ~/ebrcs_streaming
git pull origin main

# Frontend 재빌드
cd frontend
npm ci
npm run build
cd ..

# 서비스 재시작
sudo systemctl restart ebrcs
```

### 의존성 업데이트

```bash
source venv/bin/activate
pip install -r requirements.txt --upgrade

cd frontend
npm ci
cd ..

sudo systemctl restart ebrcs
```

---

## ✅ 완료!

이제 AWS EC2에서 EBRCS 웹앱이 실행 중입니다!

**접속 주소:**
- Frontend: `http://YOUR_EC2_IP:5173`
- Backend API: `http://YOUR_EC2_IP:8000/api/health`

**다음 단계:**
- [ ] 도메인 연결
- [ ] SSL 인증서 설정
- [ ] 모니터링 설정 (CloudWatch)
- [ ] 백업 자동화
