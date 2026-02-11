# 🛒 EBRCS - Embedding-Based Real-time Checkout System

**AI 기반 실시간 무인 계산 시스템**

DINOv3 + CLIP 하이브리드 임베딩을 활용한 상품 자동 인식 및 계산 시스템입니다.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-red.svg)](https://streamlit.io/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/YOUR_USERNAME/EBRCS_streaming)

## 📑 목차

- [주요 기능](#-주요-기능)
- [시스템 아키텍처](#-시스템-아키텍처)
- [프로젝트 구조](#-프로젝트-구조)
- [시작하기](#-시작하기)
  - [요구사항](#요구사항)
  - [Streamlit 데모 실행](#1-streamlit-데모-실행)
  - [웹앱 실행](#2-웹앱-실행)
- [데이터 준비](#-데이터-준비)
- [배포](#-배포)
  - [AWS EC2 배포](#-aws-ec2-배포)
  - [HTTPS 설정](#-https-설정-외부-카메라-접근-필수)
- [기술 스택](#-기술-스택)

---

## ✨ 주요 기능

### 🎯 실시간 상품 인식
- **DINOv3 + LoRA**: Facebook의 DINOv3 모델 + 커스텀 LoRA 어댑터
- **CLIP**: OpenAI의 멀티모달 임베딩
- **하이브리드 임베딩**: DINO(70%) + CLIP(30%) 가중 조합
- **FAISS**: 고속 벡터 유사도 검색

### 🛡️ 중복 방지 메커니즘
1. **Background Subtraction**: KNN 기반 동적 객체 탐지
2. **Frame Skip**: 5프레임마다 추론 (성능 최적화)
3. **Cooldown**: 동일 상품 3초 내 재카운트 방지
4. **ROI Entry Mode**: 관심 영역 진입 이벤트 감지

### 📊 두 가지 인터페이스
- **Streamlit 데모**: 빠른 프로토타입 테스트 및 데모
- **웹앱**: 프로덕션 레벨 FastAPI + React SPA

---

## 🏗️ 시스템 아키텍처

```
┌─────────────┐     WebSocket      ┌──────────────┐
│   React     │ ◄─────────────────►│   FastAPI    │
│  Frontend   │     SSE (Video)    │   Backend    │
└─────────────┘                    └──────┬───────┘
                                          │
                                    ┌─────▼────────┐
                                    │ checkout_core│
                                    │ (추론 엔진)    │
                                    └─────┬────────┘
                                          │
                        ┌─────────────────┼─────────────────┐
                        ▼                 ▼                 ▼
                   ┌─────────┐      ┌─────────┐      ┌─────────┐
                   │ DINOv3  │      │  CLIP   │      │  FAISS  │
                   │ + LoRA  │      │         │      │  Index  │
                   └─────────┘      └─────────┘      └─────────┘
```

### 핵심 알고리즘
```python
# 1. 프레임 처리 → 객체 탐지 (Background Subtraction)
# 2. ROI 진입 감지 → 추론 트리거
# 3. 임베딩 추출: DINO(0.7) + CLIP(0.3)
# 4. FAISS 검색 → Top-1 매칭
# 5. Cooldown 체크 → 카운트 업데이트
```

---

## 📁 프로젝트 구조

```
EBRCS_streaming/
├── streamlit/              # Streamlit 데모 (독립 실행)
│   ├── .venv/             # 전용 가상환경
│   ├── app.py             # 메인 앱
│   ├── pages/             # 페이지들
│   ├── requirements.txt   # 의존성
│   └── run.sh             # 실행 스크립트
│
├── app/                   # FastAPI + React 웹앱
│   ├── backend/
│   │   ├── .venv/        # Backend 가상환경
│   │   ├── main.py       # FastAPI 앱
│   │   ├── routers/      # API 라우터
│   │   ├── services/     # 비즈니스 로직
│   │   └── requirements.txt
│   ├── frontend/
│   │   ├── src/
│   │   ├── package.json
│   │   └── vite.config.ts
│   ├── run_web.sh        # 개발 모드
│   └── run_web_production.sh  # 프로덕션
│
├── checkout_core/         # 공유 추론 엔진 (수정 불가)
│   ├── inference.py      # 모델 로딩 & 임베딩 추출
│   ├── frame_processor.py # 프레임 처리 & 상품 인식
│   └── counting.py       # 중복 방지 로직
│
├── data/                  # 모델 & 임베딩 데이터
│   ├── adapter_config.json    # LoRA 설정 (Git 포함)
│   ├── adapter_model.safetensors  # LoRA 가중치 (별도 다운로드)
│   ├── embeddings.npy     # 상품 임베딩 DB (생성 필요)
│   ├── labels.npy         # 상품 레이블 (생성 필요)
│   └── faiss_index.bin    # FAISS 인덱스 (자동 생성)
│
├── product_images/        # 상품 이미지 (임베딩 생성용)
│   ├── 콜라/
│   ├── 사이다/
│   └── ...
│
├── generate_embeddings.py # 임베딩 DB 생성 스크립트
├── setup_aws_ec2.sh       # AWS EC2 자동 설정
├── .env.example           # 환경 변수 템플릿
└── PROJECT_STRUCTURE.md   # 상세 구조 문서
```

자세한 구조는 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 참고

---

## 🚀 시작하기

### 요구사항

- **Python**: 3.11+
- **Node.js**: 18+
- **Git**: 2.0+
- **CUDA** (선택): GPU 가속용

> **💡 크로스 플랫폼 지원**: Windows, macOS, Linux 모두 지원합니다!
> - Windows: `.bat` 배치 파일 사용
> - macOS/Linux: `.sh` 셸 스크립트 사용
> - 각 명령어는 OS별로 구분되어 있습니다 (🪟 Windows / 🍎 macOS / 🐧 Linux)

### 1️⃣ Streamlit 데모 실행

#### 🪟 Windows

```cmd
# 1. 저장소 클론
git clone https://github.com/YOUR_USERNAME/EBRCS_streaming.git
cd EBRCS_streaming

# 2. 환경 변수 설정
copy .env.example .env
notepad .env  # HF_TOKEN 입력

# 3. (선택) 초기 데이터 준비 - 빈 DB로 시작해도 됩니다!
#    웹 UI에서 실시간으로 상품 등록 가능

# 4. Streamlit 환경 설정
cd streamlit
setup_venv.bat

# 5. 실행
run.bat
```

#### 🍎 macOS / 🐧 Linux

```bash
# 1. 저장소 클론
git clone https://github.com/YOUR_USERNAME/EBRCS_streaming.git
cd EBRCS_streaming

# 2. 환경 변수 설정
cp .env.example .env
nano .env  # HF_TOKEN 입력

# 3. (선택) 초기 데이터 준비 - 빈 DB로 시작해도 됩니다!
#    웹 UI에서 실시간으로 상품 등록 가능

# 4. Streamlit 환경 설정
cd streamlit
./setup_venv.sh

# 5. 실행
source .venv/bin/activate
./run.sh
```

브라우저에서 http://localhost:8501 접속

### 2️⃣ 웹앱 실행

#### 🪟 Windows

```cmd
# 1. 환경 설정 (Backend + Frontend)
cd app
setup_venv.bat

# 2. 개발 모드 실행
run_web.bat
```

#### 🍎 macOS / 🐧 Linux

```bash
# 1. 환경 설정 (Backend + Frontend)
cd app
./setup_venv.sh

# 2. 개발 모드 실행
./run_web.sh
```

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000/docs

---

## 📦 데이터 준비

> **💡 중요**: 웹앱은 **빈 DB에서도 시작 가능**합니다!
>
> 상품 등록 방법:
> 1. **웹 UI 실시간 등록** (권장 ⭐) - 운영 중 언제든지 추가 가능
> 2. **오프라인 배치 생성** (선택) - 초기 대량 데이터 준비용

### Option 1: 웹 UI에서 상품 등록 (권장 ⭐)

**웹앱 실행 후**:
1. 브라우저에서 `http://localhost:5173` 접속
2. **"상품 등록"** 페이지 이동
3. 상품명 입력 + 이미지 1-3장 업로드
4. **즉시 인식 가능!** (서버 재시작 불필요)

**특징**:
- ✅ 실시간 업데이트
- ✅ 사용자 친화적 GUI
- ✅ 증분 업데이트로 빠름 (전체 재구축 안함)
- ✅ 운영 중에도 안전하게 추가 가능

---

### Option 2: 오프라인 배치 생성 (선택, 대량 등록용)

#### 🪟 Windows

```cmd
REM 1. 상품 이미지 준비 (수동으로 폴더 생성)
mkdir product_images\콜라
mkdir product_images\사이다
mkdir product_images\감자칩

REM 각 폴더에 상품 이미지 3-5장 추가
REM product_images\콜라\img1.jpg, img2.jpg, ...

REM 2. HuggingFace 토큰 설정
set HF_TOKEN=your_token_here

REM 3. 임베딩 생성 (약 5-10분 소요)
python generate_embeddings.py
```

#### 🍎 macOS / 🐧 Linux

```bash
# 1. 상품 이미지 준비
mkdir -p product_images/{콜라,사이다,감자칩}

# 각 폴더에 상품 이미지 3-5장 추가
# product_images/콜라/img1.jpg, img2.jpg, ...

# 2. HuggingFace 토큰 설정
export HF_TOKEN="your_token_here"

# 3. 임베딩 생성 (약 5-10분 소요)
python generate_embeddings.py
```

**출력**:
- `data/embeddings.npy` (245MB) - 상품 임베딩 벡터
- `data/labels.npy` (3.4MB) - 상품 이름 매핑
- FAISS 인덱스는 서버 시작 시 자동 생성

### Option 2: 사전 생성된 데이터 다운로드

#### 🪟 Windows

```cmd
REM Google Drive 또는 HuggingFace에서 다운로드
REM PowerShell 사용
powershell -Command "Invoke-WebRequest -Uri '<DOWNLOAD_LINK>' -OutFile data.zip"
powershell -Command "Expand-Archive -Path data.zip -DestinationPath data\"
```

#### 🍎 macOS / 🐧 Linux

```bash
# Google Drive 또는 HuggingFace에서 다운로드
wget <DOWNLOAD_LINK> -O data.zip
unzip data.zip -d data/
```

### 필수 파일 확인

#### 🪟 Windows
```cmd
dir data\
```

#### 🍎 macOS / 🐧 Linux
```bash
ls -lh data/
```

**필수 파일** (나머지는 자동 생성):
- ✅ `adapter_config.json` - LoRA 설정 (Git 포함)
- 📥 `adapter_model.safetensors` - LoRA 가중치 (**다운로드 필요**)

**자동 생성 파일** (없어도 서버 시작 가능):
- `embeddings.npy` - 상품 임베딩 (웹 UI 등록 시 자동 생성)
- `labels.npy` - 상품 레이블 (웹 UI 등록 시 자동 생성)
- `faiss_index.bin` - FAISS 인덱스 (서버 시작 시 자동 생성)

> **💡 빈 DB로 시작하면**: 첫 번째 상품 등록 시 자동으로 파일들이 생성됩니다!

---

## 🌐 배포

### AWS EC2 자동 배포

```bash
# EC2 Ubuntu 22.04 인스턴스에서
wget https://raw.githubusercontent.com/YOUR_USERNAME/EBRCS_streaming/main/setup_aws_ec2.sh
chmod +x setup_aws_ec2.sh
./setup_aws_ec2.sh
```

스크립트가 자동으로:
1. Python 3.11, Node.js 20 설치
2. 저장소 클론
3. 가상환경 설정 (Streamlit + Backend)
4. Frontend 빌드
5. 실행 스크립트 권한 설정

### 프로덕션 실행

```bash
cd ebrcs_streaming/app
./run_web_production.sh
```

### systemd 서비스 등록 (선택)

```bash
cd ebrcs_streaming/app
./setup_systemd.sh

# 이후 서비스 관리
sudo systemctl start ebrcs
sudo systemctl status ebrcs
sudo journalctl -u ebrcs -f
```

### Docker 배포

```bash
cd app
docker-compose up --build

# GPU 사용 시
docker-compose -f docker-compose.yml up
```

---

## 🛠️ 기술 스택

### AI/ML
- **DINOv3** (facebook/dinov2-base) + LoRA 어댑터
- **CLIP** (openai/clip-vit-base-patch32)
- **FAISS** - 고속 벡터 검색
- **PyTorch** - 딥러닝 프레임워크
- **Transformers** - HuggingFace 모델 로딩
- **PEFT** - LoRA 어댑터 적용

### Backend
- **FastAPI** - 고성능 비동기 API 프레임워크
- **Uvicorn** - ASGI 서버
- **WebSocket** - 실시간 카메라 스트리밍
- **SSE (Server-Sent Events)** - 비디오 처리 진행률
- **aiorwlock** - 비동기 Reader-Writer Lock

### Frontend
- **React 18** + TypeScript
- **Vite** - 빌드 도구
- **Tailwind CSS v4** - 스타일링
- **Zustand** - 상태 관리
- **TanStack Query** - 서버 상태 관리

### Computer Vision
- **OpenCV** - 이미지 처리
- **Background Subtraction (KNN)** - 동적 객체 탐지
- **ROI (Region of Interest)** - 관심 영역 설정

---

## 📊 성능 지표

| 지표 | 값 |
|------|-----|
| 추론 속도 | ~350ms/frame (CPU) |
| 매칭 정확도 | 85-90% (임베딩 기반) |
| 중복 방지율 | 99%+ (3초 쿨다운) |
| 동시 세션 | 10+ (FastAPI 비동기) |
| 상품 추가 시간 | ~2분 (5장 기준) |

---

## 🔐 환경 변수

`.env` 파일 설정:

```bash
# HuggingFace 토큰 (모델 다운로드용)
HF_TOKEN=your_huggingface_token_here
HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here

# 선택 사항
# KMP_DUPLICATE_LIB_OK=TRUE  # macOS OpenMP 이슈 해결
```

---

## 📚 주요 상수 (변경 금지)

`backend/config.py`:
```python
MATCH_THRESHOLD = 0.62        # FAISS 매칭 임계값
MIN_AREA = 2500              # 최소 객체 면적
DETECT_EVERY_N_FRAMES = 5    # 프레임 스킵
COUNT_COOLDOWN_SECONDS = 3.0 # 중복 방지 쿨다운
ROI_CLEAR_FRAMES = 8         # ROI 클리어 프레임
DINO_WEIGHT = 0.7            # DINO 임베딩 가중치
CLIP_WEIGHT = 0.3            # CLIP 임베딩 가중치
```

---

## 🐛 트러블슈팅

### 1. `faiss-cpu` 설치 실패

#### 🪟 Windows
```cmd
REM Anaconda 사용 (권장)
conda install -c conda-forge faiss-cpu

REM 또는 pip
pip install faiss-cpu --no-cache-dir
```

#### 🍎 macOS
```bash
# M1/M2 칩
conda install -c conda-forge faiss-cpu

# Intel 칩
pip install faiss-cpu
```

#### 🐧 Linux
```bash
pip install faiss-cpu --no-cache-dir
```

### 2. `ModuleNotFoundError: No module named 'streamlit'`

#### 🪟 Windows
```cmd
REM backend\.venv가 아닌 streamlit\.venv 사용 확인
cd streamlit
.venv\Scripts\activate
```

#### 🍎 macOS / 🐧 Linux
```bash
# backend/.venv가 아닌 streamlit/.venv 사용 확인
cd streamlit
source .venv/bin/activate
```

### 3. Python 명령어 찾을 수 없음

#### 🪟 Windows
```cmd
REM "python3"가 없으면 "python" 사용
python --version

REM PATH 확인
where python
```

#### 🍎 macOS / 🐧 Linux
```bash
# "python"이 없으면 "python3" 사용
python3 --version

# PATH 확인
which python3
```

### 4. 가상환경 활성화 오류

**Windows PowerShell 실행 정책 오류**:
```powershell
# PowerShell을 관리자 권한으로 실행 후
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**또는 CMD 사용** (PowerShell 대신):
```cmd
.venv\Scripts\activate.bat
```

### 5. CUDA Out of Memory
```python
# backend/config.py 또는 generate_embeddings.py
DEVICE = "cpu"  # GPU → CPU 전환
```

### 6. Frontend CORS 에러
```typescript
// frontend/vite.config.ts 확인
server: {
  proxy: {
    '/api': 'http://localhost:8000'
  }
}
```

### 7. Port 이미 사용 중 오류

#### 🪟 Windows
```cmd
REM 8000 포트 사용 중인 프로세스 찾기
netstat -ano | findstr :8000

REM 프로세스 종료 (PID 확인 후)
taskkill /PID <PID> /F
```

#### 🍎 macOS / 🐧 Linux
```bash
# 8000 포트 사용 중인 프로세스 찾기
lsof -ti:8000

# 프로세스 종료
kill -9 $(lsof -ti:8000)
```

---

## 🌐 AWS EC2 배포

### 🚀 완전 자동 배포 (권장)

**단 3단계로 AWS EC2에 배포 완료!**

#### 1️⃣ EC2 준비

- **인스턴스 타입**: t3.large 이상 권장 (GPU 있으면 g4dn.xlarge)
- **OS**: Ubuntu 22.04 LTS 또는 24.04 LTS
- **스토리지**: 30GB 이상
- **보안 그룹**:
  - SSH (22) - 내 IP만
  - HTTP (80) - 0.0.0.0/0
  - HTTPS (443) - 0.0.0.0/0

#### 2️⃣ 자동 설치 스크립트 실행

EC2에 SSH 접속 후:

```bash
wget https://raw.githubusercontent.com/Smart-Cart-5/EBRCS/main/setup_aws_ec2_complete.sh
chmod +x setup_aws_ec2_complete.sh
./setup_aws_ec2_complete.sh
```

**자동으로 설치되는 것**:
- ✅ Python 3.11 + Node.js 20
- ✅ Backend/Frontend 환경 설정
- ✅ Nginx 리버스 프록시 (80 포트)
- ✅ 모든 의존성 패키지

#### 3️⃣ 데이터 업로드 & 실행

**로컬에서 data 폴더 업로드**:
```bash
scp -i your-key.pem -r data/* ubuntu@YOUR_EC2_IP:~/ebrcs_streaming/data/
```

**EC2에서 웹앱 실행**:
```bash
cd ~/ebrcs_streaming/app
./run_web_production.sh
```

**접속**:
```
http://YOUR_EC2_IP
```

#### 📊 프로덕션 모드 vs 개발 모드

| 항목 | 개발 (`run_web.sh`) | 프로덕션 (`run_web_production.sh`) |
|------|---------------------|-------------------------------------|
| 접속 | localhost만 | 외부 접속 가능 |
| Frontend | Vite dev (핫 리로드) | 빌드된 정적 파일 |
| Backend | `--reload` | `--workers 2` |
| 백그라운드 | ❌ | ✅ (nohup) |
| 포트 | 5173, 8000 | 80 (Nginx) |

#### 🛑 웹앱 종료

```bash
cd ~/ebrcs_streaming/app
./stop_web.sh
```

#### 📊 로그 확인

```bash
# Backend 로그
tail -f ~/ebrcs_streaming/app/logs/backend.log

# Frontend 로그
tail -f ~/ebrcs_streaming/app/logs/frontend.log
```

---

### 🔒 HTTPS 설정 (외부 카메라 접근 필수)

**중요**: 브라우저의 보안 정책상 외부에서 카메라를 사용하려면 **반드시 HTTPS**가 필요합니다.

#### 왜 HTTPS가 필요한가?

`getUserMedia()` (카메라 API)는 다음 환경에서만 작동:
- ✅ `localhost` / `127.0.0.1`
- ✅ **HTTPS 연결**

HTTP로 외부 접속 시 카메라를 사용할 수 없습니다!

#### 자동 HTTPS 설정 (5분 완료)

```bash
cd ~/ebrcs_streaming
sudo ./setup_https.sh
```

이 스크립트가 자동으로:
1. ✅ Nginx 설치 및 설정
2. ✅ 자체 서명 SSL 인증서 생성
3. ✅ HTTP → HTTPS 리다이렉트 설정
4. ✅ WebSocket over HTTPS 지원

#### 접속 방법

```
https://YOUR_EC2_IP
```

**브라우저 보안 경고 처리**:
1. **Chrome/Edge**: "고급" → "안전하지 않음(계속 진행)" 클릭
2. **Firefox**: "고급..." → "위험을 감수하고 계속" 클릭
3. **Safari**: "세부사항 보기" → "웹 사이트 방문" 클릭

이후 카메라가 정상 작동합니다! 🎉

#### Let's Encrypt 정식 인증서 (프로덕션 권장)

도메인이 있는 경우 무료 정식 SSL 인증서 사용 가능:

```bash
# 1. 도메인을 EC2 IP에 연결 (Route 53, Cloudflare 등)

# 2. Certbot 설치
sudo snap install --classic certbot

# 3. 자동 인증서 설정
sudo certbot --nginx -d your-domain.com

# 4. 자동 갱신 확인
sudo certbot renew --dry-run
```

**장점**:
- ✅ 브라우저 경고 없음
- ✅ 무료
- ✅ 자동 갱신

#### 추가 정보

자세한 설정 방법은 [HTTPS_SETUP.md](HTTPS_SETUP.md) 참고

**AWS 보안 그룹 필수 포트**:
- 포트 **80** (HTTP) - HTTPS 리다이렉트
- 포트 **443** (HTTPS) - 메인 접속
- 포트 22 (SSH) - 서버 관리

---

## 📝 License

MIT License - 자유롭게 사용, 수정, 배포 가능

---

## 👥 기여

이슈 및 Pull Request 환영합니다!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

