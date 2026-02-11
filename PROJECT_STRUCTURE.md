# 📚 EBRCS 프로젝트 구조 상세 문서

## 📋 목차

1. [개요](#개요)
2. [전체 디렉토리 구조](#전체-디렉토리-구조)
3. [주요 디렉토리 상세](#주요-디렉토리-상세)
4. [파일별 설명](#파일별-설명)
5. [실행 흐름](#실행-흐름)
6. [데이터 흐름](#데이터-흐름)
7. [의존성 관리](#의존성-관리)
8. [배포 전략](#배포-전략)

---

## 개요

EBRCS는 **두 가지 독립적인 인터페이스**를 제공합니다:

1. **Streamlit 데모** (`streamlit/`) - 빠른 프로토타입 & 데모용
2. **웹앱** (`app/`) - 프로덕션 레벨 FastAPI + React SPA

두 인터페이스 모두 **동일한 추론 엔진** (`checkout_core/`)과 **데이터** (`data/`)를 공유하지만, 각각 **독립적인 가상환경**에서 실행됩니다.

---

## 전체 디렉토리 구조

```
EBRCS_streaming/
│
├── 📁 streamlit/              # Streamlit 데모 (독립 실행)
│   ├── .venv/                # 전용 가상환경 (Git 제외)
│   ├── app.py                # 데스크톱 메인 앱
│   ├── mobile_app.py         # 모바일 버전
│   ├── pages/                # 데스크톱 페이지
│   │   ├── 01_상품등록.py
│   │   ├── 02_체크아웃.py
│   │   ├── 03_결제확인.py
│   │   └── 04_데이터검증.py
│   ├── pages_mobile/         # 모바일 페이지
│   │   ├── 01_체크아웃.py
│   │   ├── 02_상품목록.py
│   │   └── 03_결제확인.py
│   ├── ui_theme.py           # UI 테마 & 스타일
│   ├── mobile_nav.py         # 모바일 네비게이션
│   ├── requirements.txt      # Streamlit 의존성
│   ├── setup_venv.sh        # 가상환경 설정 스크립트 ⚙️
│   ├── run.sh               # Streamlit 실행 🚀
│   └── run_mobile.sh        # 모바일 버전 실행 📱
│
├── 📁 app/                    # FastAPI + React 웹앱
│   │
│   ├── 📁 backend/
│   │   ├── .venv/           # Backend 가상환경 (Git 제외)
│   │   ├── main.py          # FastAPI 앱 진입점
│   │   ├── config.py        # 설정 & 상수
│   │   ├── st_shim.py       # Streamlit Mock (checkout_core 호환용)
│   │   ├── dependencies.py  # DI (Dependency Injection)
│   │   │
│   │   ├── 📁 routers/      # API 엔드포인트
│   │   │   ├── sessions.py     # 세션 CRUD
│   │   │   ├── checkout.py     # 체크아웃 WebSocket/SSE
│   │   │   ├── billing.py      # 결제 API
│   │   │   └── products.py     # 상품 등록 API
│   │   │
│   │   ├── 📁 services/     # 비즈니스 로직
│   │   │   ├── session_manager.py  # 세션 관리
│   │   │   └── product_manager.py  # 상품 DB 관리
│   │   │
│   │   └── requirements.txt # Backend 의존성
│   │
│   ├── 📁 frontend/
│   │   ├── 📁 src/
│   │   │   ├── 📁 pages/        # React 페이지
│   │   │   │   ├── HomePage.tsx
│   │   │   │   ├── CheckoutPage.tsx
│   │   │   │   ├── ProductsPage.tsx
│   │   │   │   └── ValidatePage.tsx
│   │   │   ├── 📁 api/          # API 클라이언트
│   │   │   │   └── client.ts
│   │   │   ├── 📁 store/        # Zustand 상태 관리
│   │   │   │   └── sessionStore.ts
│   │   │   ├── App.tsx          # 메인 앱
│   │   │   └── main.tsx         # 진입점
│   │   ├── package.json         # Node 의존성
│   │   ├── vite.config.ts       # Vite 설정
│   │   ├── tailwind.config.js   # Tailwind CSS v4
│   │   └── tsconfig.json        # TypeScript 설정
│   │
│   ├── setup_venv.sh            # Backend + Frontend 환경 설정 ⚙️
│   ├── run_web.sh               # 개발 모드 실행 🚀
│   ├── run_web_production.sh    # 프로덕션 실행 🌐
│   ├── stop_web.sh              # 웹앱 종료 🛑
│   ├── Dockerfile               # 컨테이너 이미지
│   ├── docker-compose.yml       # Docker Compose 설정
│   ├── ebrcs.service            # systemd 서비스 파일
│   └── setup_systemd.sh         # systemd 등록 스크립트
│
├── 📁 checkout_core/          # 공유 추론 엔진 (수정 불가 ⚠️)
│   ├── inference.py          # 모델 로딩 & 임베딩 추출
│   │   - load_models()          → DINOv3 + LoRA, CLIP 로딩
│   │   - extract_dino_embedding() → DINO 임베딩
│   │   - extract_clip_embedding() → CLIP 임베딩
│   │   - build_query_embedding()  → 하이브리드 임베딩
│   │   - load_product_db()        → embeddings.npy, labels.npy 로딩
│   │   - build_faiss_index()      → FAISS 인덱스 생성
│   │
│   ├── frame_processor.py    # 프레임 처리 & 상품 인식
│   │   - create_bg_subtractor()   → KNN Background Subtractor
│   │   - process_checkout_frame() → 메인 처리 루프
│   │
│   └── counting.py          # 중복 방지 로직
│       - should_count_product()   → Cooldown 체크
│       - ensure_last_seen_at_state() → 상태 초기화
│
├── 📁 data/                   # 모델 & 임베딩 데이터
│   ├── adapter_config.json       # LoRA 설정 (Git 포함 ✅)
│   ├── adapter_model.safetensors # LoRA 가중치 (27MB, 다운로드 필요 📥)
│   ├── embeddings.npy            # 상품 임베딩 (245MB, 생성 필요 🔨)
│   ├── labels.npy                # 상품 레이블 (3.4MB, 생성 필요 🔨)
│   └── faiss_index.bin           # FAISS 인덱스 (자동 생성 ⚙️)
│
├── 📁 product_images/         # 상품 이미지 (임베딩 생성용)
│   ├── 콜라/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── img3.jpg
│   ├── 사이다/
│   └── ... (상품별 폴더)
│
├── 📁 docs/                   # 문서
│   └── AWS_DEPLOYMENT_GUIDE.md
│
├── generate_embeddings.py    # 임베딩 DB 생성 스크립트 🔨
├── setup_aws_ec2.sh          # AWS EC2 자동 설정 ☁️
│
├── .env.example              # 환경 변수 템플릿
├── .env                      # 실제 환경 변수 (Git 제외 🔒)
├── .gitignore               # Git 제외 파일 목록
│
├── README.md                 # 프로젝트 메인 문서
├── PROJECT_STRUCTURE.md      # 이 문서
└── requirements.txt          # 레거시 (참고용)
```

더 자세한 내용은 각 섹션을 참고하세요.

---

## 주요 파일 설명

### 실행 스크립트 (크로스 플랫폼 지원)

**🪟 Windows (.bat)**:
| 스크립트 | 위치 | 역할 |
|---------|------|------|
| `setup_venv.bat` | `streamlit/` | Streamlit 가상환경 설정 |
| `setup_venv.bat` | `app/` | Backend + Frontend 환경 설정 |
| `run.bat` | `streamlit/` | Streamlit 앱 실행 |
| `run_web.bat` | `app/` | 웹앱 개발 모드 (새 창 2개) |

**🍎 macOS / 🐧 Linux (.sh)**:
| 스크립트 | 위치 | 역할 |
|---------|------|------|
| `setup_venv.sh` | `streamlit/` | Streamlit 가상환경 설정 |
| `setup_venv.sh` | `app/` | Backend + Frontend 환경 설정 |
| `run.sh` | `streamlit/` | Streamlit 앱 실행 |
| `run_web.sh` | `app/` | 웹앱 개발 모드 |
| `run_web_production.sh` | `app/` | 웹앱 프로덕션 모드 |
| `stop_web.sh` | `app/` | 웹앱 종료 |
| `setup_aws_ec2.sh` | 루트 | AWS EC2 자동 설정 |

---

## 사용법

### Streamlit 데모 실행

#### 🪟 Windows
```cmd
cd streamlit
setup_venv.bat         REM 최초 1회만
run.bat
```

#### 🍎 macOS / 🐧 Linux
```bash
cd streamlit
./setup_venv.sh        # 최초 1회만
source .venv/bin/activate
./run.sh
```

### 웹앱 실행

#### 🪟 Windows
```cmd
cd app
setup_venv.bat         REM 최초 1회만
run_web.bat            REM 개발 모드 (새 창 2개 열림)
```

#### 🍎 macOS / 🐧 Linux
```bash
cd app
./setup_venv.sh        # 최초 1회만
./run_web.sh           # 개발 모드
```

### 임베딩 생성

```bash
# 루트 디렉토리에서 (Windows/macOS/Linux 모두 동일)
python generate_embeddings.py
```

---

## 의존성 관리

### Streamlit (`streamlit/requirements.txt`)
- streamlit, streamlit-drawable-canvas
- AI/ML: numpy, pandas, opencv, transformers, torch, faiss-cpu

### Backend (`app/backend/requirements.txt`)
- fastapi, uvicorn, websockets, aiofiles, aiorwlock
- AI/ML: Streamlit과 동일

### Frontend (`app/frontend/package.json`)
- React 18, TypeScript, Vite
- Tailwind CSS v4, Zustand, TanStack Query

---

## 배포

### AWS EC2
```bash
./setup_aws_ec2.sh     # 자동 설정
cd app
./run_web_production.sh
```

### Docker
```bash
cd app
docker-compose up --build
```

---

## 환경 변수

`.env` 파일 (프로젝트 루트):
```
HF_TOKEN=your_huggingface_token
HUGGINGFACE_HUB_TOKEN=your_huggingface_token
```

---

## 주의사항

1. **checkout_core/** 디렉토리는 수정하지 않음
2. **data/** 폴더는 streamlit과 app에서 공유
3. **generate_embeddings.py**는 루트에서 실행
4. **.env**는 루트에 위치
5. **PYTHONPATH**는 실행 스크립트에서 자동 설정

---

## 트러블슈팅

### 가상환경 활성화 안됨
```bash
cd streamlit  # 또는 cd app
./setup_venv.sh
```

### 모듈 import 에러
- PYTHONPATH 확인
- 실행 스크립트 사용 권장

### Frontend 빌드 에러
```bash
cd app/frontend
rm -rf node_modules
npm install
```

---

**작성일**: 2025-02-11
