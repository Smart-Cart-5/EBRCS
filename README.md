# 🛒 EBRCS - Embedding-Based Real-time Checkout System

**AI 기반 실시간 무인 계산 시스템**

DINOv3 + CLIP 하이브리드 임베딩을 활용한 상품 자동 인식 및 계산 시스템입니다.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/Smart-Cart-5/EBRCS)

## 📑 목차

- [주요 기능](#-주요-기능)
- [서비스 화면](#-서비스-화면)
  - [모바일 버전 (일반 사용자)](#-모바일-버전-일반-사용자)
  - [관리자 버전 (모바일)](#-관리자-버전-모바일)
- [시스템 아키텍처](#-시스템-아키텍처)
- [프로젝트 구조](#-프로젝트-구조)
- [시작하기](#-시작하기)
  - [요구사항](#요구사항)
  - [웹앱 실행](#웹앱-실행)
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
- **EasyOCR**: 컵밥 계열 상품 정밀 라벨 보정 (한국어 + 영어)

### 🔍 EasyOCR 컵밥 정밀 인식
컵밥/컵라면처럼 외관이 유사한 상품은 FAISS 임베딩만으로 정확한 식별이 어렵습니다. 이를 위해 EasyOCR 기반의 2단계 인식 파이프라인이 작동합니다:

1. **FAISS 1차 매칭**: 임베딩 유사도로 "컵밥 계열"임을 판별
2. **OCR Pending 진입**: 컵밥 상품 감지 시 자동으로 OCR 대기 상태 전환
3. **실시간 OCR 시도**: 사용자가 상품 앞면을 카메라에 비추면 EasyOCR이 텍스트를 읽어 정확한 상품명(예: 황태국밥, 스팸마요덮밥 등) 확정
4. **타임아웃 Fallback**: 6초 내 인식 실패 시 FAISS 결과로 대체
5. **사용자 UI**: "상품 인식이 잘 안됐어요" 모달 + 취소 버튼 제공

> 대상 상품: CJ햇반컵반, 오뚜기컵밥, 동원컵밥 등 25종

### 🛡️ 중복 방지 메커니즘
1. **Background Subtraction**: KNN 기반 동적 객체 탐지
2. **Frame Skip**: 5프레임마다 추론 (성능 최적화)
3. **Cooldown**: 동일 상품 3초 내 재카운트 방지
4. **ROI Entry Mode**: 관심 영역 진입 이벤트 감지
5. **OCR Track Cache**: 동일 track ID의 OCR 결과 재사용 (중복 OCR 실행 방지)

### 🔐 사용자 인증 & 관리
- **JWT 기반 인증**: 회원가입, 로그인, 자동 로그인 (7일)
- **역할 기반 접근 제어 (RBAC)**: User / Admin 분리
- **Admin 대시보드**: 실시간 통계, 인기 상품 TOP 5, 최근 구매 내역
- **구매 내역 관리**: 사용자별 구매 기록 저장 및 조회
- **모바일 최적화**: 반응형 UI, 프로필 드롭다운 메뉴

---

## 🎬 서비스 화면

> GIF 파일은 `docs/gifs/` 디렉토리에 아래 경로 규칙으로 추가하면 자동으로 표시됩니다.

---

### 📱 모바일 버전 (일반 사용자)

<table>
  <tr>
    <td align="center" width="33%">
      <b>🔐 로그인 / 회원가입</b><br><br>
      <img src="docs/gifs/모바일로그인.gif" width="220" alt="로그인 화면" /><br><br>
      <sub>JWT 기반 로그인 · 7일 자동 유지<br>자체 회원가입 (username / password)</sub>
    </td>
    <td align="center" width="33%">
      <b>🏠 홈 화면</b><br><br>
      <img src="docs/gifs/홈화면.gif" width="220" alt="홈 화면" /><br><br>
      <sub>체크아웃 세션 시작<br>마이페이지 · 프로필 드롭다운 접근</sub>
    </td>
    <td align="center" width="33%">
      <b>📷 실시간 체크아웃</b><br><br>
      <img src="docs/gifs/체크아웃페이지.gif" width="220" alt="체크아웃 화면" /><br><br>
      <sub>카메라 스트리밍 → 상품 자동 인식<br>실시간 장바구니 업데이트 · 추가/제거 토스트</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <b>🔍 컵밥 OCR 정밀 인식</b><br><br>
      <img src="docs/gifs/OCR 정밀 인식.gif" width="220" alt="OCR 인식 화면" /><br><br>
      <sub>유사 외관 컵밥 상품 정밀 구분<br>EasyOCR 2단계 보정 · 6초 타임아웃</sub>
    </td>
    <td align="center" width="33%">
      <b>🧾 영수증 확인 / 결제</b><br><br>
      <img src="docs/gifs/영수증확인/결제.gif" width="220" alt="영수증 확인 화면" /><br><br>
      <sub>최종 장바구니 수량 조정<br>구매 확정 및 내역 저장</sub>
    </td>
    <td align="center" width="33%">
      <b>👤 마이페이지</b><br><br>
      <img src="docs/gifs/마이페이지.gif" width="220" alt="마이페이지 화면" /><br><br>
      <sub>내 구매 내역 조회<br>날짜 · 금액 · 상품 목록 확인</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <b>🤖 AI 쇼핑 어시스턴트 (챗봇)</b><br><br>
      <img src="docs/gifs/챗봇기능.gif" width="220" alt="챗봇 화면" /><br><br>
      <sub>LLM 기반 자연어 장바구니 제어<br>"스팸마요 2개 추가해줘" · 음성 입력 지원<br>상품 금액 · 할인 정보 질의응답</sub>
    </td>
    <td align="center" width="33%">
      <!-- 추가 화면이 생기면 여기에 삽입 -->
    </td>
    <td align="center" width="33%">
      <!-- 추가 화면이 생기면 여기에 삽입 -->
    </td>
  </tr>
</table>

---

### 📱 관리자 버전 (모바일)

<table>
  <tr>
    <td align="center" width="33%">
      <b>📊 관리자 대시보드</b><br><br>
      <img src="docs/gifs/관리자 대시보드.gif" width="220" alt="관리자 대시보드" /><br><br>
      <sub>실시간 매출 통계 · 인기 상품 TOP 5<br>최근 구매 내역 · 전체 사용자 현황</sub>
    </td>
    <td align="center" width="33%">
      <b>📦 상품 관리</b><br><br>
      <img src="docs/gifs/상품관리.gif" width="220" alt="상품 관리 화면" /><br><br>
      <sub>상품 등록 (이미지 1-3장 업로드)<br>실시간 임베딩 생성 · 즉시 인식 적용 · 상품 삭제</sub>
    </td>
    <td align="center" width="33%">
      <b>🗂️ 전체 구매 내역</b><br><br>
      <img src="docs/gifs/구매내역관리.gif" width="220" alt="전체 구매 내역" /><br><br>
      <sub>전체 사용자 구매 기록 일괄 조회<br>사용자별 · 날짜별 필터링</sub>
    </td>
  </tr>
</table>

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
# 5. [컵밥 계열] EasyOCR Pending → 텍스트 정밀 보정 (6초 타임아웃)
# 6. Cooldown 체크 → 카운트 업데이트
```

---

## 📁 프로젝트 구조

```
EBRCS/
├── app/                   # FastAPI + React 웹앱
│   ├── backend/
│   │   ├── .venv/        # Backend 가상환경
│   │   ├── main.py       # FastAPI 앱
│   │   ├── config.py     # 설정 상수
│   │   ├── database.py   # SQLAlchemy 데이터베이스 설정
│   │   ├── models.py     # 데이터베이스 모델 (User, PurchaseHistory)
│   │   ├── st_shim.py    # checkout_core 호환 레이어 (Streamlit stub)
│   │   ├── routers/      # API 라우터
│   │   │   ├── auth.py       # 인증 (회원가입, 로그인)
│   │   │   ├── sessions.py   # 세션 관리
│   │   │   ├── checkout.py   # 실시간 체크아웃
│   │   │   ├── billing.py    # 장바구니 관리
│   │   │   ├── products.py   # 상품 관리
│   │   │   └── purchases.py  # 구매 내역
│   │   ├── services/     # 비즈니스 로직
│   │   │   └── session_manager.py  # CheckoutSession 관리
│   │   └── requirements.txt
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── pages/         # 페이지 컴포넌트
│   │   │   │   ├── HomePage.tsx         # 홈 / 대시보드
│   │   │   │   ├── CheckoutPage.tsx     # 실시간 체크아웃
│   │   │   │   ├── ValidatePage.tsx     # 영수증 확인
│   │   │   │   ├── ProductsPage.tsx     # 상품 관리 (관리자)
│   │   │   │   ├── MyPage.tsx           # 마이페이지
│   │   │   │   ├── AdminPurchasesPage.tsx  # 구매 내역 (관리자)
│   │   │   │   ├── LoginPage.tsx        # 로그인
│   │   │   │   └── SignupPage.tsx       # 회원가입
│   │   │   ├── stores/        # Zustand 상태 관리
│   │   │   │   ├── authStore.ts     # 인증 상태
│   │   │   │   └── sessionStore.ts  # 세션 상태
│   │   │   ├── api/           # API 클라이언트
│   │   │   └── App.tsx
│   │   ├── package.json
│   │   └── vite.config.ts
│   ├── setup_db.sh       # DB 스키마 초기화/검증
│   ├── setup_all.sh      # 환경+DB 원샷 세팅
│   ├── run_web.sh        # 개발 모드 실행
│   └── run_web_production.sh  # 프로덕션 실행
│
├── checkout_core/         # 공유 추론 엔진
│   ├── inference.py      # 모델 로딩 & 임베딩 추출
│   ├── frame_processor.py # 프레임 처리 & 상품 인식
│   └── counting.py       # 중복 방지 로직
│
├── data/                  # 모델 & 임베딩 데이터
│   ├── adapter_config.json    # LoRA 설정 (Git 포함)
│   ├── adapter_model.safetensors  # LoRA 가중치 (선택, 별도 수령)
│   ├── embeddings.npy     # 상품 임베딩 DB (서버 시작 필수, 별도 수령)
│   ├── labels.npy         # 상품 라벨 (서버 시작 필수, 별도 수령)
│   └── faiss_index.bin    # FAISS 인덱스 (서버 시작 시 자동 생성)
│
├── db/                    # DB 시드 관리
│   ├── export_full_seed.sh      # EC2 → 시드 파일 내보내기 (7개 테이블)
│   ├── import_full_seed.sh      # 시드 파일 → 로컬 MySQL 가져오기
│   ├── start_local_mysql.sh     # Docker MySQL 컨테이너 시작
│   ├── stop_local_mysql.sh      # Docker MySQL 컨테이너 종료
│   ├── docker-compose.mysql.yml # 로컬 MySQL 컨테이너 설정
│   ├── seeds/
│   │   └── full_seed_latest.sql.gz  # 전체 DB 시드 (7개 테이블)
│   └── README.md
│
├── nginx/                 # Nginx 리버스 프록시 설정
├── docs/                  # 프로젝트 문서
├── setup_aws_ec2.sh       # AWS EC2 자동 설정
├── setup_https.sh         # HTTPS 자동 설정
├── .env.example           # 환경 변수 템플릿
└── DB구축.md              # 로컬 DB 구축 파이프라인
```

---

## 🔌 API 엔드포인트

### 인증 (Authentication)
- `POST /api/auth/signup` - 회원가입 (username, password, name)
- `POST /api/auth/login` - 로그인 (JWT 토큰 발급, 7일 유효)
- `GET /api/auth/me` - 현재 사용자 정보 조회

### 세션 (Sessions)
- `POST /api/sessions` - 체크아웃 세션 생성
- `GET /api/sessions/{id}` - 세션 정보 조회
- `WebSocket /api/ws/checkout/{id}` - 실시간 카메라 스트리밍 (바이너리 JPEG → JSON 응답)
- `POST /api/sessions/{id}/ocr-cancel` - OCR 대기 상태 취소 (컵밥 정밀 인식 중 사용자 수동 취소)

### 결제 (Billing)
- `GET /api/sessions/{id}/billing` - 장바구니 조회
- `PUT /api/sessions/{id}/billing` - 장바구니 수정
- `POST /api/sessions/{id}/billing/confirm` - 구매 확정

### 상품 (Products)
- `POST /api/products` - 상품 등록 (이미지 업로드, 관리자 전용)
- `GET /api/products` - 상품 목록 조회
- `DELETE /api/products/{id}` - 상품 삭제 (관리자 전용)

### 구매 내역 (Purchases)
- `GET /api/purchases/my` - 내 구매 내역
- `GET /api/purchases/all` - 전체 구매 내역 (관리자 전용)
- `POST /api/purchases` - 구매 기록 생성
- `GET /api/purchases/dashboard` - 대시보드 통계 (관리자 전용)

### 인증 방식
모든 보호된 엔드포인트는 JWT Bearer 토큰 필요:
```bash
Authorization: Bearer <your_jwt_token>
```

---

## 🚀 시작하기

### 요구사항

- **Python**: 3.11+
- **Node.js**: 20.19.0+
- **Git**: 2.0+
- **CUDA** (선택): GPU 가속용

> **💡 크로스 플랫폼 지원**: Windows, macOS, Linux 모두 지원합니다!
> - Windows: `.bat` 배치 파일 사용
> - macOS/Linux: `.sh` 셸 스크립트 사용
> - 각 명령어는 OS별로 구분되어 있습니다 (🪟 Windows / 🍎 macOS / 🐧 Linux)

### 웹앱 실행

#### 🪟 Windows

```cmd
# 1. 환경 설정 (Backend + Frontend)
cd app
setup_venv.bat

# 2. DB 스키마 초기화/검증
setup_db.bat

# 3. 개발 모드 실행
run_web.bat

# (선택) 원샷 설정
# setup_all.bat
```

#### 🍎 macOS / 🐧 Linux

```bash
# 1. 환경 설정 (Backend + Frontend)
cd app
./setup_venv.sh

# 2. DB 스키마 초기화/검증
./setup_db.sh

# 3. 개발 모드 실행
./run_web.sh

# (선택) 원샷 설정
# ./setup_all.sh
```

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000/docs

#### DB 연결 설정 (SQLite / MySQL / Docker MySQL)

기본값은 SQLite(`data/ebrcs.db`)이며, `.env`의 `DATABASE_URL`을 설정하면 MySQL로 전환됩니다.

`DATABASE_URL` 변경 후에는 스키마를 다시 맞춰주세요:

**🪟 Windows**
```cmd
cd app
setup_db.bat
```

**🍎 macOS / 🐧 Linux**
```bash
cd app
./setup_db.sh
```

`.env` 예시:

```env
# SQLite (기본)
# DATABASE_URL=sqlite:///data/ebrcs.db

# MySQL (운영/공유 DB)
# DATABASE_URL=mysql+pymysql://<USER>:<PASSWORD>@127.0.0.1:3306/mydb

# Docker 로컬 MySQL (권장)
# DATABASE_URL=mysql+pymysql://ebrcs_app:ebrcs_pass@127.0.0.1:3307/mydb
```

#### Django 사용자용 매핑 (makemigrations / migrate와의 차이)

- 이 프로젝트는 Django ORM이 아니라 SQLAlchemy 기반입니다.
- `makemigrations`에 해당하는 자동 파일 생성 단계는 없습니다.
- `migrate`에 해당하는 단계는 `cd app && ./setup_db.sh` 입니다.
- 모델 변경 시에는 아래를 함께 수정해야 합니다:
  - `app/backend/models.py` (ORM 모델)
  - `app/backend/db_bootstrap.py` (DB bootstrap SQL)
- 서버 실행 시(`./run_web.sh`, `./run_web_production.sh`) DB check(`setup_db.sh --check`)를 먼저 수행하며, 필요하면 bootstrap을 자동 재시도합니다.

#### ✅ 로컬 DB 구축 파이프라인

자세한 절차는 [DB구축.md](DB구축.md) 참고.

퀵스타트:

```bash
# 1. 로컬 MySQL 컨테이너 기동 (Docker 필요)
./db/start_local_mysql.sh

# 2. .env에서 DATABASE_URL을 로컬 MySQL로 설정
# DATABASE_URL=mysql+pymysql://ebrcs_app:ebrcs_pass@127.0.0.1:3307/mydb

# 3. 스키마 생성 + 전체 시드 복원 (7개 테이블)
cd app && ./setup_venv.sh && ./setup_db.sh && cd ..
./db/import_full_seed.sh --seed ./db/seeds/full_seed_latest.sql.gz

# 4. 실행
cd app && ./run_web.sh
```

### 2️⃣ 관리자 계정 생성

웹앱을 처음 실행하면 일반 사용자 계정만 생성됩니다. 관리자 계정을 만들려면:

#### 방법 1: 일반 계정을 관리자로 변경 (권장)

**🪟 Windows**

```cmd
REM 1. 웹 UI에서 일반 계정 생성 (예: admin / password123)

REM 2. 데이터베이스에서 역할 변경
cd app
backend\.venv\Scripts\activate
python -c "from backend.database import SessionLocal; from backend import models; db = SessionLocal(); user = db.query(models.User).filter(models.User.username == 'admin').first(); user.role = 'admin' if user else None; db.commit() if user else None; print(f'✅ {user.username} 계정이 관리자로 변경되었습니다.') if user else print('❌ 사용자를 찾을 수 없습니다.'); db.close()"
```

**🍎 macOS / 🐧 Linux**

```bash
# 1. 웹 UI에서 일반 계정 생성 (예: admin / password123)

# 2. 데이터베이스에서 역할 변경
cd app
source backend/.venv/bin/activate
python -c "
from backend.database import SessionLocal
from backend import models

db = SessionLocal()
user = db.query(models.User).filter(models.User.username == 'admin').first()
if user:
    user.role = 'admin'
    db.commit()
    print(f'✅ {user.username} 계정이 관리자로 변경되었습니다.')
else:
    print('❌ 사용자를 찾을 수 없습니다.')
db.close()
"
```

#### 방법 2: SQLite 도구 사용

**🪟 Windows**

```cmd
REM 1. 웹 UI에서 일반 계정 생성 (예: admin / password123)

REM 2. SQLite CLI로 직접 수정 (sqlite3.exe 설치 필요)
sqlite3 data\ebrcs.db "UPDATE users SET role = 'admin' WHERE username = 'admin';"

REM 또는 대화형 모드
sqlite3 data\ebrcs.db
UPDATE users SET role = 'admin' WHERE username = 'admin';
.quit
```

**🍎 macOS / 🐧 Linux**

```bash
# 1. 웹 UI에서 일반 계정 생성 (예: admin / password123)

# 2. SQLite CLI로 직접 수정
sqlite3 data/ebrcs.db "UPDATE users SET role = 'admin' WHERE username = 'admin';"

# 또는 대화형 모드
sqlite3 data/ebrcs.db
UPDATE users SET role = 'admin' WHERE username = 'admin';
.quit
```

관리자로 로그인하면 대시보드, 상품 관리, 전체 구매 내역 조회 가능합니다.

---

## 📦 데이터 준비

> **⚠️ 중요**: `embeddings.npy`와 `labels.npy`는 **서버 시작 전에 반드시 존재해야** 합니다.
> 파일이 없으면 FastAPI 서버가 startup 시 `FileNotFoundError`로 크래시됩니다.

### 파일별 상태 정리

| 파일 | Git 포함 | 필수 여부 | 생성 방법 |
|------|---------|---------|---------|
| `adapter_config.json` | ✅ | 필수 | git에 포함 |
| `adapter_model.safetensors` | ❌ | **선택** | EC2/팀원에게 별도 수령 (없으면 기본 DINO 모델로 동작) |
| `embeddings.npy` | ❌ | **필수** | EC2/팀원에게 수령하거나 상품 임베딩 사전 생성 필요 |
| `labels.npy` | ❌ | **필수** | 동상 |
| `faiss_index.bin` | ❌ | 자동 | 서버 시작 시 embeddings.npy로부터 자동 생성 |
| `ebrcs.db` | ❌ | 자동 | setup_db.sh 실행 시 자동 생성 |

### embeddings.npy / labels.npy 확보 방법

현재 팀의 EC2에 임베딩 DB가 이미 구축되어 있으므로, 파일을 복사해서 사용합니다:

```bash
# EC2에서 로컬로 복사
scp -i your-key.pem ubuntu@YOUR_EC2_IP:~/ebrcs_streaming/data/embeddings.npy data/
scp -i your-key.pem ubuntu@YOUR_EC2_IP:~/ebrcs_streaming/data/labels.npy data/
```

파일이 준비된 후에는 서버 시작 시 FAISS 인덱스(`faiss_index.bin`)가 자동으로 생성됩니다.

### 상품 추가 (웹 UI)

서버 실행 중에는 웹 UI에서 상품을 추가할 수 있습니다. 추가 시 `embeddings.npy`, `labels.npy`, `faiss_index.bin`이 **증분 업데이트**되므로 서버 재시작이 불필요합니다.

**웹앱 실행 후**:
1. 브라우저에서 `http://localhost:5173` 접속
2. **"상품 등록"** 페이지 이동
3. 상품명 입력 + 이미지 1-3장 업로드
4. **즉시 인식 가능!** (서버 재시작 불필요)

---

## 🌐 배포

### AWS EC2 자동 배포

```bash
# EC2 Ubuntu 22.04 인스턴스에서
wget https://raw.githubusercontent.com/Smart-Cart-5/EBRCS/main/setup_aws_ec2.sh
chmod +x setup_aws_ec2.sh
./setup_aws_ec2.sh
```

스크립트가 자동으로:
1. Python 3.11, Node.js 20.19+ 설치
2. 저장소 클론
3. 가상환경 및 의존성 설치 (Backend + Frontend)
4. 실행 스크립트 권한 설정

### 프로덕션 실행

```bash
cd ebrcs_streaming/app
./setup_db.sh
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
- **EasyOCR** - 컵밥 계열 상품 텍스트 인식 (한국어 + 영어, GPU 가속)
- **PyTorch** - 딥러닝 프레임워크
- **Transformers** - HuggingFace 모델 로딩
- **PEFT** - LoRA 어댑터 적용

### Backend
- **FastAPI** - 고성능 비동기 API 프레임워크
- **Uvicorn** - ASGI 서버
- **WebSocket** - 실시간 카메라 스트리밍
- **SSE (Server-Sent Events)** - 비디오 처리 진행률
- **aiorwlock** - 비동기 Reader-Writer Lock
- **SQLAlchemy** - ORM (데이터베이스 추상화)
- **SQLite / MySQL** - 개발/운영 데이터베이스
- **python-jose** - JWT 토큰 생성/검증
- **bcrypt** - 비밀번호 해싱
- **Pydantic** - 요청/응답 데이터 검증

### Frontend
- **React 19** + TypeScript
- **Vite** - 빌드 도구
- **Tailwind CSS v4** - 스타일링
- **Zustand** - 상태 관리
- **TanStack Query** - 서버 상태 관리

### Computer Vision
- **OpenCV** - 이미지 처리 (CLAHE 대비 강화, 이미지 upscaling)
- **Background Subtraction (KNN)** - 동적 객체 탐지
- **ROI (Region of Interest)** - 관심 영역 설정
- **EasyOCR** - 실시간 상품 텍스트 인식 (OCR Pending 파이프라인)

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

`.env` 파일 설정 (`.env.example` 참고):

```bash
# HuggingFace 토큰 (모델 다운로드용)
HF_TOKEN=your_huggingface_token_here
HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here

# JWT 인증용 비밀 키 (랜덤 문자열 생성 권장)
SECRET_KEY=your_random_secret_key_here

# DB 연결 (미설정 시 SQLite 사용)
# DATABASE_URL=sqlite:///data/ebrcs.db
# DATABASE_URL=mysql+pymysql://<USER>:<PASSWORD>@127.0.0.1:3306/mydb
# DATABASE_URL=mysql+pymysql://ebrcs_app:ebrcs_pass@127.0.0.1:3307/mydb

# 선택 사항
# KMP_DUPLICATE_LIB_OK=TRUE  # macOS OpenMP 이슈 해결
```

**SECRET_KEY 생성 방법**:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
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

`checkout_core/frame_processor.py`:
```python
_OCR_PENDING_TIMEOUT_SEC = 6.0  # 컵밥 OCR 대기 최대 시간 (초)
# EasyOCR: easyocr.Reader(['ko', 'en'], gpu=True)
# 이미지 전처리: 200px 이하 2배 upscale + CLAHE 대비 강화
# 대상: 컵밥/컵라면 계열 25종 (CJ햇반컵반, 오뚜기컵밥, 동원컵밥 등)
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

### 2. Python 명령어 찾을 수 없음

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

### 3. 가상환경 활성화 오류

**Windows PowerShell 실행 정책 오류**:
```powershell
# PowerShell을 관리자 권한으로 실행 후
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**또는 CMD 사용** (PowerShell 대신):
```cmd
.venv\Scripts\activate.bat
```

### 4. CUDA Out of Memory
```python
# backend/config.py에서 디바이스 강제 지정
DEVICE = "cpu"  # GPU → CPU 전환
```

### 5. Frontend CORS 에러
```typescript
// frontend/vite.config.ts 확인
server: {
  proxy: {
    '/api': 'http://localhost:8000'
  }
}
```

### 6. Port 이미 사용 중 오류

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
- ✅ Python 3.11 + Node.js 20.19+
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
