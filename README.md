# EBRCS Web Branch

MySQL + FastAPI + React 기반 웹앱 개발 전용 브랜치입니다.

이 브랜치는 Streamlit 데모를 제외하고, 팀원이 로컬에서 웹앱 개발을 바로 이어갈 수 있는 구성만 남겼습니다.

## 핵심 구성

- `app/`: FastAPI 백엔드 + React(Vite) 프론트엔드
- `checkout_core/`: 공용 추론 엔진
- `db/`: MySQL Docker 실행 + 가격 시드 import/export 스크립트
- `data/`: 모델/임베딩 파일

## 요구사항

- Python 3.11+
- Node.js 20.19+
- Docker + Docker Compose
- Git
- Windows의 `db/*.bat` 사용 시: Git for Windows(bash 포함)

## 빠른 시작 (로컬 Docker MySQL)

### 1) `.env` 준비

```bash
cp .env.example .env
```

`HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN` 설정 후 사용하세요.

기본 DB는 로컬 Docker MySQL(3307)로 맞춰져 있습니다.

### 2) 로컬 MySQL 시작

- macOS / Linux

```bash
./db/start_local_mysql.sh
```

- Windows (CMD/PowerShell)

```bat
db\start_local_mysql.bat
```

### 3) 웹앱 의존성 설치

- macOS / Linux

```bash
cd app
./setup_venv.sh
```

- Windows

```bat
cd app
setup_venv.bat
```

### 4) 스키마 준비

- macOS / Linux

```bash
cd app
./setup_db.sh
./setup_db.sh --check
```

- Windows

```bat
cd app
setup_db.bat
setup_db.bat --check
```

### 5) 가격 시드 import (선택)

시드 파일 예시: `db/seeds/price_seed_latest.sql.gz`

- macOS / Linux

```bash
./db/import_price_seed.sh --seed ./db/seeds/price_seed_latest.sql.gz
```

- Windows

```bat
db\import_price_seed.bat --seed ./db/seeds/price_seed_latest.sql.gz
```

### 6) 개발 서버 실행

- macOS / Linux

```bash
cd app
./run_web.sh
```

- Windows

```bat
cd app
run_web.bat
```

접속:
- Frontend: `http://localhost:5173`
- Backend Docs: `http://localhost:8000/docs`

## DB 운영 스크립트

- 시드 export
  - macOS / Linux: `./db/export_price_seed.sh`
  - Windows: `db\export_price_seed.bat`
- DB 중지
  - macOS / Linux: `./db/stop_local_mysql.sh`
  - Windows: `db\stop_local_mysql.bat`
- DB 중지 + 볼륨 삭제
  - macOS / Linux: `./db/stop_local_mysql.sh --purge`
  - Windows: `db\stop_local_mysql.bat --purge`

자세한 시드 워크플로는 `db/README.md` 참고.

## 레거시 SQLite에서 사용자/구매내역 이관

현재 웹앱은 MySQL-only 정책입니다.

예전 `app/data/ebrcs.db` 데이터를 쓰던 개발자는 `users`, `purchase_history`를 한 번 이관한 후 MySQL만 사용하세요.

## 문제 해결

### 1) `DATABASE_URL is required`

`.env`의 `DATABASE_URL` 확인:

```env
DATABASE_URL=mysql+pymysql://ebrcs_app:ebrcs_pass@127.0.0.1:3307/item_db
```

### 2) `401 Unauthorized`

현재 DB에 계정이 없는 상태일 수 있습니다.
- 먼저 회원가입 후 로그인
- 관리자 권한이 필요하면 `users.role='admin'`으로 변경

### 3) `vite http proxy error ECONNREFUSED`

백엔드 리로드 타이밍에 일시적으로 발생할 수 있습니다.
지속 발생 시 `app/run_web.sh` 또는 `app/run_web.bat`를 재실행하세요.
