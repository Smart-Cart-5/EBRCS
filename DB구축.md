# DB 구축 및 로컬 실행 파이프라인

이 문서는 팀원이 저장소를 clone한 뒤, 가격 매칭까지 포함해 로컬에서 웹앱을 실행하는 절차만 정리합니다.

## 목표

- `db/seeds/price_seed_latest.sql.gz`만 전달받으면
- 로컬 Docker MySQL + 로컬 웹앱으로 동일한 가격 매칭 흐름 재현

## 사전 준비

- Git
- Docker + Docker Compose
- Python 3.11+
- Node.js 20.19+

## 1) 저장소 clone

```bash
git clone https://github.com/Smart-Cart-5/EBRCS.git
cd EBRCS
```

## 2) 환경 파일 준비

```bash
cp .env.example .env
```

`.env`에서 `DATABASE_URL`을 아래로 설정:

```env
DATABASE_URL=mysql+pymysql://ebrcs_app:ebrcs_pass@127.0.0.1:3307/item_db
```

## 3) 가격 시드 파일 배치

전달받은 파일을 아래 경로에 둡니다:

- `db/seeds/price_seed_latest.sql.gz`

## 4) 로컬 MySQL 컨테이너 기동

```bash
./db/start_local_mysql.sh
```

## 5) 앱 환경 + DB 스키마 준비

```bash
cd app
./setup_venv.sh
./setup_db.sh
cd ..
```

## 6) 가격 시드 임포트

```bash
./db/import_price_seed.sh --seed ./db/seeds/price_seed_latest.sql.gz
```

## 7) 웹앱 실행

```bash
cd app
./run_web.sh
```

접속:
- Frontend: `http://localhost:5173`
- Backend Docs: `http://127.0.0.1:8000/docs`

## 8) 종료

웹앱 종료:

```bash
cd app
./stop_web.sh
```

MySQL 컨테이너 종료:

```bash
./db/stop_local_mysql.sh
```

데이터까지 완전 삭제:

```bash
./db/stop_local_mysql.sh --purge
```
