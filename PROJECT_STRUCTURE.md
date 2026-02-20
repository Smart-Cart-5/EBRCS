# PROJECT_STRUCTURE (web branch)

웹앱 개발에 필요한 구조만 남긴 브랜치 기준 문서입니다.

```
EBRCS_streaming/
├── app/
│   ├── backend/                 # FastAPI
│   │   ├── main.py
│   │   ├── database.py          # MySQL-only 설정
│   │   ├── db_bootstrap.py      # 스키마 bootstrap/check
│   │   ├── models.py            # users, purchase_history
│   │   ├── routers/
│   │   └── services/
│   ├── frontend/                # React + Vite
│   ├── setup_venv.sh/.bat
│   ├── setup_db.sh/.bat
│   ├── run_web.sh/.bat
│   └── run_web_production.sh
├── checkout_core/               # 공용 추론 엔진
├── db/
│   ├── docker-compose.mysql.yml
│   ├── start_local_mysql.sh/.bat
│   ├── stop_local_mysql.sh/.bat
│   ├── import_price_seed.sh/.bat
│   ├── export_price_seed.sh/.bat
│   ├── seeds/
│   └── README.md
├── data/                        # 모델/임베딩 파일
├── docs/
│   └── AWS_DEPLOYMENT_GUIDE.md
├── .env.example
├── README.md
└── PROJECT_STRUCTURE.md
```

## 실행 기준

### macOS / Linux

```bash
cd app
./setup_venv.sh
./setup_db.sh
./run_web.sh
```

### Windows

```bat
cd app
setup_venv.bat
setup_db.bat
run_web.bat
```

## DB 기준

- 웹앱은 MySQL만 사용
- `.env`의 `DATABASE_URL` 필수
- 로컬 개발은 `db/start_local_mysql.*` 사용 권장
