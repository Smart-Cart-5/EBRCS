# Database Migrations

이 디렉토리는 EBRCS Streaming 시스템의 데이터베이스 마이그레이션 스크립트를 포함합니다.

## 현재 구조

현재 시스템은 **파일 기반** (embeddings.npy, labels.npy, faiss_index.bin)으로 작동합니다.
이 마이그레이션은 **향후 DB 기반 시스템 전환**을 위한 준비 단계입니다.

## 마이그레이션 파일

### 001_init_schema.sql
- 사용자 인증 (users)
- 상품 관리 (products, product_images)
- 세션 관리 (checkout_sessions, billing_items)
- 주문 내역 (orders, order_items)

## PostgreSQL 설치 및 실행

### macOS (Homebrew)
```bash
brew install postgresql@15
brew services start postgresql@15

# 데이터베이스 생성
createdb ebrcs_streaming

# 마이그레이션 실행
psql -d ebrcs_streaming -f backend/migrations/001_init_schema.sql
```

### Docker (추천)
```bash
# docker-compose.yml에 이미 정의됨
docker compose up -d db

# 마이그레이션 실행
docker compose exec db psql -U ebrcs -d ebrcs_streaming -f /migrations/001_init_schema.sql
```

## 기본 계정

스크립트 실행 후 다음 계정이 생성됩니다:

| Username | Password | Role | 용도 |
|----------|----------|------|------|
| admin | admin123 | admin | 상품 등록 |
| user1 | user123 | user | 일반 체크아웃 |

⚠️ **프로덕션 환경에서는 반드시 비밀번호를 변경하세요!**

## 스키마 확인

```sql
-- 테이블 목록
\dt

-- users 테이블 구조
\d users

-- View 확인
SELECT * FROM v_active_sessions;
SELECT * FROM v_order_history;
```

## 향후 작업

현재 시스템은 DB 없이도 작동하지만, 다음 기능을 추가하려면 DB 연동이 필요합니다:

1. ✅ 사용자 로그인/인증
2. ✅ 관리자 권한 분리
3. ✅ 주문 내역 조회 (마이페이지)
4. ✅ 상품-임베딩 매핑 관리
5. ✅ 세션 영속성 (서버 재시작 시에도 유지)

## 주의사항

- 현재 `products.py`, `checkout.py`는 DB를 사용하지 않습니다
- DB 연동 구현은 별도 PR/브랜치에서 진행 예정
- 이 마이그레이션은 **스키마 정의**만 제공합니다
