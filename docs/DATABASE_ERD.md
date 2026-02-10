# EBRCS Streaming Database ERD

## 전체 시스템 아키텍처

```mermaid
graph TB
    subgraph "Frontend (React)"
        U[User Interface]
        A[Admin Interface]
    end

    subgraph "Backend (FastAPI)"
        API[API Routes]
        AUTH[JWT Auth]
        SESS[Session Manager]
    end

    subgraph "AI/ML Pipeline"
        CAM[Camera/Video Input]
        PROC[Frame Processor]
        DINO[DINOv3 Model]
        CLIP[CLIP Model]
        EMB[Embedding Generator]
        FAISS[FAISS Index]
    end

    subgraph "Database (PostgreSQL)"
        DB[(PostgreSQL)]
    end

    subgraph "File Storage"
        FS[embeddings.npy<br/>labels.npy<br/>faiss_index.bin]
    end

    U -->|API Calls| API
    A -->|API Calls| API
    API --> AUTH
    API --> SESS
    API --> DB

    CAM --> PROC
    PROC --> EMB
    EMB --> DINO
    EMB --> CLIP
    EMB --> FAISS
    FAISS --> FS

    API --> FAISS
    DB <--> API
```

## Entity-Relationship Diagram

```mermaid
erDiagram
    USERS ||--o{ CHECKOUT_SESSIONS : creates
    USERS ||--o{ ORDERS : places
    USERS ||--o{ PRODUCTS : "manages (admin)"

    PRODUCTS ||--o{ PRODUCT_IMAGES : has
    PRODUCTS ||--o{ BILLING_ITEMS : "recognized as"
    PRODUCTS ||--o{ ORDER_ITEMS : contains

    CHECKOUT_SESSIONS ||--o{ BILLING_ITEMS : accumulates
    CHECKOUT_SESSIONS ||--|| ORDERS : "confirms to"

    ORDERS ||--o{ ORDER_ITEMS : includes

    PRODUCT_IMAGES ||--|| EMBEDDINGS : "maps to"

    USERS {
        uuid id PK
        varchar username UK
        varchar password_hash
        varchar role "user or admin"
        timestamp created_at
        timestamp last_login
    }

    PRODUCTS {
        uuid id PK
        varchar name
        text description
        decimal price "optional"
        uuid created_by FK
        timestamp created_at
        boolean is_active
    }

    PRODUCT_IMAGES {
        uuid id PK
        uuid product_id FK
        varchar image_path "S3 or local"
        integer embedding_id "index in npy file"
        timestamp created_at
    }

    EMBEDDINGS {
        integer id PK "array index"
        uuid product_image_id FK
        binary vector "in embeddings.npy"
        varchar label "in labels.npy"
    }

    CHECKOUT_SESSIONS {
        uuid id PK
        uuid user_id FK
        timestamp created_at
        timestamp last_active
        varchar status "active/completed/expired"
    }

    BILLING_ITEMS {
        uuid id PK
        uuid session_id FK
        uuid product_id FK
        integer quantity
        decimal avg_score "FAISS similarity"
        timestamp added_at
    }

    ORDERS {
        uuid id PK
        uuid user_id FK
        uuid session_id FK
        integer total_items
        decimal total_amount "optional"
        timestamp confirmed_at
        varchar payment_status "mock_paid"
    }

    ORDER_ITEMS {
        uuid id PK
        uuid order_id FK
        uuid product_id FK
        integer quantity
        decimal avg_score
        decimal unit_price "optional"
    }
```

## 데이터 플로우 다이어그램

### 1. 사용자 체크아웃 플로우

```mermaid
sequenceDiagram
    participant U as User (Frontend)
    participant API as FastAPI
    participant DB as PostgreSQL
    participant SESS as SessionManager
    participant AI as AI Pipeline
    participant FAISS as FAISS Index

    U->>API: POST /api/sessions (with JWT)
    API->>DB: INSERT INTO checkout_sessions
    API->>SESS: Create in-memory session
    API-->>U: session_id

    U->>API: WebSocket /api/ws/checkout/{id}
    loop Real-time Camera Feed
        U->>API: Send JPEG frame
        API->>AI: process_frame()
        AI->>FAISS: search(embedding, k=1)
        FAISS-->>AI: (product_idx, score)
        AI->>DB: SELECT product WHERE embedding_id=idx
        AI-->>API: product_name, score
        API->>SESS: Update billing_items
        API->>DB: UPSERT INTO billing_items
        API-->>U: WebSocket response
    end

    U->>API: GET /api/sessions/{id}/billing
    API->>DB: SELECT billing_items WHERE session_id
    API-->>U: Current cart

    U->>API: POST /api/sessions/{id}/billing/confirm
    API->>DB: BEGIN TRANSACTION
    API->>DB: INSERT INTO orders
    API->>DB: INSERT INTO order_items
    API->>DB: UPDATE checkout_sessions SET status='completed'
    API->>DB: COMMIT
    API-->>U: order_id
```

### 2. 관리자 상품 등록 플로우 (증분 업데이트)

```mermaid
sequenceDiagram
    participant A as Admin (Frontend)
    participant API as FastAPI
    participant DB as PostgreSQL
    participant AI as AI Pipeline
    participant FS as File System
    participant FAISS as FAISS Index

    A->>API: POST /api/products (with images + JWT role=admin)
    API->>AI: generate_embeddings(images)

    par Generate Embeddings
        AI->>AI: DINOv3 forward pass
        AI->>AI: CLIP forward pass
        AI->>AI: Concatenate + Normalize
    end

    API->>API: Acquire RWLock.writer

    API->>DB: BEGIN TRANSACTION
    API->>DB: INSERT INTO products
    API->>DB: INSERT INTO product_images (with embedding_id)
    API->>DB: COMMIT

    API->>FS: Load embeddings.npy
    API->>FS: Append new embeddings
    API->>FS: Save embeddings.npy
    API->>FS: Load labels.npy
    API->>FS: Append new labels
    API->>FS: Save labels.npy

    Note over API,FAISS: 증분 업데이트 (핵심!)
    API->>FAISS: index.add(new_weighted_vectors)
    API->>FS: Save faiss_index.bin

    API->>API: Update app_state (atomic swap)
    API->>API: Release RWLock.writer

    API-->>A: Success (product_id, total_embeddings)
```

### 3. 마이페이지 주문 내역 조회

```mermaid
sequenceDiagram
    participant U as User (Frontend)
    participant API as FastAPI
    participant DB as PostgreSQL

    U->>API: GET /api/orders (with JWT)
    API->>DB: SELECT orders WHERE user_id = jwt.user_id
    API->>DB: JOIN order_items ON orders.id
    API->>DB: JOIN products ON order_items.product_id
    DB-->>API: Order history with details
    API-->>U: [
        {
          order_id, confirmed_at, total_items,
          items: [{product_name, quantity, score}, ...]
        },
        ...
      ]
```

## 임베딩 파일 구조

```mermaid
graph LR
    subgraph "File System"
        E[embeddings.npy<br/>shape: (N, 1280)<br/>DINO 1024 + CLIP 256]
        L[labels.npy<br/>shape: (N,)<br/>product names]
        F[faiss_index.bin<br/>IndexFlatIP<br/>dimension: 1280]
    end

    subgraph "Database Mapping"
        PI[product_images<br/>embedding_id = row index]
    end

    E -.row index.-> PI
    L -.row index.-> PI
    F -.search result idx.-> PI
```

## 증분 업데이트 메커니즘

```mermaid
flowchart TD
    START[New Product Upload] --> GEN[Generate Embeddings<br/>DINOv3 + CLIP]
    GEN --> LOCK[Acquire RWLock Writer]

    LOCK --> LOAD[Load Existing<br/>embeddings.npy<br/>labels.npy]

    LOAD --> APPEND[Append New Data<br/>old_emb + new_emb<br/>old_lbl + new_lbl]

    APPEND --> SAVE[Save Updated Files<br/>embeddings.npy<br/>labels.npy]

    SAVE --> INCR{Incremental<br/>Update?}

    INCR -->|YES ⭐| ADD[index.add(new_weighted)<br/>O 1 Complexity]
    INCR -->|NO ❌| REBUILD[Rebuild Full Index<br/>O n Complexity]

    ADD --> PERSIST[faiss.write_index<br/>faiss_index.bin]
    REBUILD --> PERSIST

    PERSIST --> SWAP[Atomic Swap<br/>app_state.faiss_index<br/>app_state.labels]

    SWAP --> UNLOCK[Release RWLock]
    UNLOCK --> END[Return Success]

    style ADD fill:#90EE90
    style REBUILD fill:#FFB6C6
    style INCR fill:#FFD700
```

## 동시성 제어

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> Reading: Reader Lock
    Reading --> Idle: Release

    Idle --> Writing: Writer Lock
    Writing --> Idle: Release

    Reading --> Waiting: Writer Request
    Waiting --> Writing: All Readers Released

    note right of Reading
        Multiple readers allowed
        (Inference requests)
    end note

    note right of Writing
        Single writer only
        (Product registration)
    end note
```

---

## SQL 마이그레이션 스크립트

위 ERD를 구현하는 실제 SQL은 다음 파일에서 확인:
- `backend/migrations/001_init_schema.sql`

## 시각화 방법

이 파일은 Mermaid 다이어그램을 포함하고 있습니다:
- **GitHub/GitLab**: 자동 렌더링
- **VSCode**: Mermaid Preview 확장 설치
- **온라인**: https://mermaid.live 에서 코드 복사/붙여넣기
