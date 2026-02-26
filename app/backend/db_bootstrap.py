"""Database bootstrap/check helpers for reproducible local setup.

Creates the minimum schema needed by this repository:
- SQLAlchemy models: users, purchase_history
- Catalog tables used by pricing service: products, product_prices, product_discounts
- Store layout mapping tables: store_corners, category_corner_map
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

# Table names the app expects to exist when DB integration is enabled.
REQUIRED_TABLES = (
    "users",
    "purchase_history",
    "products",
    "product_prices",
    "product_discounts",
    "store_corners",
    "category_corner_map",
)

# Default category_l -> corner_no assignments used when first seeding mappings.
DEFAULT_CATEGORY_CORNERS: tuple[tuple[str, int], ...] = (
    ("과자", 1),
    ("면류", 2),
    ("상온HMR", 3),
    ("소스", 4),
    ("유제품", 5),
    ("음료", 6),
    ("통조림/안주", 7),
)

# Schema mirrors the EC2 production DB (mydb) exactly.
MYSQL_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS products (
        id             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        item_no        VARCHAR(32)     NOT NULL,
        barcd          VARCHAR(32)     DEFAULT NULL,
        product_name   VARCHAR(255)    NOT NULL,
        company        VARCHAR(128)    DEFAULT NULL,
        volume         VARCHAR(64)     DEFAULT NULL,
        category_l     VARCHAR(64)     DEFAULT NULL,
        category_m     VARCHAR(64)     DEFAULT NULL,
        category_s     VARCHAR(64)     DEFAULT NULL,
        nutrition_info JSON            DEFAULT NULL,
        src_meta_xml   VARCHAR(512)    DEFAULT NULL,
        dedup_key_type VARCHAR(32)     DEFAULT NULL,
        dedup_key      VARCHAR(64)     DEFAULT NULL,
        created_at     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        picture        VARCHAR(255)    DEFAULT NULL,
        PRIMARY KEY (id),
        UNIQUE KEY uq_products_barcd (barcd),
        KEY idx_products_item_no (item_no),
        KEY idx_products_product_name (product_name),
        KEY idx_products_category (category_l, category_m, category_s),
        KEY idx_products_company (company),
        FULLTEXT KEY ft_products_name_company (product_name, company)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS product_prices (
        id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        product_id  BIGINT UNSIGNED NOT NULL,
        price       INT             NOT NULL,
        currency    CHAR(3)         NOT NULL DEFAULT 'KRW',
        source      VARCHAR(64)     DEFAULT NULL,
        checked_at  DATETIME(6)     NOT NULL,
        query_type  VARCHAR(32)     DEFAULT NULL,
        query_value VARCHAR(255)    DEFAULT NULL,
        mall_name   VARCHAR(128)    DEFAULT NULL,
        match_title VARCHAR(512)    DEFAULT NULL,
        created_at  TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uq_product_prices_snapshot (product_id, checked_at, source, price),
        KEY idx_product_prices_product_checked (product_id, checked_at DESC),
        KEY idx_product_prices_checked_at (checked_at DESC),
        KEY idx_product_prices_source_checked (source, checked_at DESC),
        CONSTRAINT fk_product_prices_product_id
            FOREIGN KEY (product_id) REFERENCES products(id)
            ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS product_discounts (
        id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        product_price_id BIGINT UNSIGNED NOT NULL,
        is_discounted    TINYINT(1)      NOT NULL DEFAULT 0,
        discount_rate    DECIMAL(5,2)    DEFAULT NULL,
        discount_amount  INT             DEFAULT NULL,
        created_at       TIMESTAMP       NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at       TIMESTAMP       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uq_product_discounts_price (product_price_id),
        KEY idx_product_discounts_is_discounted (is_discounted),
        CONSTRAINT fk_product_discounts_price
            FOREIGN KEY (product_price_id) REFERENCES product_prices(id)
            ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS store_corners (
        id          BIGINT   NOT NULL AUTO_INCREMENT,
        corner_no   INT      NOT NULL,
        corner_name VARCHAR(100) NULL,
        created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uq_store_corners_corner_no (corner_no)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS category_corner_map (
        id         BIGINT       NOT NULL AUTO_INCREMENT,
        category_l VARCHAR(255) NOT NULL,
        corner_id  BIGINT       NOT NULL,
        created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uq_category_corner_map_category_l (category_l),
        INDEX idx_category_corner_map_corner_id (corner_id),
        CONSTRAINT fk_category_corner_map_corner
            FOREIGN KEY (corner_id) REFERENCES store_corners(id)
            ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
)

SQLITE_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS products (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        item_no        TEXT    NOT NULL,
        barcd          TEXT    DEFAULT NULL,
        product_name   TEXT    NOT NULL,
        company        TEXT    DEFAULT NULL,
        volume         TEXT    DEFAULT NULL,
        category_l     TEXT    DEFAULT NULL,
        category_m     TEXT    DEFAULT NULL,
        category_s     TEXT    DEFAULT NULL,
        nutrition_info TEXT    DEFAULT NULL,
        src_meta_xml   TEXT    DEFAULT NULL,
        dedup_key_type TEXT    DEFAULT NULL,
        dedup_key      TEXT    DEFAULT NULL,
        created_at     TEXT    NOT NULL DEFAULT (CURRENT_TIMESTAMP),
        updated_at     TEXT    NOT NULL DEFAULT (CURRENT_TIMESTAMP),
        picture        TEXT    DEFAULT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_products_item_no ON products (item_no)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_products_product_name ON products (product_name)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_products_category ON products (category_l, category_m, category_s)
    """,
    """
    CREATE TABLE IF NOT EXISTS product_prices (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id  INTEGER NOT NULL,
        price       INTEGER NOT NULL,
        currency    TEXT    NOT NULL DEFAULT 'KRW',
        source      TEXT    DEFAULT NULL,
        checked_at  TEXT    NOT NULL,
        query_type  TEXT    DEFAULT NULL,
        query_value TEXT    DEFAULT NULL,
        mall_name   TEXT    DEFAULT NULL,
        match_title TEXT    DEFAULT NULL,
        created_at  TEXT    NOT NULL DEFAULT (CURRENT_TIMESTAMP),
        FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_product_prices_product_checked
    ON product_prices (product_id, checked_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS product_discounts (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        product_price_id INTEGER NOT NULL UNIQUE,
        is_discounted    INTEGER NOT NULL DEFAULT 0,
        discount_rate    REAL    DEFAULT NULL,
        discount_amount  INTEGER DEFAULT NULL,
        created_at       TEXT    DEFAULT (CURRENT_TIMESTAMP),
        updated_at       TEXT    DEFAULT (CURRENT_TIMESTAMP),
        FOREIGN KEY(product_price_id) REFERENCES product_prices(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_product_discounts_is_discounted
    ON product_discounts (is_discounted)
    """,
    """
    CREATE TABLE IF NOT EXISTS store_corners (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        corner_no  INTEGER NOT NULL UNIQUE,
        corner_name TEXT,
        created_at TEXT    NOT NULL DEFAULT (CURRENT_TIMESTAMP),
        updated_at TEXT    NOT NULL DEFAULT (CURRENT_TIMESTAMP)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS category_corner_map (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        category_l TEXT    NOT NULL UNIQUE,
        corner_id  INTEGER NOT NULL,
        created_at TEXT    NOT NULL DEFAULT (CURRENT_TIMESTAMP),
        updated_at TEXT    NOT NULL DEFAULT (CURRENT_TIMESTAMP),
        FOREIGN KEY(corner_id) REFERENCES store_corners(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_category_corner_map_corner_id
    ON category_corner_map (corner_id)
    """,
)


def _load_env_file() -> None:
    """Load PROJECT_ROOT/.env into os.environ if present.

    Keeps existing environment values as-is to avoid overriding explicit exports.
    """
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if not env_path.is_file():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()

# Import after .env load so DATABASE_URL is read correctly.
# Ensure SQLAlchemy model metadata is registered before create_all().
from backend import models  # noqa: F401,E402
from backend.database import Base, engine  # noqa: E402


def _catalog_schema_statements(dialect: str) -> tuple[str, ...]:
    if dialect == "sqlite":
        return SQLITE_SCHEMA_STATEMENTS
    if dialect in {"mysql", "mariadb"}:
        return MYSQL_SCHEMA_STATEMENTS
    raise RuntimeError(
        f"Unsupported DB backend for bootstrap: {dialect}. "
        "Use sqlite or mysql/mariadb."
    )


def _inspect_tables() -> list[str]:
    inspector = inspect(engine)
    return sorted(inspector.get_table_names())


def _missing_tables(found_tables: list[str]) -> list[str]:
    found = set(found_tables)
    return [name for name in REQUIRED_TABLES if name not in found]


def _seed_corner_mappings() -> None:
    """Seed category->corner mappings from products.category_l if missing.

    Existing mappings are preserved. Missing categories are assigned to the next
    available corner number while honoring DEFAULT_CATEGORY_CORNERS first.
    """
    with engine.begin() as conn:
        products_columns = {
            str(col.get("name") or "").strip()
            for col in inspect(conn).get_columns("products")
            if col.get("name")
        }
        if "category_l" not in products_columns:
            return

        category_rows = conn.execute(
            text(
                "SELECT DISTINCT category_l "
                "FROM products "
                "WHERE category_l IS NOT NULL AND TRIM(category_l) <> '' "
                "ORDER BY category_l"
            )
        ).mappings().all()
        categories = [str(row["category_l"]).strip() for row in category_rows if row.get("category_l")]
        if not categories:
            return

        assigned: dict[str, int] = {}
        for category, corner_no in DEFAULT_CATEGORY_CORNERS:
            if category in categories:
                assigned[category] = int(corner_no)

        used_corner_nos = set(assigned.values())
        next_corner_no = 1
        for category in categories:
            if category in assigned:
                continue
            while next_corner_no in used_corner_nos:
                next_corner_no += 1
            assigned[category] = next_corner_no
            used_corner_nos.add(next_corner_no)
            next_corner_no += 1

        corner_id_by_no: dict[int, int] = {}
        existing_corners = conn.execute(
            text("SELECT id, corner_no FROM store_corners")
        ).mappings().all()
        for row in existing_corners:
            corner_id_by_no[int(row["corner_no"])] = int(row["id"])

        for corner_no in sorted(set(assigned.values())):
            if corner_no in corner_id_by_no:
                continue
            conn.execute(
                text(
                    "INSERT INTO store_corners (corner_no, corner_name, created_at, updated_at) "
                    "VALUES (:corner_no, :corner_name, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {
                    "corner_no": int(corner_no),
                    "corner_name": f"{int(corner_no)}번 코너",
                },
            )
            corner_row = conn.execute(
                text("SELECT id FROM store_corners WHERE corner_no = :corner_no LIMIT 1"),
                {"corner_no": int(corner_no)},
            ).mappings().first()
            if corner_row and corner_row.get("id") is not None:
                corner_id_by_no[int(corner_no)] = int(corner_row["id"])

        existing_maps = conn.execute(
            text("SELECT category_l FROM category_corner_map")
        ).mappings().all()
        mapped_categories = {
            str(row["category_l"]).strip()
            for row in existing_maps
            if row.get("category_l")
        }

        for category, corner_no in assigned.items():
            if category in mapped_categories:
                continue
            corner_id = corner_id_by_no.get(int(corner_no))
            if not corner_id:
                continue
            conn.execute(
                text(
                    "INSERT INTO category_corner_map (category_l, corner_id, created_at, updated_at) "
                    "VALUES (:category_l, :corner_id, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {
                    "category_l": category,
                    "corner_id": int(corner_id),
                },
            )


def bootstrap_database() -> dict[str, object]:
    """Create required tables without touching existing data."""
    Base.metadata.create_all(bind=engine)

    dialect = engine.url.get_backend_name()
    statements = _catalog_schema_statements(dialect)

    with engine.begin() as conn:
        if dialect == "sqlite":
            conn.execute(text("PRAGMA foreign_keys = ON"))
        for stmt in statements:
            conn.execute(text(stmt))

    _seed_corner_mappings()

    tables = _inspect_tables()
    return {
        "backend": dialect,
        "tables": tables,
        "missing_tables": _missing_tables(tables),
    }


def check_database() -> dict[str, object]:
    """Check whether required tables already exist."""
    dialect = engine.url.get_backend_name()
    tables = _inspect_tables()
    return {
        "backend": dialect,
        "tables": tables,
        "missing_tables": _missing_tables(tables),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap/check EBRCS DB schema "
            "(users, purchase_history, products, product_prices, product_discounts, store_corners, category_corner_map)."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check only (do not create tables).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output.",
    )
    args = parser.parse_args()

    try:
        result = check_database() if args.check else bootstrap_database()
    except SQLAlchemyError as exc:
        print(f"❌ Database error: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1

    if not args.quiet:
        mode = "Check" if args.check else "Bootstrap"
        print(f"{mode} complete (backend: {result['backend']})")
        print(f"Tables: {', '.join(result['tables']) or '(none)'}")

    missing = result["missing_tables"]
    if missing:
        print(f"❌ Missing required tables: {', '.join(missing)}")
        return 1

    print("✅ Required tables are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
