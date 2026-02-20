"""Database bootstrap/check helpers for reproducible local setup.

Creates the minimum schema needed by this repository:
- SQLAlchemy models: users, purchase_history
- Catalog tables used by pricing service: products, product_prices
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

# Table names the app expects to exist when DB integration is enabled.
REQUIRED_TABLES = ("users", "purchase_history", "products", "product_prices")

MYSQL_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS products (
        id BIGINT NOT NULL AUTO_INCREMENT,
        item_no VARCHAR(64) NOT NULL,
        barcd VARCHAR(64) NULL,
        product_name VARCHAR(255) NOT NULL,
        maker_name VARCHAR(255) NULL,
        capacity VARCHAR(64) NULL,
        lcls_name VARCHAR(128) NULL,
        mcls_name VARCHAR(128) NULL,
        scls_name VARCHAR(128) NULL,
        nutri_json LONGTEXT NULL,
        meta_xml_path TEXT NULL,
        match_key_type VARCHAR(32) NULL,
        match_key_value VARCHAR(255) NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        INDEX idx_products_item_no (item_no),
        INDEX idx_products_name (product_name),
        INDEX idx_products_barcd (barcd)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS product_prices (
        id BIGINT NOT NULL AUTO_INCREMENT,
        product_id BIGINT NOT NULL,
        price INT NOT NULL,
        currency VARCHAR(8) NOT NULL DEFAULT 'KRW',
        source VARCHAR(128) NULL,
        checked_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        match_key_type VARCHAR(32) NULL,
        match_key_value VARCHAR(255) NULL,
        mall_name VARCHAR(255) NULL,
        product_title TEXT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        INDEX idx_product_prices_product_checked (product_id, checked_at),
        CONSTRAINT fk_product_prices_product
            FOREIGN KEY (product_id) REFERENCES products(id)
            ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
)

# Backward-compatible columns for pre-existing DBs created with a minimal schema.
MYSQL_COMPAT_COLUMNS: dict[str, dict[str, str]] = {
    "products": {
        "maker_name": "VARCHAR(255) NULL",
        "capacity": "VARCHAR(64) NULL",
        "lcls_name": "VARCHAR(128) NULL",
        "mcls_name": "VARCHAR(128) NULL",
        "scls_name": "VARCHAR(128) NULL",
        "nutri_json": "LONGTEXT NULL",
        "meta_xml_path": "TEXT NULL",
        "match_key_type": "VARCHAR(32) NULL",
        "match_key_value": "VARCHAR(255) NULL",
    },
    "product_prices": {
        "match_key_type": "VARCHAR(32) NULL",
        "match_key_value": "VARCHAR(255) NULL",
        "mall_name": "VARCHAR(255) NULL",
        "product_title": "TEXT NULL",
    },
}


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

# Import models first so Base.metadata includes users/purchase_history.
from backend import models  # noqa: F401,E402

# Import after .env load so DATABASE_URL is read correctly.
from backend.database import Base, engine  # noqa: E402


def _catalog_schema_statements(dialect: str) -> tuple[str, ...]:
    if dialect in {"mysql", "mariadb"}:
        return MYSQL_SCHEMA_STATEMENTS
    raise RuntimeError(
        f"Unsupported DB backend for bootstrap: {dialect}. "
        "Use mysql/mariadb."
    )


def _inspect_tables() -> list[str]:
    inspector = inspect(engine)
    return sorted(inspector.get_table_names())


def _missing_tables(found_tables: list[str]) -> list[str]:
    found = set(found_tables)
    return [name for name in REQUIRED_TABLES if name not in found]


def _ensure_compat_columns() -> None:
    """Add seed-compatible columns for DBs bootstrapped with older minimal schema."""
    compat_map = MYSQL_COMPAT_COLUMNS

    with engine.begin() as conn:
        current_db = conn.execute(text("SELECT DATABASE()")).scalar()
        if not current_db:
            return

        rows = conn.execute(
            text(
                """
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = :schema_name
                  AND table_name IN ('products', 'product_prices')
                """
            ),
            {"schema_name": current_db},
        ).all()
        existing_map: dict[str, set[str]] = {}
        for table_name, column_name in rows:
            existing_map.setdefault(str(table_name), set()).add(str(column_name))

        for table, columns in compat_map.items():
            existing_cols = existing_map.get(table, set())
            for column, ddl in columns.items():
                if column in existing_cols:
                    continue
                conn.execute(text(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {ddl}"))

        # Older runs may have created products.nutri_json as JSON type; seed contains
        # non-JSON payloads for some rows, so keep this column as LONGTEXT.
        conn.execute(text("ALTER TABLE `products` MODIFY COLUMN `nutri_json` LONGTEXT NULL"))


def bootstrap_database() -> dict[str, object]:
    """Create required tables without touching existing data."""
    Base.metadata.create_all(bind=engine)

    dialect = engine.url.get_backend_name()
    statements = _catalog_schema_statements(dialect)

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

    _ensure_compat_columns()

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
        description="Bootstrap/check EBRCS DB schema (users, purchase_history, products, product_prices)."
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
