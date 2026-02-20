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
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        INDEX idx_products_item_no (item_no),
        INDEX idx_products_name (product_name)
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
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        INDEX idx_product_prices_product_checked (product_id, checked_at),
        CONSTRAINT fk_product_prices_product
            FOREIGN KEY (product_id) REFERENCES products(id)
            ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
)

SQLITE_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_no TEXT NOT NULL,
        barcd TEXT,
        product_name TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
        updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_products_item_no
    ON products (item_no)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_products_name
    ON products (product_name)
    """,
    """
    CREATE TABLE IF NOT EXISTS product_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL,
        price INTEGER NOT NULL,
        currency TEXT NOT NULL DEFAULT 'KRW',
        source TEXT,
        checked_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
        created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
        FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_product_prices_product_checked
    ON product_prices (product_id, checked_at)
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
