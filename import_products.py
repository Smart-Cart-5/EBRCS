"""Import products from product_db_new.json into SQLite.

Usage:
    python import_products.py [--json PATH] [--reset]

This script:
1. Drops and recreates products/product_prices tables (with --reset)
2. Reads product_db_new.json (264 products with prices)
3. Inserts into products + product_prices (split schema)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DB_PATH = DATA_DIR / "ebrcs.db"
DEFAULT_JSON = Path(os.environ.get(
    "PRODUCT_JSON",
    r"C:\Users\jhj20\Downloads\product_db_new.json",
))

# ---------- Schema (matching db_bootstrap.py SQLITE_SCHEMA_STATEMENTS) ------

CREATE_PRODUCTS = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_no TEXT NOT NULL,
    barcd TEXT,
    product_name TEXT NOT NULL,
    company TEXT,
    volume TEXT,
    category_l TEXT,
    category_m TEXT,
    category_s TEXT,
    nutrition_info TEXT,
    src_meta_xml TEXT,
    dedup_key_type TEXT,
    dedup_key TEXT,
    created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
)
"""

CREATE_PRODUCTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_products_item_no ON products (item_no)",
    "CREATE INDEX IF NOT EXISTS idx_products_name ON products (product_name)",
    "CREATE INDEX IF NOT EXISTS idx_products_category ON products (category_l, category_m, category_s)",
    "CREATE INDEX IF NOT EXISTS idx_products_company ON products (company)",
]

CREATE_PRODUCT_PRICES = """
CREATE TABLE IF NOT EXISTS product_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    price INTEGER NOT NULL,
    currency TEXT NOT NULL DEFAULT 'KRW',
    source TEXT,
    checked_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    query_type TEXT,
    query_value TEXT,
    mall_name TEXT,
    match_title TEXT,
    created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE
)
"""

CREATE_PRICES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_product_prices_product_checked ON product_prices (product_id, checked_at)",
]


def reset_tables(conn: sqlite3.Connection) -> None:
    """Drop and recreate products/product_prices tables."""
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DROP TABLE IF EXISTS product_prices")
    conn.execute("DROP TABLE IF EXISTS products")
    conn.execute("PRAGMA foreign_keys = ON")

    conn.execute(CREATE_PRODUCTS)
    for idx in CREATE_PRODUCTS_INDEXES:
        conn.execute(idx)
    conn.execute(CREATE_PRODUCT_PRICES)
    for idx in CREATE_PRICES_INDEXES:
        conn.execute(idx)
    conn.commit()
    print("✅ Tables recreated (products, product_prices)")


def import_json(conn: sqlite3.Connection, json_path: Path) -> tuple[int, int]:
    """Load JSON and insert into products + product_prices."""
    with open(json_path, "r", encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    if not items:
        print("⚠️  JSON file is empty.")
        return 0, 0

    now = datetime.utcnow().isoformat()
    products_inserted = 0
    prices_inserted = 0

    for item in items:
        item_no = str(item.get("item_no", "")).strip()
        barcd = item.get("barcd") or None
        product_name = (item.get("product_name") or "").strip()
        if not item_no or not product_name:
            continue

        company = item.get("company") or None
        volume = item.get("volume") or None
        category_l = item.get("category_l") or None
        category_m = item.get("category_m") or None
        category_s = item.get("category_s") or None
        nutrition_info = item.get("nutrition_info") or None
        src_meta_xml = item.get("_src_meta_xml") or None
        dedup_key_type = item.get("_dedup_key_type") or None
        dedup_key = item.get("_dedup_key") or None

        # Insert product
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO products
                (item_no, barcd, product_name, company, volume,
                 category_l, category_m, category_s,
                 nutrition_info, src_meta_xml, dedup_key_type, dedup_key,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_no, barcd, product_name, company, volume,
                category_l, category_m, category_s,
                nutrition_info, src_meta_xml, dedup_key_type, dedup_key,
                now, now,
            ),
        )
        product_id = cur.lastrowid
        products_inserted += 1

        # Insert price if available
        price = item.get("price")
        if price is not None:
            checked_at = item.get("price_checked_at") or now
            source = item.get("price_source") or None
            query_type = item.get("price_query_type") or None
            query_value = item.get("price_query") or None
            mall_name = item.get("price_mall_name") or None
            match_title = item.get("price_match_title") or None

            conn.execute(
                """
                INSERT OR IGNORE INTO product_prices
                    (product_id, price, currency, source, checked_at,
                     query_type, query_value, mall_name, match_title, created_at)
                VALUES (?, ?, 'KRW', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id, int(price), source, checked_at,
                    query_type, query_value, mall_name, match_title, now,
                ),
            )
            prices_inserted += 1

    conn.commit()
    return products_inserted, prices_inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Import products into SQLite")
    parser.add_argument(
        "--json", type=Path, default=DEFAULT_JSON,
        help=f"Path to product_db_new.json (default: {DEFAULT_JSON})",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Drop and recreate tables before import",
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help=f"SQLite DB path (default: {DB_PATH})",
    )
    args = parser.parse_args()

    if not args.json.is_file():
        print(f"❌ JSON file not found: {args.json}")
        return 1

    print(f"📂 DB: {args.db}")
    print(f"📄 JSON: {args.json}")
    print(f"   Items in JSON: ", end="")
    with open(args.json, "r", encoding="utf-8") as f:
        count = len(json.load(f))
    print(count)

    conn = sqlite3.connect(str(args.db))
    conn.execute("PRAGMA foreign_keys = ON")

    if args.reset:
        reset_tables(conn)
    else:
        # Ensure tables exist (add missing columns via recreate if needed)
        conn.execute(CREATE_PRODUCTS)
        for idx in CREATE_PRODUCTS_INDEXES:
            conn.execute(idx)
        conn.execute(CREATE_PRODUCT_PRICES)
        for idx in CREATE_PRICES_INDEXES:
            conn.execute(idx)
        conn.commit()

    products_count, prices_count = import_json(conn, args.json)

    # Verify
    total_products = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    total_prices = conn.execute("SELECT COUNT(*) FROM product_prices").fetchone()[0]

    print(f"\n✅ Import complete!")
    print(f"   Products inserted: {products_count}")
    print(f"   Prices inserted:   {prices_count}")
    print(f"   Total products:    {total_products}")
    print(f"   Total prices:      {total_prices}")

    # Show sample
    print("\n📋 Sample products:")
    rows = conn.execute(
        """
        SELECT p.item_no, p.product_name, pp.price
        FROM products p
        LEFT JOIN product_prices pp ON pp.product_id = p.id
        ORDER BY p.id
        LIMIT 5
        """
    ).fetchall()
    for r in rows:
        print(f"   [{r[0]}] {r[1]} → {r[2]:,}원" if r[2] else f"   [{r[0]}] {r[1]} → (no price)")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
