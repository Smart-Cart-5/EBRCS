"""Database configuration and session management."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# MySQL-only policy: DATABASE_URL must be explicitly configured.
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is required. "
        "Example: mysql+pymysql://ebrcs_app:ebrcs_pass@127.0.0.1:3307/item_db"
    )

# SQLAlchemy expects driver-qualified URL for PyMySQL.
# Support legacy mysql:// style by normalizing to mysql+pymysql://.
if DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)
elif DATABASE_URL.startswith("mariadb://"):
    DATABASE_URL = DATABASE_URL.replace("mariadb://", "mysql+pymysql://", 1)

if not DATABASE_URL.startswith("mysql+pymysql://"):
    raise RuntimeError(
        "Unsupported DATABASE_URL backend. "
        "Use mysql+pymysql://<USER>:<PASSWORD>@<HOST>:<PORT>/<DB_NAME>"
    )

engine_kwargs: dict = {
    "echo": False,  # Set to True for SQL debugging
    "pool_pre_ping": True,  # Reconnect stale DB connections safely
}

# Create engine
engine = create_engine(DATABASE_URL, **engine_kwargs)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for declarative models
Base = declarative_base()


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
