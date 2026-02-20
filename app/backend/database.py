"""Database configuration and session management."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend import config

# Default to local SQLite. Set DATABASE_URL to switch to MySQL in production.
DEFAULT_SQLITE_URL = f"sqlite:///{config.DATA_DIR}/ebrcs.db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL).strip()

# SQLAlchemy expects driver-qualified MySQL URL.
# Supports legacy mysql:// style in .env by normalizing to mysql+pymysql://.
if DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)

engine_kwargs: dict = {
    "echo": False,  # Set to True for SQL debugging
    "pool_pre_ping": True,  # Reconnect stale DB connections safely
}

if DATABASE_URL.startswith("sqlite"):
    # SQLite needs single-thread check override in FastAPI context.
    engine_kwargs["connect_args"] = {"check_same_thread": False}

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
