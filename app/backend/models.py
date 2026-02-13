"""SQLAlchemy models for authentication and user management."""

from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Integer, JSON, String, Text, ForeignKey
from sqlalchemy.orm import relationship

from backend.database import Base


class User(Base):
    """User model for authentication."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(100))
    role = Column(String(20), default="user", nullable=False)  # 'user' or 'admin'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    purchase_history = relationship("PurchaseHistory", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, role={self.role})>"


class PurchaseHistory(Base):
    """Purchase history model."""

    __tablename__ = "purchase_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    items = Column(JSON, nullable=False)  # [{"name": "코카콜라", "count": 2, "price": 3000}, ...]
    total_amount = Column(Integer, nullable=False)  # Total price in won
    timestamp = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)  # Optional notes

    # Relationship
    user = relationship("User", back_populates="purchase_history")

    def __repr__(self):
        return f"<PurchaseHistory(id={self.id}, user_id={self.user_id}, total={self.total_amount})>"
