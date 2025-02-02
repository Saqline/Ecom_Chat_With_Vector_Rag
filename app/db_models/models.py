import datetime
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Boolean, Column, DateTime, Integer, String, create_engine
from .db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    role = Column(String(50), default="customer", nullable=False)
    email_verified = Column(Boolean, default=False)
    ev_code = Column(String(6), nullable=True)
    ev_code_expire = Column(DateTime, nullable=True)
    fp_code = Column(String(6), nullable=True)
    fp_code_expire = Column(DateTime, nullable=True)
    access_token = Column(String(255), nullable=True)
    register_type = Column(String(55), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow) 