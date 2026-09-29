from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, default="guest@audiodecoded.local")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    uploads = relationship("AudioUpload", back_populates="user")


class AudioUpload(Base):
    __tablename__ = "audio_uploads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=True)
    content_type = Column(String(100), nullable=True)
    size_bytes = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="uploads")
    results = relationship("AnalysisResult", back_populates="upload")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(Integer, ForeignKey("audio_uploads.id"), nullable=False)
    bpm = Column(String(32), nullable=False)
    key = Column(String(64), nullable=False)
    note = Column(Text, nullable=True)
    analyzed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    upload = relationship("AudioUpload", back_populates="results")
