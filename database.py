import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/audio_decoded",
)

Base = declarative_base()

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def initialize_db() -> bool:
    try:
        Base.metadata.create_all(bind=engine)
        return True
    except Exception:
        logger.warning("PostgreSQL is unavailable; continuing without persistence.", exc_info=True)
        return False


def get_db() -> Session | None:
    try:
        return SessionLocal()
    except Exception:
        logger.warning("Could not open a database session; continuing without persistence.", exc_info=True)
        return None
