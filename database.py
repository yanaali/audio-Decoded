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

# Render supplies a plain PostgreSQL URL; explicitly select our installed
# psycopg 3 driver instead of SQLAlchemy's version-dependent default.
if DATABASE_URL.startswith(("postgres://", "postgresql://")):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL.split("://", 1)[1]

Base = declarative_base()

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
    connect_args={"connect_timeout": 10},
)
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
