from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.core.config import get_settings
from src.core.exceptions import DatabaseError
from src.core.logging import get_logger

logger = get_logger(__name__)

def build_engine():
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_pre_ping=True, 
        echo=(settings.log_level == "DEBUG"),
    )
    if settings.log_level == "DEBUG":
        @event.listens_for(engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            logger.debug("db_connection_acquired")
    return engine


engine = build_engine()
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

class Base(DeclarativeBase):
    pass

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("db_session_error", error=str(exc), exc_info=True)
        raise DatabaseError(f"Database operation failed: {exc}") from exc
    finally:
        db.close()

@contextmanager
def db_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("db_session_error", error=str(exc), exc_info=True)
        raise DatabaseError(f"Database operation failed: {exc}") from exc
    finally:
        db.close()

def check_db_connection() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("db_health_check_failed", error=str(exc))
        return False

def create_all_tables() -> None:
    logger.info("creating_db_tables")
    # Import the data_logger Base that has PredictionLog registered.
    # core/db.py's own Base has no models attached, so we must use the one
    # from data_logger.models to ensure the prediction_logs table gets created.
    from src.data_logger.models import Base as DataLoggerBase  # noqa: PLC0415
    DataLoggerBase.metadata.create_all(bind=engine)
    logger.info("db_tables_created")