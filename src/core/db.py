from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, inspect, text
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

def add_missing_columns(bind, metadata) -> list[str]:
    """Add nullable columns that exist in the models but not yet in the database.

    `create_all()` only creates missing *tables*; it never alters an existing one, so a
    column added to a model later would be missing from any database created before it.
    Idempotent. Only nullable columns are handled, because adding a NOT NULL column with
    no default to a populated table would fail. Returns the "table.column" names added.
    """
    added: list[str] = []
    inspector = inspect(bind)
    preparer = bind.dialect.identifier_preparer
    with bind.begin() as conn:
        for table in metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                if not column.nullable:
                    logger.warning(
                        "column_missing_but_not_nullable_needs_manual_migration",
                        table=table.name,
                        column=column.name,
                    )
                    continue
                col_type = column.type.compile(dialect=bind.dialect)
                conn.execute(text(
                    f"ALTER TABLE {preparer.quote(table.name)} "
                    f"ADD COLUMN {preparer.quote(column.name)} {col_type}"
                ))
                if column.index:
                    index_name = f"ix_{table.name}_{column.name}"
                    conn.execute(text(
                        f"CREATE INDEX IF NOT EXISTS {preparer.quote(index_name)} "
                        f"ON {preparer.quote(table.name)} ({preparer.quote(column.name)})"
                    ))
                added.append(f"{table.name}.{column.name}")
                logger.info("db_column_added", table=table.name, column=column.name)
    return added


def create_all_tables() -> None:
    logger.info("creating_db_tables")
    # Import the data_logger Base that has PredictionLog registered.
    # core/db.py's own Base has no models attached, so we must use the one
    # from data_logger.models to ensure the prediction_logs table gets created.
    from src.data_logger.models import Base as DataLoggerBase  # noqa: PLC0415
    DataLoggerBase.metadata.create_all(bind=engine)
    add_missing_columns(engine, DataLoggerBase.metadata)
    logger.info("db_tables_created")
