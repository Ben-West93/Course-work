"""
SQLAlchemy engine, session factory and the declarative Base.

Postgres in Docker, SQLite everywhere else (local runs and the tests). Tables
are created at startup with create_all, there are no migrations yet.
"""

import logging
import time
from typing import Annotated

from fastapi import Depends
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings

log = logging.getLogger("uvicorn.error")

is_sqlite = settings.database_url.startswith("sqlite")

engine = create_engine(
    settings.database_url,
    # FastAPI runs sync endpoints in a threadpool, SQLite objects default to
    # the thread that made them
    connect_args={"check_same_thread": False} if is_sqlite else {},
    # Postgres restarting under the API would otherwise leave dead
    # connections in the pool and fail the next request
    pool_pre_ping=True,
)

if is_sqlite:
    # SQLite ignores ON DELETE CASCADE unless this is switched on per connection
    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# expire_on_commit=False keeps the attributes loaded after commit, so a route
# can return the object it just saved without another SELECT
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# one session per request, closed when the response is sent
DbSession = Annotated[Session, Depends(get_db)]


def init_db(attempts: int = 10, wait: float = 3.0):
    # models has to be imported for its tables to be on Base.metadata
    import models  # noqa: F401

    # Postgres restarting while the API starts used to end the backend for
    # good ("Application startup failed. Exiting."). Retry for ~30s first
    for attempt in range(1, attempts + 1):
        try:
            Base.metadata.create_all(engine)
            return
        except OperationalError as e:
            if attempt == attempts:
                raise
            log.warning("database not ready (attempt %d/%d): %s", attempt, attempts, str(e).splitlines()[0])
            time.sleep(wait)


def ping() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - any failure means the database is down
        return False
