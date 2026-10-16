"""
ORM models. MVP tables only: users and job_postings.

skills, user_skills, posting_skills and analyses from the design come with
the match score, after the ask loop works end to end.
"""

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def utcnow() -> datetime:
    """Now in UTC, for the timestamp columns."""
    return datetime.now(UTC)


# SQLite gives a new row the highest id again once that row is deleted, so a
# deleted account's unexpired token named the next account to register.
# AUTOINCREMENT stops that. Postgres ignores it, its ids are never reused
NEVER_REUSE_IDS = {"sqlite_autoincrement": True}
# two deletes of one row at once (a double click, two tabs) both answer 204,
# and the second finds nothing left to delete. That's expected, not worth the
# SAWarning SQLAlchemy logs for it
DELETE_MAY_FIND_NOTHING = {"confirm_deleted_rows": False}


class User(Base):
    """An account. Deleting it deletes its postings."""

    __tablename__ = "users"
    __table_args__ = NEVER_REUSE_IDS
    __mapper_args__ = DELETE_MAY_FIND_NOTHING

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # passive_deletes leaves it to the database's ON DELETE CASCADE instead of
    # loading every posting just to delete it
    postings: Mapped[list["JobPosting"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )


class JobPosting(Base):
    """A saved posting. Its text is the source of truth, ChromaDB only indexes it."""

    __tablename__ = "job_postings"
    # the same posting text saved twice would put duplicate chunks in every answer
    __table_args__ = (UniqueConstraint("user_id", "content_hash", name="uq_posting_user_hash"), NEVER_REUSE_IDS)
    __mapper_args__ = DELETE_MAY_FIND_NOTHING

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    company: Mapped[str] = mapped_column(String(200))
    location: Mapped[str | None] = mapped_column(String(200))
    url: Mapped[str | None] = mapped_column(String(500))
    # pasted, upload or sample
    source: Mapped[str] = mapped_column(String(10))
    text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    # set when the chunks are in ChromaDB. None means search won't find it
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="postings")

    @property
    def indexed(self) -> bool:
        """Whether its chunks are in ChromaDB, so search can find it."""
        return self.indexed_at is not None
