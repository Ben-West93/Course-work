"""Pydantic request and response models for every route."""

import re
import unicodedata
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

USERNAME = re.compile(r"^[a-z0-9_.-]+$")

# ids are Postgres INTEGER columns. A bigger number in a path or a filter made
# Postgres raise "integer out of range", which surfaced as a 500
MAX_ID = 2**31 - 1
Id = Annotated[int, Field(ge=1, le=MAX_ID, strict=True)]

# typos like {"titel": ...} or a smuggled "user_id" are a 422, not silently ignored
REQUEST = ConfigDict(extra="forbid")


def drop_control(value: str) -> str:
    # Postgres can't store NUL in a text column (it was a 500), and other
    # control characters only get in the way of chunking. Newlines and tabs stay
    return "".join(ch for ch in value if ch in "\n\t" or unicodedata.category(ch) != "Cc")


def visible_count(value: str) -> int:
    # strip() leaves zero-width spaces behind, these don't count
    return sum(ch.isprintable() and not ch.isspace() for ch in value)


# ── Auth ───────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    model_config = REQUEST

    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username", mode="before")
    @classmethod
    def normalise_username(cls, v):
        # stored lowercase so "Ben" and "ben" can't both register
        if isinstance(v, str):
            v = v.strip().lower()
            if v and not USERNAME.match(v):
                raise ValueError("username can only use letters, digits, '_', '.' and '-'")
        return v


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


# ── Job postings ───────────────────────────────────────────────────────────

class JobFields(BaseModel):
    model_config = REQUEST

    @field_validator("title", "company", "location", "url", mode="before", check_fields=False)
    @classmethod
    def one_line(cls, v):
        # chunks are prefixed with "title at company", a newline in either
        # would break that header apart
        if isinstance(v, str):
            v = " ".join(drop_control(v).split())
            if v and not visible_count(v):
                raise ValueError("must contain visible text")
        return v

    @field_validator("location", mode="after", check_fields=False)
    @classmethod
    def blank_is_none(cls, v):
        return v or None

    @field_validator("url", mode="after", check_fields=False)
    @classmethod
    def web_link(cls, v):
        # the UI may turn it into a link one day, javascript: and file: never belong there
        if v and not v.lower().startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return v or None

    @field_validator("text", mode="before", check_fields=False)
    @classmethod
    def clean_text(cls, v):
        if isinstance(v, str):
            v = drop_control(v.replace("\r\n", "\n").replace("\r", "\n")).strip()
        return v

    @field_validator("text", mode="after", check_fields=False)
    @classmethod
    def readable_text(cls, v):
        # 250 zero-width spaces passed the length check and got indexed
        if v is not None and visible_count(v) < 100:
            raise ValueError("doesn't look like a job posting, it has fewer than 100 visible characters")
        return v


class JobCreate(JobFields):
    title: str = Field(min_length=1, max_length=200)
    company: str = Field(min_length=1, max_length=200)
    location: str | None = Field(None, max_length=200)
    url: str | None = Field(None, max_length=500)
    source: Literal["pasted", "upload"] = "pasted"
    # under 200 characters isn't a posting, over 20,000 is usually a whole page dump
    text: str = Field(min_length=200, max_length=20_000)


class JobUpdate(JobFields):
    title: str | None = Field(None, min_length=1, max_length=200)
    company: str | None = Field(None, min_length=1, max_length=200)
    location: str | None = Field(None, max_length=200)
    url: str | None = Field(None, max_length=500)
    text: str | None = Field(None, min_length=200, max_length=20_000)

    # leaving a field out keeps it. An explicit null would hit NOT NULL in the database
    @field_validator("title", "company", "text", mode="after")
    @classmethod
    def not_null(cls, v):
        if v is None:
            raise ValueError("can't be null, leave it out to keep the current value")
        return v


class JobSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    company: str
    location: str | None
    source: str
    indexed: bool
    created_at: datetime


class JobOut(JobSummary):
    url: str | None
    text: str
    indexed_at: datetime | None
    updated_at: datetime


class SamplesResult(BaseModel):
    added: int
    skipped: int


class ReindexResult(BaseModel):
    indexed: int
    failed: list[int]


# ── Ask ────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"question": "Which of my saved jobs ask for Docker experience?", "n_results": 5},
    ]})

    question: str = Field(min_length=3, max_length=500)
    # strict ids: "1" is a 422, not quietly turned into 1
    job_ids: list[Id] | None = Field(None, min_length=1, max_length=50)
    # strict, otherwise pydantic turns true into 1 and "5" into 5
    n_results: int = Field(5, ge=1, le=10, strict=True)

    @field_validator("question", mode="before")
    @classmethod
    def not_blank(cls, v):
        if isinstance(v, str):
            v = drop_control(v).strip()
            if not visible_count(v):
                raise ValueError("question must not be empty")
        return v


class Source(BaseModel):
    n: int
    job_id: int
    title: str
    company: str
    chunk: int
    excerpt: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    sources: list[Source]
    # None when no LLM call was made (nothing saved or nothing close enough)
    model: str | None
    # the answer hit the token cap and was cut off
    truncated: bool = False


# ── Health ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    database: Literal["ok", "error"]
    chroma: Literal["ok", "error"]
    ollama: Literal["connected", "disconnected"]
    model_ready: bool
    model: str
