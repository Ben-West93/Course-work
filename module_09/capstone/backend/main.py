"""
JobFit API.

    uvicorn main:app --reload --port 8000     # SQLite + ./chroma, Ollama on localhost:11434

Swagger UI at http://localhost:8000/docs. Register, then use Authorize with the
same username and password.

Routes are thin: validation is in schemas.py, auth in auth.py, and the RAG
steps (chunk, index, search, prompt, generate, score) in rag.py. The README's
"How a question is answered" section walks through /ask step by step, and
walkthrough.py runs it against the live services.

Status codes every route can return: 401 bad or missing token, 404 not found
(or not yours), 409 duplicate, 422 invalid input, 503 a service the API needs
(Postgres, ChromaDB, Ollama) is down. A 500 is a bug.
"""

import hashlib
import math
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError
from sqlalchemy.orm import Session

import auth
import database
import rag
from auth import CurrentUser
from config import settings
from database import DbSession
from models import JobPosting, User
from schemas import (
    MAX_ID,
    AskRequest,
    AskResponse,
    HealthResponse,
    JobCreate,
    JobOut,
    JobSummary,
    JobUpdate,
    ReindexResult,
    SamplesResult,
    Source,
    Token,
    UserCreate,
    UserOut,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # no migrations yet, create_all only adds missing tables
    database.init_db()
    yield


app = FastAPI(
    title="JobFit API",
    version="0.1",
    description=(
        "Saves job postings, indexes them in ChromaDB and answers questions about them "
        f"with {settings.model_name} through Ollama, citing the posting lines it used."
    ),
    lifespan=lifespan,
    debug=settings.debug,
)

# a path id past Postgres INTEGER was a 500 ("integer out of range")
JobId = Annotated[int, Path(ge=1, le=MAX_ID)]

NO_POSTINGS_ANSWER = "You haven't saved any job postings yet. Add one on the Jobs page, or load the samples."
NO_MATCH_ANSWER = (
    "None of your saved postings say anything close enough to this question to answer it. "
    "Try asking about a skill, a requirement or a company by name."
)


def content_hash(text: str) -> str:
    # whitespace and case only, so re-pasting the same posting is caught but an edited one isn't
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def get_owned_job(job_id: int, user: User, db: Session) -> JobPosting:
    job = db.get(JobPosting, job_id)
    # someone else's posting is a 404 too, so ids can't be probed
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Job posting not found")
    return job


def try_index(job: JobPosting, db: Session) -> bool:
    """Index the posting and stamp indexed_at. A Chroma failure leaves it saved but unindexed."""
    try:
        rag.index_posting(job)
    except Exception as e:  # noqa: BLE001 - Postgres is the source of truth, reindex fixes it later
        rag.log.warning("indexing job %s failed: %r", job.id, e)
        job.indexed_at = None
    else:
        job.indexed_at = datetime.now(UTC)
    db.commit()
    return job.indexed


# FastAPI's default 422 handler echoes the input back, and NaN or a lone
# surrogate in it can't be written out as JSON
def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": json_safe(jsonable_encoder(exc.errors()))})


# Postgres stopped or restarting. Every route that touches the database was a
# bare 500 with a traceback; it's a dependency being down, so 503 like Ollama
@app.exception_handler(OperationalError)
@app.exception_handler(InterfaceError)
async def database_unavailable(_request: Request, exc: Exception):
    rag.log.warning("database unavailable: %s", str(exc).splitlines()[0])
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, responses={503: {"description": "The database is down."}})
def health(response: Response):
    # ok: everything up. degraded: logins and CRUD work but search or answers
    # don't (Chroma broken, Ollama down, model not pulled yet). down: no database
    db_ok = database.ping()
    try:
        rag.count_chunks()
        chroma_ok = True
    except Exception as e:  # noqa: BLE001
        rag.log.warning("ChromaDB health check failed: %r", e)
        chroma_ok = False
    ollama_ok, model_ready = rag.ollama_status()

    if not db_ok:
        status_ = "down"
        response.status_code = 503
    elif chroma_ok and ollama_ok and model_ready:
        status_ = "ok"
    else:
        status_ = "degraded"

    return HealthResponse(
        status=status_,
        database="ok" if db_ok else "error",
        chroma="ok" if chroma_ok else "error",
        ollama="connected" if ollama_ok else "disconnected",
        model_ready=model_ready,
        model=settings.model_name,
    )


# ── Auth ───────────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=UserOut, status_code=201,
          responses={409: {"description": "Username already taken."}})
def register(req: UserCreate, db: DbSession):
    user = User(username=req.username, hashed_password=auth.hash_password(req.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # the unique index decides, so two registrations at once can't both win
        db.rollback()
        raise HTTPException(status_code=409, detail="Username already taken") from None
    return user


@app.post("/auth/token", response_model=Token, responses={401: {"description": "Wrong username or password."}})
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DbSession):
    user = auth.authenticate(db, form.username, form.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=auth.create_access_token(user.id), expires_in=settings.token_minutes * 60)


@app.get("/me", response_model=UserOut)
def me(user: CurrentUser):
    return user


@app.delete("/me", status_code=204)
def delete_me(user: CurrentUser, db: DbSession):
    user_id = user.id
    db.delete(user)
    db.commit()
    # after the commit: if this fails the chunks are orphans, which /ask
    # already drops because their postings aren't in the database
    try:
        rag.delete_user(user_id)
    except Exception as e:  # noqa: BLE001
        rag.log.warning("removing chunks for user %s failed: %r", user_id, e)


# ── Job postings ───────────────────────────────────────────────────────────

@app.get("/jobs", response_model=list[JobSummary])
def list_jobs(user: CurrentUser, db: DbSession, limit: Annotated[int, Query(ge=1, le=100)] = 50,
              offset: Annotated[int, Query(ge=0, le=MAX_ID)] = 0):
    stmt = (select(JobPosting).where(JobPosting.user_id == user.id)
            .order_by(JobPosting.created_at.desc(), JobPosting.id.desc()).limit(limit).offset(offset))
    return db.scalars(stmt).all()


@app.post("/jobs", response_model=JobOut, status_code=201,
          responses={409: {"description": "This posting is already saved."}})
def create_job(req: JobCreate, user: CurrentUser, db: DbSession):
    job = JobPosting(user_id=user.id, content_hash=content_hash(req.text), **req.model_dump())
    db.add(job)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="You've already saved this posting") from None
    try_index(job, db)
    return job


@app.post("/jobs/samples", response_model=SamplesResult)
def add_samples(user: CurrentUser, db: DbSession):
    existing = set(db.scalars(select(JobPosting.content_hash).where(JobPosting.user_id == user.id)))
    added = skipped = 0
    for sample in rag.load_samples():
        digest = content_hash(sample["text"])
        if digest in existing:
            skipped += 1
            continue
        job = JobPosting(user_id=user.id, source="sample", content_hash=digest, **sample)
        db.add(job)
        try:
            db.commit()
        except IntegrityError:
            # a second request (a double click) saved it first. Was a 500
            db.rollback()
            skipped += 1
            continue
        try_index(job, db)
        existing.add(digest)
        added += 1
    return SamplesResult(added=added, skipped=skipped)


@app.post("/jobs/reindex", response_model=ReindexResult)
def reindex(user: CurrentUser, db: DbSession):
    # rebuilds this user's chunks from the database one posting at a time. It
    # used to delete everything first, so a question asked mid-reindex found nothing
    jobs = db.scalars(select(JobPosting).where(JobPosting.user_id == user.id)).all()
    failed = [job.id for job in jobs if not try_index(job, db)]
    # then clears out orphans from postings deleted while Chroma was failing
    try:
        rag.delete_orphans(user.id, [job.id for job in jobs])
    except Exception as e:  # noqa: BLE001 - orphans are filtered out of /ask anyway
        rag.log.warning("removing orphan chunks for user %s failed: %r", user.id, e)
    return ReindexResult(indexed=len(jobs) - len(failed), failed=failed)


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: JobId, user: CurrentUser, db: DbSession):
    return get_owned_job(job_id, user, db)


@app.patch("/jobs/{job_id}", response_model=JobOut, responses={409: {"description": "Same text as another saved posting."}})
def update_job(job_id: JobId, req: JobUpdate, user: CurrentUser,
               db: DbSession):
    job = get_owned_job(job_id, user, db)
    changes = req.model_dump(exclude_unset=True)
    for field, value in changes.items():
        setattr(job, field, value)
    if "text" in changes:
        job.content_hash = content_hash(job.text)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Another saved posting has the same text") from None
    # title, company and location are in every chunk's header, so they reindex too
    if changes.keys() & {"text", "title", "company", "location"}:
        try_index(job, db)
    return job


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: JobId, user: CurrentUser, db: DbSession):
    job = get_owned_job(job_id, user, db)
    db.delete(job)
    db.commit()
    try:
        rag.delete_posting(job_id)
    except Exception as e:  # noqa: BLE001 - orphans are filtered out of /ask anyway
        rag.log.warning("removing chunks for job %s failed: %r", job_id, e)


# ── Ask ────────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse,
          responses={503: {"description": "Ollama is down, timed out or doesn't have the model, "
                                          "or ChromaDB can't be searched."}})
def ask(req: AskRequest, user: CurrentUser, db: DbSession):
    """
    1. load the user's postings (Postgres is the source of truth)
    2. scope the search: picked postings, else the companies the question names, else all
    3. retrieve this user's chunks from ChromaDB: the closest by meaning, plus
       chunks containing the question's distinctive words
    4. drop orphans, chunks whose posting is gone from Postgres
    5. prompt Ollama with the numbered chunks, then score the answer's citations

    Steps 1 and 4 can end early with a fixed answer and no model call.
    """
    jobs = {job.id: job for job in db.scalars(select(JobPosting).where(JobPosting.user_id == user.id))}
    if req.job_ids:
        unknown = set(req.job_ids) - jobs.keys()
        if unknown:
            raise HTTPException(status_code=404, detail=f"Job posting not found: {sorted(unknown)}")
    if not jobs:
        # nothing to ground an answer in, so the model isn't called
        return AskResponse(answer=NO_POSTINGS_ANSWER, confidence="low", sources=[], model=None)

    # "what does Cobalt Freight want?" only searches Cobalt's postings
    job_ids = req.job_ids or rag.named_postings(req.question, jobs.values())
    try:
        found = rag.retrieve(user.id, req.question, job_ids, req.n_results, rag.company_words(jobs.values()))
    except Exception as e:
        rag.log.warning("ChromaDB search failed: %r", e)
        raise HTTPException(status_code=503, detail="ChromaDB search failed") from e

    # Postgres is the source of truth: a chunk whose posting is gone is an orphan
    chunks = [c for c in found if c["job_id"] in jobs]
    if not chunks:
        return AskResponse(answer=NO_MATCH_ANSWER, confidence="low", sources=[], model=None)

    answer, truncated = rag.generate(rag.build_prompt(req.question, chunks))
    sources = [
        Source(n=i, job_id=c["job_id"], title=jobs[c["job_id"]].title, company=jobs[c["job_id"]].company,
               chunk=c["chunk"], excerpt=c["excerpt"], distance=c["distance"])
        for i, c in enumerate(chunks, start=1)
    ]
    # an answer cut off at the token cap is usually the model repeating itself
    confidence = "low" if truncated else rag.compute_confidence(chunks, answer)
    return AskResponse(answer=answer, confidence=confidence, sources=sources, model=settings.model_name,
                       truncated=truncated)
