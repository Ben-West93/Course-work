"""
JobFit API.

    uvicorn main:app --reload --port 8000     # SQLite + ./chroma, Ollama on localhost:11434

Swagger UI at http://localhost:8000/docs. Register, then use Authorize with the
same username and password.

Routes are thin: validation is in schemas.py, auth in auth.py, and the RAG
steps (chunk, index, search, prompt, generate, score) in rag_pipeline.py. The README's
"How a question is answered" section walks through /ask step by step, and
walkthrough.py runs it against the live services.

Status codes every route can return: 401 bad or missing token, 404 not found
(or not yours, or deleted by another request while this one ran), 409
duplicate, 413 a body over 1 MB, 422 invalid input, 429 too many failed
sign-ins, 503 a service the API needs (Postgres, ChromaDB, Ollama) is down.
A 500 is a bug.
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
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import ObjectDeletedError, StaleDataError

import auth
import database
import rag_pipeline
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
    """Create the tables before the API takes its first request."""
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
    """
    SHA-256 of the text, lowercased and with whitespace collapsed. Only case
    and whitespace, so re-pasting the same posting is caught but an edited one isn't.
    """
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def get_owned_job(job_id: int, user: User, db: Session) -> JobPosting:
    """The user's posting, or a 404. Someone else's posting is a 404 too, so ids can't be probed."""
    job = db.get(JobPosting, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Job posting not found")
    return job


# another request deleted the row this one was saving: a posting deleted while
# it was edited or reindexed, or the whole account deleted mid-request. Every
# route that ran into one was a 500
DELETED_MEANWHILE = (StaleDataError, ObjectDeletedError)


def try_index(job: JobPosting, db: Session) -> bool | None:
    """
    Index the posting and stamp indexed_at. True if search can find it, False
    if ChromaDB failed (it stays saved and a reindex fixes it), None if another
    request deleted it in the meantime.
    """
    # from the session's identity map, so it's there even once the row is gone
    job_id = inspect(job).identity[0]
    try:
        try:
            rag_pipeline.index_posting(job)
        except DELETED_MEANWHILE:
            raise
        except Exception as e:  # noqa: BLE001 - Postgres is the source of truth, reindex fixes it later
            rag_pipeline.log.warning("indexing job %s failed: %r", job_id, e)
            job.indexed_at = None
        else:
            job.indexed_at = datetime.now(UTC)
        db.commit()
    except DELETED_MEANWHILE:
        # the chunks just stored belong to a posting that's gone, so they go too
        db.rollback()
        rag_pipeline.log.info("job %s was deleted while it was being indexed", job_id)
        try:
            rag_pipeline.delete_posting(job_id)
        except Exception as e:  # noqa: BLE001 - orphans are filtered out of /ask anyway
            rag_pipeline.log.warning("removing chunks for job %s failed: %r", job_id, e)
        return None
    return job.indexed


def json_safe(value):
    """
    The value with NaN, infinities and lone surrogates made writable as JSON.
    FastAPI's default 422 handler echoes the input back, and those couldn't be.
    """
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
    """A 422 listing each field and why it was rejected."""
    return JSONResponse(status_code=422, content={"detail": json_safe(jsonable_encoder(exc.errors()))})


@app.exception_handler(OperationalError)
@app.exception_handler(InterfaceError)
async def database_unavailable(_request: Request, exc: Exception):
    """
    A 503 "Database unavailable" while Postgres is stopped or restarting.
    Every route that touches the database was a bare 500 with a traceback,
    but it's a dependency being down, so it's a 503 like Ollama.
    """
    rag_pipeline.log.warning("database unavailable: %s", str(exc).splitlines()[0])
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})


@app.exception_handler(StaleDataError)
@app.exception_handler(ObjectDeletedError)
async def deleted_meanwhile(_request: Request, exc: Exception):
    """
    A 404 for a row another request deleted while this one was using it. The
    routes that can race a delete handle it themselves, this catches anything
    else that does instead of a 500.
    """
    rag_pipeline.log.warning("row deleted during the request: %s", str(exc).splitlines()[0])
    return JSONResponse(status_code=404, content={"detail": "Not found: it was deleted while this request was running"})


# the biggest real request, a 20,000-character posting, is under 250 KB even
# when every character is escaped. A 30 MB one was read and parsed in full
# before validation turned it down, so anything over this is refused before
# its body is read. Only Content-Length is checked, which requests, the UI
# and Swagger always send
MAX_BODY_BYTES = 1_000_000


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """A 413 for a request whose body is over MAX_BODY_BYTES, without reading it."""
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Request body is over 1 MB"})
    return await call_next(request)


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, responses={503: {"description": "The database is down."}})
def health(response: Response):
    """
    Whether the database, ChromaDB and Ollama are up, and the model is pulled.

    ok: everything is up. degraded: logins and postings work but search or
    answers don't (ChromaDB broken, Ollama down, the model not pulled yet).
    down: no database, the only status that comes with a 503.
    """
    db_ok = database.ping()
    try:
        rag_pipeline.count_chunks()
        chroma_ok = True
    except Exception as e:  # noqa: BLE001
        rag_pipeline.log.warning("ChromaDB health check failed: %r", e)
        chroma_ok = False
    ollama_ok, model_ready = rag_pipeline.ollama_status()

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
    """Create an account. The username is stored lowercase, and one that's taken is a 409."""
    user = User(username=req.username, hashed_password=auth.hash_password(req.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # the unique index decides, so two registrations at once can't both win
        db.rollback()
        raise HTTPException(status_code=409, detail="Username already taken") from None
    return user


@app.post("/auth/token", response_model=Token, responses={
    401: {"description": "Wrong username or password."},
    429: {"description": "Too many wrong passwords for this username in the last 15 minutes."},
})
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DbSession):
    """
    Log in with the form fields `username` and `password`, which is what
    Swagger's Authorize button sends. Returns a bearer token for every other route.
    After 5 wrong passwords for a username in 15 minutes it's a 429 until the
    oldest of them is 15 minutes old, even with the right password.
    """
    # counted before the password is checked, so guesses sent at once can't
    # all get in first. The right password is refused too while it's blocked,
    # otherwise the block would only slow the guessing down
    retry_after = auth.start_login_attempt(form.username)
    if retry_after:
        minutes = math.ceil(retry_after / 60)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed sign-ins for this username. Try again in {minutes} "
                   f"minute{'s' if minutes != 1 else ''}.",
            headers={"Retry-After": str(retry_after)},
        )
    user = auth.authenticate(db, form.username, form.password)
    if user is None:
        # the attempt was already counted, and stays counted
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    auth.clear_failed_logins(form.username)
    return Token(access_token=auth.create_access_token(user.id), expires_in=settings.jwt_expire_minutes * 60)


@app.get("/me", response_model=UserOut)
def me(user: CurrentUser):
    """The signed-in user."""
    return user


@app.delete("/me", status_code=204)
def delete_me(user: CurrentUser, db: DbSession):
    """Delete the account, its postings and their chunks in ChromaDB."""
    user_id = user.id
    db.delete(user)
    db.commit()
    # after the commit: if this fails the chunks are orphans, which /ask
    # already drops because their postings aren't in the database
    try:
        rag_pipeline.delete_user(user_id)
    except Exception as e:  # noqa: BLE001
        rag_pipeline.log.warning("removing chunks for user %s failed: %r", user_id, e)


# ── Job postings ───────────────────────────────────────────────────────────

@app.get("/jobs", response_model=list[JobSummary])
def list_jobs(user: CurrentUser, db: DbSession, limit: Annotated[int, Query(ge=1, le=100)] = 50,
              offset: Annotated[int, Query(ge=0, le=MAX_ID)] = 0):
    """The user's postings, newest first, without their text."""
    stmt = (select(JobPosting).where(JobPosting.user_id == user.id)
            .order_by(JobPosting.created_at.desc(), JobPosting.id.desc()).limit(limit).offset(offset))
    return db.scalars(stmt).all()


@app.post("/jobs", response_model=JobOut, status_code=201,
          responses={409: {"description": "This posting is already saved."}})
def create_job(req: JobCreate, user: CurrentUser, db: DbSession):
    """
    Save a posting, then chunk and index it for search. If indexing fails the
    posting is still saved, with `indexed` false until a reindex.
    """
    user_id = user.id
    job = JobPosting(user_id=user_id, content_hash=content_hash(req.text), **req.model_dump())
    db.add(job)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        # the foreign key, not a duplicate: the account was deleted while this
        # request ran. It said "You've already saved this posting"
        if db.get(User, user_id) is None:
            raise auth.CREDENTIALS_ERROR from None
        raise HTTPException(status_code=409, detail="You've already saved this posting") from None
    if try_index(job, db) is None:
        # deleted straight after it was saved, usually with the whole account
        if db.get(User, user_id) is None:
            raise auth.CREDENTIALS_ERROR
        raise HTTPException(status_code=404, detail="Job posting not found")
    return job


@app.post("/jobs/samples", response_model=SamplesResult)
def add_samples(user: CurrentUser, db: DbSession):
    """Save and index the bundled sample postings, skipping any the user already has."""
    # read once: after a rollback, user.id would reload a row that may be gone
    user_id = user.id
    existing = set(db.scalars(select(JobPosting.content_hash).where(JobPosting.user_id == user_id)))
    added = skipped = 0
    for sample in rag_pipeline.load_samples():
        digest = content_hash(sample["text"])
        if digest in existing:
            skipped += 1
            continue
        job = JobPosting(user_id=user_id, source="sample", content_hash=digest, **sample)
        db.add(job)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            # the account was deleted part-way through. Was a 500
            if db.get(User, user_id) is None:
                raise auth.CREDENTIALS_ERROR from None
            # a second request (a double click) saved it first. Was a 500
            skipped += 1
            continue
        if try_index(job, db) is None:
            skipped += 1
            continue
        existing.add(digest)
        added += 1
    return SamplesResult(added=added, skipped=skipped)


@app.post("/jobs/reindex", response_model=ReindexResult)
def reindex(user: CurrentUser, db: DbSession):
    """
    Rebuild the user's chunks in ChromaDB from the database, one posting at a
    time, then remove any left from deleted postings. Returns how many were
    indexed and the ids of any that failed.
    """
    # it used to delete everything first, so a question asked mid-reindex found nothing
    user_id = user.id
    jobs = db.scalars(select(JobPosting).where(JobPosting.user_id == user_id)).all()
    # ids read up front: one posting found deleted rolls back, which expires the rest
    results = {job_id: try_index(job, db) for job_id, job in zip([job.id for job in jobs], jobs)}
    # a posting deleted part-way through (None) is neither indexed nor failed. Was a 500
    failed = [job_id for job_id, ok in results.items() if ok is False]
    still_saved = [job_id for job_id, ok in results.items() if ok is not None]
    # then clears out orphans from postings deleted while Chroma was failing
    try:
        rag_pipeline.delete_orphans(user_id, still_saved)
    except Exception as e:  # noqa: BLE001 - orphans are filtered out of /ask anyway
        rag_pipeline.log.warning("removing orphan chunks for user %s failed: %r", user_id, e)
    return ReindexResult(indexed=len(still_saved) - len(failed), failed=failed)


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: JobId, user: CurrentUser, db: DbSession):
    """One of the user's postings, with its full text."""
    return get_owned_job(job_id, user, db)


@app.patch("/jobs/{job_id}", response_model=JobOut, responses={409: {"description": "Same text as another saved posting."}})
def update_job(job_id: JobId, req: JobUpdate, user: CurrentUser,
               db: DbSession):
    """
    Edit a posting. Fields left out keep their value. A change to the text,
    title, company or location re-indexes it.
    """
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
    except DELETED_MEANWHILE:
        # deleted by another request after this one loaded it. Was a 500
        db.rollback()
        raise HTTPException(status_code=404, detail="Job posting not found") from None
    # title, company and location are in every chunk's header, so they reindex too
    if changes.keys() & {"text", "title", "company", "location"} and try_index(job, db) is None:
        raise HTTPException(status_code=404, detail="Job posting not found")
    return job


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: JobId, user: CurrentUser, db: DbSession):
    """Delete a posting and its chunks in ChromaDB."""
    job = get_owned_job(job_id, user, db)
    db.delete(job)
    db.commit()
    try:
        rag_pipeline.delete_posting(job_id)
    except Exception as e:  # noqa: BLE001 - orphans are filtered out of /ask anyway
        rag_pipeline.log.warning("removing chunks for job %s failed: %r", job_id, e)


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
    job_ids = req.job_ids or rag_pipeline.named_postings(req.question, jobs.values())
    try:
        found = rag_pipeline.retrieve(user.id, req.question, job_ids, req.n_results, rag_pipeline.company_words(jobs.values()))
    except Exception as e:
        rag_pipeline.log.warning("ChromaDB search failed: %r", e)
        raise HTTPException(status_code=503, detail="ChromaDB search failed") from e

    # Postgres is the source of truth: a chunk whose posting is gone is an orphan
    chunks = [c for c in found if c["job_id"] in jobs]
    if not chunks:
        return AskResponse(answer=NO_MATCH_ANSWER, confidence="low", sources=[], model=None)

    answer, truncated = rag_pipeline.generate_answer(rag_pipeline.build_prompt(req.question, chunks))
    sources = [
        Source(n=i, job_id=c["job_id"], title=jobs[c["job_id"]].title, company=jobs[c["job_id"]].company,
               chunk=c["chunk"], excerpt=c["excerpt"], distance=c["distance"])
        for i, c in enumerate(chunks, start=1)
    ]
    # an answer cut off at the token cap is usually the model repeating itself
    confidence = "low" if truncated else rag_pipeline.compute_confidence(chunks, answer)
    return AskResponse(answer=answer, confidence=confidence, sources=sources, model=settings.model_name,
                       truncated=truncated)
