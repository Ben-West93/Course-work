"""
Connecting RAG to FastAPI
Run from this folder with:
    uvicorn my_rag_api:app --reload --port 8000

or with Ollama alongside it in Docker Compose (see README.md):
    docker-compose up --build -d

Then open http://localhost:8000/docs to test the endpoints in Swagger UI.
All configuration comes from config.py, i.e. environment variables or .env.
"""

from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import chromadb
import requests
import logging
import math
import os
import re
import time

from config import settings

app = FastAPI(
    title="RAG API",
    version="1.2",
    description=(
        "Question answering over the files in `docs/`. Call `POST /ingest` once to load "
        f"them into ChromaDB, then `POST /ask`. Answers come from {settings.model_name} through "
        "Ollama, and the chunks used as context are returned as `sources`."
    ),
    # unhandled errors come back with the traceback instead of a bare 500
    debug=settings.debug,
)

# wide open so any local frontend can call it. Fine for coursework, but a real
# deployment would list its own origins here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# uvicorn's own logger, so these lines show up in docker logs next to the requests
log = logging.getLogger("uvicorn.error")

# ── Config ─────────────────────────────────────────────────────────────────
# the model, Ollama URL, db path and /ask defaults all come from settings, see
# config.py. Inside a container localhost is the container itself, so compose
# sets OLLAMA_URL to http://ollama:11434, the ollama service's name

# DEBUG=true adds what retrieval found and how long Ollama took to the logs
if settings.debug:
    log.setLevel(logging.DEBUG)
    log.debug("Loaded %r", settings)

# a relative CHROMA_PATH (the default ./rag_db) is taken from this file's folder,
# so it's the same db no matter where uvicorn is started from. An absolute one,
# like /app/rag_db from compose, is used as is (join drops BASE_DIR)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.normpath(os.path.join(BASE_DIR, settings.chroma_path))
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# ── ChromaDB setup ─────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("documents")


# ── Pydantic schemas ───────────────────────────────────────────────────────

class AskRequest(BaseModel):
    # all-MiniLM-L6-v2 only reads the first 256 tokens (roughly 1000 characters),
    # so anything longer wouldn't change the search anyway
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="The question. Surrounding whitespace is stripped, and it can't be blank.",
    )
    # strict: without it pydantic quietly turns true into 1 and "5" into 5
    n_results: int = Field(
        settings.max_results, ge=1, le=20, strict=True,
        description="How many chunks to fetch from ChromaDB. Fewer come back when some are past "
                    "max_distance, blank, or repeat the text of a closer one. Defaults to MAX_RESULTS.",
    )
    # Python's json parser accepts NaN and Infinity, so those need ruling out too
    max_distance: float = Field(
        settings.confidence_threshold, ge=0, le=4, strict=True, allow_inf_nan=False,
        description="Chunks further than this from the question are dropped. Squared L2 "
                    "distance: 0 is identical, around 1.2 is loosely related, 4 is opposite. "
                    "Defaults to CONFIDENCE_THRESHOLD.",
    )

    model_config = {"json_schema_extra": {"examples": [
        {"question": "How does ChromaDB store embeddings?",
         "n_results": settings.max_results, "max_distance": settings.confidence_threshold}
    ]}}

    # min_length alone lets "   " through. A ValueError inside a validator
    # becomes a 422 response before the route even runs
    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


class SourceChunk(BaseModel):
    text: str = Field(description="The chunk that was given to the model as context, trimmed. Never blank.")
    source: str = Field(description="File the chunk came from, trimmed. \"unknown\" when it has no source "
                                    "or the source is blank.")
    distance: float = Field(description="Distance from the question, rounded to 4 places. Lower is closer.")


class AskResponse(BaseModel):
    answer: str = Field(description="The model's answer, or a fixed message when there was nothing to answer from.")
    sources: list[SourceChunk] = Field(
        description="Closest first, each text only once. Empty when nothing was stored or nothing "
                    "was within max_distance."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How far to trust the answer. See the README for how it's worked out."
    )

    model_config = {"json_schema_extra": {"examples": [{
        "answer": "ChromaDB stores embeddings using the all-MiniLM-L6-v2 sentence-transformer "
                  "model. According to the source, it produces 384-dimensional vectors.",
        "sources": [
            {
                "text": "# ChromaDB\n\nChromaDB is an open-source vector database. It stores documents, "
                        "their embeddings, metadata and ids in collections, and handles the embedding "
                        "step automatically if you pass raw text.",
                "source": "chromadb.md",
                "distance": 0.4948,
            },
            {
                "text": "ChromaDB uses the all-MiniLM-L6-v2 sentence-transformer model by default. It "
                        "produces 384-dimensional vectors and runs locally on CPU, so no API key is needed.",
                "source": "embeddings.md",
                "distance": 0.8273,
            },
        ],
        "confidence": "high",
    }]}}


class IngestResponse(BaseModel):
    chunks_ingested: int = Field(description="Chunks written to ChromaDB on this call.")
    files_ingested: int = Field(description="Files those chunks came from.")
    chunks_removed: int = Field(description="Old chunks deleted because their file was removed or got shorter.")
    message: str = Field(description="The same counts as a sentence.")


class StatsResponse(BaseModel):
    document_count: int = Field(description="Chunks currently stored.")
    model: str = Field(description="Ollama model used by /ask (MODEL_NAME).")
    db_path: str = Field(description="Where ChromaDB keeps its files (CHROMA_PATH).")
    ollama_url: str = Field(description="Where the API reaches Ollama (OLLAMA_URL).")
    max_results: int = Field(description="Default n_results for /ask (MAX_RESULTS).")
    confidence_threshold: float = Field(description="Default max_distance for /ask (CONFIDENCE_THRESHOLD).")
    debug: bool = Field(description="Whether debug logging is on (DEBUG).")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"] = Field(description="Overall status, see the endpoint description.")
    chromadb: Literal["ok", "error"] = Field(description="Whether the collection can be read.")
    ollama: Literal["connected", "disconnected"] = Field(description="Whether Ollama answered on /api/tags.")
    document_count: int | None = Field(description="Chunks stored, null when ChromaDB can't be read.")
    model: str = Field(description="Ollama model used by /ask (MODEL_NAME).")


class ErrorResponse(BaseModel):
    detail: str = Field(description="What went wrong.")


class RootResponse(BaseModel):
    message: str = Field(description="Always \"RAG API is running\".")
    docs: str = Field(description="Where the Swagger UI is.")


# ── RAG helpers ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer ONLY from the provided context. "
    "If the context doesn't contain the answer, say you don't have enough "
    "information. Cite source documents by name. Keep responses under 200 words."
)

EMPTY_DB_ANSWER = "No documents have been ingested yet. Call POST /ingest first."
NO_MATCH_ANSWER = "No relevant documents found for this question. Try rephrasing it or raising max_distance."


def is_heading(para: str) -> bool:
    return para.startswith("#") and "\n" not in para


def load_documents(directory: str) -> list[dict]:
    if not os.path.isdir(directory):
        return []

    chunks = []
    for filename in sorted(os.listdir(directory)):
        path = os.path.join(directory, filename)
        if not filename.lower().endswith((".txt", ".md")) or not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read().replace("\r\n", "\n")

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # a heading on its own is a few words with nothing to answer from, but it
        # embeds really close to short questions ("ChromaDB" -> "# ChromaDB") and
        # then takes up a source slot. Stick it onto the paragraph under it instead.
        # Headings with nothing after them are dropped
        merged, heading = [], None
        for para in paragraphs:
            if is_heading(para):
                heading = para if heading is None else f"{heading}\n\n{para}"
                continue
            merged.append(f"{heading}\n\n{para}" if heading else para)
            heading = None

        for i, para in enumerate(merged):
            chunks.append({
                "text": para,
                "id": f"{filename}_{i}",
                "metadata": {"source": filename, "chunk_index": i},
            })
    return chunks


def source_name(meta: dict | None) -> str:
    # /ingest always stores the file name, but chroma also allows numbers and
    # bools as metadata, and pydantic won't turn those into a str (that was a
    # 500). A blank name says nothing about where the text came from
    source = (meta or {}).get("source")
    if source is None or isinstance(source, bool):
        return "unknown"
    return str(source).strip() or "unknown"


def retrieve(query: str, n_results: int = settings.max_results,
             max_distance: float = settings.confidence_threshold) -> list[dict]:
    count = collection.count()
    if count == 0:
        return []

    # chroma errors if n_results is more than what's stored
    results = collection.query(query_texts=[query], n_results=min(n_results, count))

    chunks, seen = [], set()
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0], strict=True
    ):
        # lower distance = closer match. Anything past the cutoff is more
        # likely noise than context, so it never reaches the prompt
        if dist > max_distance:
            continue
        # /ingest never stores blank text, but anything added to the
        # collection some other way might
        text = (doc or "").strip()
        if not text:
            continue
        # the same paragraph stored twice (in two files, or under another id)
        # would fill two source slots with one piece of context. Results come
        # back closest first, so the copy that's kept is the closest one
        if text in seen:
            continue
        seen.add(text)
        chunks.append({"text": text, "source": source_name(meta), "distance": round(dist, 4)})
    return chunks


# 2+ digit numbers and acronyms like MIT or API. If the answer has one that's
# nowhere in the question or the sources, the model most likely got it from
# its training data rather than the context
SPECIFIC_NUMBER = re.compile(r"\d[\d,]*\d")
ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]+s?\b")
# the system prompt tells the model to say it doesn't have enough information
REFUSAL = re.compile(r"(\bnot|n['’]t|\bno)\b[^.]{0,40}\benough information\b", re.IGNORECASE)
LEVELS = ["low", "medium", "high"]


def unsupported_specifics(answer: str, question: str, chunks: list[dict]) -> set[str]:
    reference = " ".join([question] + [f"{c['source']} {c['text']}" for c in chunks])
    known_numbers = {n.replace(",", "") for n in SPECIFIC_NUMBER.findall(reference)}
    numbers = {n.replace(",", "") for n in SPECIFIC_NUMBER.findall(answer)} - known_numbers
    acronyms = {a.rstrip("s") for a in ACRONYM.findall(answer)}
    # whole words only, otherwise "MIT" counts as found in "limit"
    acronyms = {a for a in acronyms if not re.search(rf"\b{re.escape(a)}s?\b", reference, re.IGNORECASE)}
    return numbers | acronyms


def compute_confidence(chunks: list[dict], answer: str = "", question: str = "") -> str:
    if not chunks:
        return "low"
    # the model says the context doesn't cover it, so a close match doesn't help
    if REFUSAL.search(answer):
        return "low"

    # squared L2 between unit vectors is 2 - 2*cos, so 0.5 and 0.8 are cosine
    # similarity 0.75 and 0.6 (and the default 1.0 CONFIDENCE_THRESHOLD is 0.5)
    best = min(c["distance"] for c in chunks)
    level = 2 if best < 0.5 else 1 if best < 0.8 else 0

    # only one step down, the check is a heuristic and can't prove anything
    if unsupported_specifics(answer, question, chunks):
        level = max(level - 1, 0)
    return LEVELS[level]


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n---\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"},
        {"role": "user", "content": question},
    ]


def ollama_error(status: int, detail: str, cause: str) -> HTTPException:
    # the client only gets the short detail, the log gets the actual cause
    # (e.g. Ollama's own "model not found, try pulling it first")
    log.warning("Ollama /api/chat failed, returning %s: %s", status, cause)
    return HTTPException(status_code=status, detail=detail)


def call_ollama(messages: list[dict]) -> str:
    payload = {"model": settings.model_name, "messages": messages, "stream": False,
               "options": {"temperature": 0}}
    # 503 rather than 500: the API itself is fine, the service it depends on
    # isn't, and the client can retry once Ollama is back up
    start = time.perf_counter()
    try:
        r = requests.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=120)
    except requests.exceptions.ConnectionError as e:
        raise ollama_error(503, "Ollama service is unavailable", repr(e)) from e
    except requests.exceptions.Timeout as e:
        raise ollama_error(503, "Ollama timed out", repr(e)) from e
    except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError) as e:
        # Ollama was reached, the reply just got cut off part way, so it's a bad
        # response (502) rather than "couldn't reach it"
        raise ollama_error(502, "Ollama's response was cut off", repr(e)) from e
    except requests.exceptions.RequestException as e:
        raise ollama_error(503, f"Could not reach Ollama ({type(e).__name__})", repr(e)) from e
    log.debug("Ollama %s replied %s in %.1fs", settings.model_name, r.status_code, time.perf_counter() - start)

    # e.g. 404 when the model hasn't been pulled yet
    if r.status_code != 200:
        raise ollama_error(503, f"Ollama returned {r.status_code} for model {settings.model_name}", r.text[:300])

    # 502 from here on: Ollama did answer, just not with anything usable
    try:
        content = r.json()["message"]["content"]
    except (ValueError, KeyError, TypeError) as e:
        raise ollama_error(502, "Ollama's response had no answer in it", r.text[:300]) from e
    if not isinstance(content, str) or not content.strip():
        raise ollama_error(502, "Ollama returned an empty answer", r.text[:300])
    return content.strip()


def chroma_error(detail: str, cause: Exception) -> HTTPException:
    # 503 like the Ollama errors, and the actual cause goes to the log
    log.warning("ChromaDB failed, returning 503: %r", cause)
    return HTTPException(status_code=503, detail=detail)


def check_ollama_health() -> bool:
    try:
        r = requests.get(f"{settings.ollama_url}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ── Endpoints ──────────────────────────────────────────────────────────────

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


# FastAPI's default 422 handler echoes the bad input back. Python's json parser
# accepts NaN, Infinity and lone surrogates like "\ud800", none of which can be
# written back out as JSON, so the error response itself crashed with a 500
@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": json_safe(jsonable_encoder(exc.errors()))})


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def error_doc(description: str, example: str, model=ErrorResponse) -> dict:
    return {"model": model, "description": description,
            "content": {"application/json": {"example": {"detail": example}}}}


ASK_ERRORS = {
    400: error_doc("The body couldn't be read at all: not valid UTF-8, or JSON nested tens of "
                   "thousands of levels deep.", "There was an error parsing the body"),
    502: error_doc("Ollama replied, but with no usable answer (bad JSON, empty text, or cut off part way).",
                   "Ollama returned an empty answer"),
    503: error_doc("Ollama is unreachable, timed out, or returned an error, e.g. the model isn't pulled. "
                   "Also when ChromaDB can't be read (`ChromaDB is not readable`) or searched "
                   "(`ChromaDB search failed`).",
                   "Ollama service is unavailable"),
}


@app.get(
    "/",
    response_model=RootResponse,
    summary="Check the API is up",
    description="Doesn't touch ChromaDB or Ollama, `/health` is the one that checks those.",
)
def root():
    return RootResponse(message="RAG API is running", docs="/docs")


@app.post(
    "/ask",
    response_model=AskResponse,
    responses=ASK_ERRORS,
    summary="Answer a question from the ingested docs",
    description=(
        "Embeds the question, fetches the `n_results` closest chunks, drops any further than "
        "`max_distance`, and sends the rest to the model as context. If nothing is left, it "
        "returns a fixed answer with no sources and doesn't call the model, so this still "
        "works with Ollama down."
    ),
)
def ask(req: AskRequest):
    try:
        stored = collection.count()
    except Exception as e:
        raise chroma_error("ChromaDB is not readable", e) from e

    # two different reasons for having nothing to answer from, so two messages
    if stored == 0:
        return AskResponse(answer=EMPTY_DB_ANSWER, sources=[], confidence="low")

    try:
        chunks = retrieve(req.question, req.n_results, req.max_distance)
    except Exception as e:
        # the question is embedded in here too, so a broken embedding model
        # ends up here as well as a broken db
        raise chroma_error("ChromaDB search failed", e) from e
    log.debug("%d chunks within %s for %r: %s", len(chunks), req.max_distance, req.question,
              [(c["source"], c["distance"]) for c in chunks])
    if not chunks:
        return AskResponse(answer=NO_MATCH_ANSWER, sources=[], confidence="low")

    answer = call_ollama(build_messages(req.question, chunks))
    return AskResponse(
        answer=answer,
        sources=[SourceChunk(**c) for c in chunks],
        confidence=compute_confidence(chunks, answer, req.question),
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    responses={503: error_doc("ChromaDB couldn't be written to. Calling `/ingest` again once it's "
                              "back finishes the job.", "Could not write to ChromaDB")},
    summary="Load docs/ into ChromaDB",
    description=(
        "Splits every .txt and .md file in `docs/` into paragraph chunks and upserts them. "
        "Safe to call repeatedly: chunk ids are stable, and chunks whose file was removed or "
        "got shorter are deleted. If `docs/` is missing or empty nothing is changed."
    ),
)
def ingest():
    chunks = load_documents(DOCS_DIR)
    if not chunks:
        # leave what's stored alone, an empty or missing folder is more likely
        # a mistake than a request to wipe the collection
        return IngestResponse(
            chunks_ingested=0, files_ingested=0, chunks_removed=0,
            message=f"No .txt or .md content found in {DOCS_DIR}, nothing changed",
        )

    # ids are stable (filename_index), so re-ingesting overwrites instead of duplicating
    ids = [c["id"] for c in chunks]
    try:
        collection.upsert(
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=ids,
        )

        # upsert never deletes, so chunks from a file that was removed or got
        # shorter would stick around and keep coming back as sources
        stale = sorted(set(collection.get(include=[])["ids"]) - set(ids))
        if stale:
            collection.delete(ids=stale)
    except Exception as e:
        # every step here can be repeated, so a retry picks up where this stopped
        raise chroma_error("Could not write to ChromaDB", e) from e

    files = len({c["metadata"]["source"] for c in chunks})
    message = f"Ingested {plural(len(chunks), 'chunk')} from {plural(files, 'file')}"
    if stale:
        message += f", removed {plural(len(stale), 'old chunk')}"
    return IngestResponse(
        chunks_ingested=len(chunks), files_ingested=files,
        chunks_removed=len(stale), message=message,
    )


@app.get(
    "/stats",
    response_model=StatsResponse,
    responses={503: error_doc("ChromaDB can't be read.", "ChromaDB is not readable")},
    summary="Collection size and config",
    description="How many chunks are stored, plus the settings the API loaded from the "
                "environment. Handy for checking a change to `.env` took effect.",
)
def stats():
    try:
        count = collection.count()
    except Exception as e:
        raise chroma_error("ChromaDB is not readable", e) from e
    return StatsResponse(
        document_count=count,
        model=settings.model_name,
        db_path=DB_PATH,
        ollama_url=settings.ollama_url,
        max_results=settings.max_results,
        confidence_threshold=settings.confidence_threshold,
        debug=settings.debug,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {
        "model": HealthResponse,
        "description": "ChromaDB can't be read. `document_count` is also in the body as null, "
                       "Swagger just leaves nulls out of examples.",
        "content": {"application/json": {"example": {
            "status": "error", "chromadb": "error", "ollama": "connected",
            "document_count": None, "model": settings.model_name,
        }}},
    }},
    summary="Check ChromaDB and Ollama",
    description=(
        "`ok`: both work. `degraded` (still 200): ChromaDB works but Ollama doesn't, so "
        "`/ingest` and retrieval work but `/ask` can't generate answers. `error` (503): "
        "ChromaDB can't be read."
    ),
)
def health(response: Response):
    # chroma: if count() works the db is open and readable
    try:
        document_count = collection.count()
        chroma_ok = True
    except Exception as e:
        log.warning("ChromaDB count failed: %r", e)
        document_count = None
        chroma_ok = False

    # ollama: /api/tags is cheap and doesn't load a model
    ollama_ok = check_ollama_health()

    if chroma_ok and ollama_ok:
        status = "ok"
    elif chroma_ok:
        status = "degraded"  # can still ingest and retrieve, just can't answer
    else:
        status = "error"
        response.status_code = 503

    return HealthResponse(
        status=status,
        chromadb="ok" if chroma_ok else "error",
        ollama="connected" if ollama_ok else "disconnected",
        document_count=document_count,
        model=settings.model_name,
    )
