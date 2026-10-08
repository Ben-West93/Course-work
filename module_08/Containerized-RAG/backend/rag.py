"""
RAG pipeline: chunking, ChromaDB storage/retrieval, prompt, Ollama call and
confidence scoring. main.py only wires these into endpoints.

    python rag.py        # ingest ./docs and run a couple of test questions
"""

import logging
import os
import re
import time

import chromadb
import requests
from fastapi import HTTPException

from config import settings

# uvicorn's logger, so these lines end up in `docker-compose logs backend`
log = logging.getLogger("uvicorn.error")
if settings.debug:
    log.setLevel(logging.DEBUG)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# a relative CHROMA_PATH (the default ./rag_db) is taken from this folder, so
# it's the same db wherever uvicorn or pytest is started from. In compose it's
# /app/rag_db, where the chroma_data volume is mounted. The volume lives outside
# the container, so the chunks survive `docker-compose down` and rebuilds
DB_PATH = os.path.normpath(os.path.join(BASE_DIR, os.path.expanduser(settings.chroma_path)))

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection("documents")


# ── 1. Document loading ────────────────────────────────────────────────────

# all-MiniLM-L6-v2 only reads the first 256 tokens (~1000 characters) of a
# chunk, anything after that wouldn't count towards the search
MAX_CHUNK_CHARS = 1000
# shorter than this and a paragraph is usually a tail of the one above it
# ("This handles requests like: GET /search?q=python&limit=5")
MIN_CHUNK_CHARS = 80


def is_code(para: str) -> bool:
    return all(line[:1] in (" ", "\t") for line in para.splitlines() if line.strip())


def is_heading(para: str) -> bool:
    # one short line that isn't a sentence: "Python Functions", "What is Ollama?",
    # "# ChromaDB", or a lead-in like "A minimal FastAPI app requires only a few lines:"
    if "\n" in para or len(para) > 80:
        return False
    if para.startswith("#") or para.endswith(":"):
        return True
    return ":" not in para and not para.endswith((".", "!", ","))


def split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        # a single sentence over the limit just gets cut
        while len(sentence) > MAX_CHUNK_CHARS:
            pieces.append(sentence[:MAX_CHUNK_CHARS])
            sentence = sentence[MAX_CHUNK_CHARS:]
        if current and len(current) + 1 + len(sentence) > MAX_CHUNK_CHARS:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str) -> list[str]:
    """Split on blank lines, then glue headings, code blocks and short tails back on."""
    paragraphs = [p.rstrip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]

    chunks, pending = [], ""
    for para in paragraphs:
        if is_code(para):
            # indented code belongs to the sentence that introduces it, even
            # when a blank line sits between them
            if pending:
                chunks.append(f"{pending}\n\n{para}")
                pending = ""
            elif chunks:
                chunks[-1] += f"\n\n{para}"
            else:
                chunks.append(para)
            continue

        para = para.strip()
        if is_heading(para):
            # a heading alone has nothing to answer from but embeds very close
            # to short questions, so it's carried onto the next paragraph
            pending = f"{pending}\n\n{para}" if pending else para
        elif pending:
            chunks.append(f"{pending}\n\n{para}")
            pending = ""
        elif len(para) < MIN_CHUNK_CHARS and chunks:
            chunks[-1] += f"\n\n{para}"
        else:
            chunks.append(para)
    # a heading at the very end with nothing under it is dropped

    return [piece for chunk in chunks for piece in split_long(chunk)]


def load_documents(directory: str) -> list[dict]:
    """Chunk every .txt/.md file in `directory` into {"text", "id", "metadata"} dicts."""
    if not os.path.isdir(directory):
        return []

    docs = []
    for filename in sorted(os.listdir(directory)):
        path = os.path.join(directory, filename)
        if filename.startswith(".") or not filename.lower().endswith((".txt", ".md")) or not os.path.isfile(path):
            continue
        # a stray non-UTF-8 byte shouldn't stop the whole ingest
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for i, chunk in enumerate(chunk_text(text)):
            docs.append({
                "text": chunk,
                # stable ids, so re-ingesting overwrites instead of duplicating
                "id": f"{filename}_{i}",
                "metadata": {"source": filename, "chunk_index": i},
            })
    return docs


def index_documents(docs: list[dict]) -> int:
    """Upsert the chunks and delete stored ones that no longer exist. Returns how many were deleted."""
    ids = [d["id"] for d in docs]
    collection.upsert(
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
        ids=ids,
    )
    # upsert never deletes, so chunks from a removed or shortened file would
    # keep turning up as sources
    stale = sorted(set(collection.get(include=[])["ids"]) - set(ids))
    if stale:
        collection.delete(ids=stale)
    return len(stale)


# ── 2. Retrieval ───────────────────────────────────────────────────────────

def source_name(meta: dict | None) -> str:
    # chroma allows numbers/bools/None as metadata, and pydantic won't turn
    # those into a str. Anything unusable becomes "unknown" instead of a 500
    source = (meta or {}).get("source")
    if not isinstance(source, str) or not source.strip():
        return "unknown"
    return source.strip()


def retrieve(query: str, n_results: int = settings.max_results,
             max_distance: float = settings.max_distance) -> list[dict]:
    """Closest chunks within max_distance as {"text", "source", "distance"}, closest first."""
    count = collection.count()
    if count == 0:
        return []

    # chroma raises if asked for more results than it holds
    results = collection.query(query_texts=[query], n_results=min(n_results, count))

    chunks, seen = [], set()
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        # lower is closer. Past the cutoff it's more likely noise than context,
        # so it doesn't go in the prompt or the sources
        if dist > max_distance:
            continue
        text = (doc or "").strip()
        # blank or repeated text would only fill a source slot
        if not text or text in seen:
            continue
        seen.add(text)
        chunks.append({"text": text, "source": source_name(meta), "distance": round(dist, 4)})

    log.debug("retrieve %r -> %s", query, [(c["source"], c["distance"]) for c in chunks])
    return chunks


# ── 3. Prompt building ─────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer ONLY from the provided context. "
    "If the context doesn't contain the answer, say you don't have enough "
    "information. Cite source documents by name. Keep responses under 200 words."
)


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n---\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"},
        {"role": "user", "content": question},
    ]


# ── 4. Generation ──────────────────────────────────────────────────────────

# a 1b model on CPU can take a while on the first call while it loads
OLLAMA_TIMEOUT = 120


def ollama_unavailable(detail: str, cause) -> HTTPException:
    # 503: the API is fine, a service it depends on isn't, retrying later can
    # work. The client gets the short detail, the log gets the real cause
    log.warning("Ollama call failed (%s): %s", detail, cause)
    return HTTPException(status_code=503, detail=detail)


def generate(messages: list[dict]) -> str:
    payload = {
        "model": settings.model_name,
        "messages": messages,
        "stream": False,
        # same question, same answer, which makes the confidence checks repeatable
        "options": {"temperature": 0},
    }
    start = time.perf_counter()
    try:
        r = requests.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
    except requests.exceptions.ConnectionError as e:
        raise ollama_unavailable("Ollama unavailable", repr(e)) from e
    except requests.exceptions.Timeout as e:
        raise ollama_unavailable("Ollama timed out", repr(e)) from e
    except requests.exceptions.RequestException as e:
        raise ollama_unavailable(f"Request to Ollama failed ({type(e).__name__})", repr(e)) from e
    log.debug("Ollama replied %s in %.1fs", r.status_code, time.perf_counter() - start)

    # 404 is the usual one: the model hasn't been pulled into the volume yet
    if r.status_code != 200:
        raise ollama_unavailable(f"Ollama returned {r.status_code} for model {settings.model_name}", r.text[:300])
    try:
        content = r.json()["message"]["content"]
    except (ValueError, KeyError, TypeError) as e:
        raise ollama_unavailable("Ollama's response had no answer in it", r.text[:300]) from e
    if not isinstance(content, str) or not content.strip():
        raise ollama_unavailable("Ollama returned an empty answer", r.text[:300])
    return content.strip()


def full_model_name(name: str) -> str:
    # Ollama lists "llama3" as "llama3:latest"
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


def ollama_status() -> tuple[bool, bool]:
    """(connected, model_pulled), from /api/tags, which is cheap and loads nothing."""
    try:
        r = requests.get(f"{settings.ollama_url}/api/tags", timeout=3)
        r.raise_for_status()
        models = r.json()["models"]
        pulled = {full_model_name(m.get("name") or m.get("model") or "") for m in models}
    except requests.exceptions.RequestException:
        return False, False
    except (ValueError, KeyError, TypeError, AttributeError):
        # answered, just not with a model list we can read
        return True, False
    return True, full_model_name(settings.model_name) in pulled


# ── 5. Confidence scoring ──────────────────────────────────────────────────

# chroma's default space is squared L2. The embeddings are unit length, so that's
# 2 - 2*cosine: 0.5 is cosine 0.75, 1.0 is cosine 0.5. Medium runs up to
# CONFIDENCE_THRESHOLD (1.0), and from there to MAX_DISTANCE (1.2) is borderline:
# on this corpus roughly one in five of those top chunks was the wrong section,
# and answers built from the right ones were often still off
HIGH_BELOW = 0.5
# two different files this close together at the top means retrieval can't
# tell which topic was meant, and the model tends to blend them. Only noise
# queries tied this closely, real cross-topic ones were 0.10 apart or more
AMBIGUITY_GAP = 0.05
# the system prompt asks the model to say this when the context doesn't help
REFUSAL = re.compile(r"(\bnot|n['’]t|\bno)\b[^.]{0,40}\benough information\b", re.IGNORECASE)
CITED_FILE = re.compile(r"\b[\w-]+\.(?:txt|md)\b", re.IGNORECASE)


def made_up_citations(answer: str, chunks: list[dict]) -> set[str]:
    # file names the answer cites that weren't in its context. Names that occur
    # inside a chunk (requirements.txt in docker_basics) are fine
    known = " ".join(f"{c['source']} {c['text']}" for c in chunks).lower()
    return {name for name in CITED_FILE.findall(answer) if name.lower() not in known}


def distance_gap(chunks: list[dict]) -> float | None:
    """How far the closest chunk is ahead of the closest one from a different file."""
    ranked = sorted(chunks, key=lambda c: c["distance"])
    best = ranked[0]
    rival = next((c for c in ranked[1:] if c["source"] != best["source"]), None)
    return None if rival is None else round(rival["distance"] - best["distance"], 4)


def compute_confidence(chunks: list[dict], answer: str = "") -> str:
    """
    high:   top chunk < 0.5
    medium: top chunk 0.5 - CONFIDENCE_THRESHOLD (1.0)
    low:    no chunks, top chunk past the threshold (borderline), two files
            nearly tied at the top, the model said it lacked information, or
            it cited a file that wasn't in its context
    """
    if not chunks:
        return "low"
    best = min(c["distance"] for c in chunks)
    if best > settings.confidence_threshold:
        return "low"
    gap = distance_gap(chunks)
    if gap is not None and gap < AMBIGUITY_GAP:
        return "low"
    # a close match doesn't help if the model says the context doesn't cover it
    if REFUSAL.search(answer):
        return "low"
    # a made-up source makes the answer look better grounded than it is
    if made_up_citations(answer, chunks):
        return "low"
    return "high" if best < HIGH_BELOW else "medium"


# ── Quick test (run as a script) ───────────────────────────────────────────

if __name__ == "__main__":
    docs = load_documents(DOCS_DIR)
    print(f"Loaded {len(docs)} chunks from {DOCS_DIR}")
    if docs:
        removed = index_documents(docs)
        print(f"Stored {collection.count()} chunks in {DB_PATH} (removed {removed} stale)")
        for q in ["What is a list comprehension?", "What is Pydantic?", "How do I bake sourdough bread?"]:
            found = retrieve(q)
            print(f"\n{q}")
            for c in found:
                print(f"  {c['distance']:.4f}  {c['source']}  {c['text'][:60]!r}")
            print(f"  confidence={compute_confidence(found)}")
