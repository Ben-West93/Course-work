"""
CI Tests: FastAPI RAG API
=========================
Run locally:
    cd backend
    pip install pytest httpx
    pytest tests/ -v

These run in GitHub Actions on every push and PR to main. There's no Ollama
there, so every call main.py makes to it is faked. The /ask and /ingest tests
also swap the ChromaDB collection for a fake, since a real query would have to
download the embedding model first.
"""

import json
import os

# config.py reads the environment once, when main is first imported, so this
# has to happen before that. setdefault keeps what the workflow passes in and
# gives a local run the same values, so it uses backend/test_chroma and never
# the rag_db that uvicorn writes to
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("CHROMA_PATH", "./test_chroma")
# pinned to config.py's defaults. load_dotenv never overrides a variable that's
# already set, so a local .env with MAX_RESULTS=1 can't change what these see
os.environ.update({"MODEL_NAME": "llama3.2:1b", "MAX_RESULTS": "3",
                   "CONFIDENCE_THRESHOLD": "1.0", "DEBUG": "false"})

import pytest
import requests
from fastapi.testclient import TestClient

import main
from main import app, compute_confidence

# TestClient calls the app in-process through httpx: no uvicorn, no port, and
# each client.get/post goes through routing, validation and the route itself
# exactly like a real request would
client = TestClient(app)

ANSWER = "ChromaDB keeps each chunk and its embedding in a collection [chromadb.md]."

# (text, source, distance), closest first like chroma returns them
ROWS = [
    ("ChromaDB is an open-source vector database. It stores documents and their embeddings.",
     "chromadb.md", 0.31234),
    ("all-MiniLM-L6-v2 turns each chunk into a 384-dimensional vector.", "embeddings.md", 0.6181),
]


class FakeResponse:
    # data is the JSON body. Pass text instead for a body that isn't JSON,
    # json() then raises ValueError the way requests' own one does
    def __init__(self, status_code=200, data=None, text=None):
        self.status_code = status_code
        self._data = data
        self.text = json.dumps(data) if text is None else text

    def json(self):
        if self._data is None:
            raise ValueError("Expecting value")
        return self._data


class FakeCollection:
    # stands in for the chroma collection main.py opens at import. Nothing gets
    # embedded, query() returns the rows it was given. A row with source None
    # comes back with no metadata at all
    def __init__(self, rows=(), stored_ids=(), broken=False, search_broken=False):
        self.rows = list(rows)
        self.ids = set(stored_ids)
        self.broken = broken
        self.search_broken = search_broken
        self.queried_with = None

    def _check(self):
        if self.broken:
            raise RuntimeError("database is locked")

    def count(self):
        self._check()
        return len(self.rows)

    def query(self, query_texts, n_results):
        self._check()
        # count() still works, only the search (which embeds the question) fails
        if self.search_broken:
            raise RuntimeError("embedding model missing")
        self.queried_with = n_results
        rows = self.rows[:n_results]
        return {
            "documents": [[text for text, _, _ in rows]],
            "metadatas": [[None if source is None else {"source": source} for _, source, _ in rows]],
            "distances": [[distance for _, _, distance in rows]],
        }

    def upsert(self, documents, metadatas, ids):
        self._check()
        self.ids.update(ids)

    def get(self, include):
        return {"ids": sorted(self.ids)}

    def delete(self, ids):
        self.ids -= set(ids)


def use_collection(monkeypatch, **kwargs):
    fake = FakeCollection(**kwargs)
    # the routes look up main.collection on every call, so this swaps it for
    # one test and monkeypatch puts the real one back afterwards
    monkeypatch.setattr(main, "collection", fake)
    return fake


@pytest.fixture(autouse=True)
def ollama_offline(monkeypatch):
    # runs for every test: requests.get/post fail straight away, like a refused
    # connection. So nothing reaches a real Ollama, even one running locally,
    # and nothing sits waiting on a timeout in CI. TestClient uses httpx, not
    # requests, so the test's own calls to the app aren't affected
    def refuse(*args, **kwargs):
        raise requests.exceptions.ConnectionError("no Ollama in tests")

    monkeypatch.setattr(main.requests, "get", refuse)
    monkeypatch.setattr(main.requests, "post", refuse)


@pytest.fixture
def ollama(monkeypatch):
    # Ollama up with the model pulled. Returns the /api/chat payloads it got
    sent = []

    def post(url, json=None, timeout=None):
        sent.append(json)
        return FakeResponse(200, {"message": {"role": "assistant", "content": ANSWER}})

    def get(url, timeout=None):
        return FakeResponse(200, {"models": [{"name": main.settings.model_name}]})

    monkeypatch.setattr(main.requests, "post", post)
    monkeypatch.setattr(main.requests, "get", get)
    return sent


def ollama_replies(monkeypatch, reply):
    # reply is a FakeResponse for /api/chat to return, or an exception for
    # requests.post to raise
    def post(*args, **kwargs):
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(main.requests, "post", post)


def ask(question="How does ChromaDB store embeddings?", **extra):
    return client.post("/ask", json={"question": question, **extra})


def post_raw(body: str | bytes):
    # for bodies client.post(json=...) won't send: httpx refuses NaN, and bytes
    # that aren't valid UTF-8 can't come from a dict at all
    return client.post("/ask", content=body, headers={"Content-Type": "application/json"})


def assert_ask_shape(data):
    # every /ask 200 has exactly these three fields, none of them null
    assert set(data) == {"answer", "sources", "confidence"}
    assert isinstance(data["answer"], str)
    assert data["answer"].strip()
    assert data["confidence"] in {"high", "medium", "low"}
    assert isinstance(data["sources"], list)
    for s in data["sources"]:
        assert set(s) == {"text", "source", "distance"}
        assert s["text"].strip()
        assert s["source"].strip()
        assert isinstance(s["distance"], float)
        assert s["distance"] == round(s["distance"], 4)
    distances = [s["distance"] for s in data["sources"]]
    assert distances == sorted(distances)


# ── the three from the exercise ────────────────────────────────────────────

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "RAG API is running", "docs": "/docs"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "chromadb" in data


def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "document_count" in data
    # this one is the real (empty) test_chroma collection, not a fake
    assert isinstance(data["document_count"], int)
    assert data["document_count"] >= 0


# ── /health and /stats ─────────────────────────────────────────────────────

def test_health_ok_when_ollama_has_the_model(ollama):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["ollama"] == "connected"
    assert data["model_pulled"] is True


def test_health_degraded_when_ollama_is_down():
    # still a 200: ingest and retrieval work, only /ask can't answer
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["chromadb"] == "ok"
    assert data["ollama"] == "disconnected"
    assert data["model_pulled"] is False


def test_health_503_when_chromadb_is_broken(monkeypatch):
    use_collection(monkeypatch, broken=True)
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "error"
    assert data["chromadb"] == "error"
    assert data["document_count"] is None


@pytest.mark.parametrize(
    ("tags", "ollama_state"),
    [
        (FakeResponse(200, {"models": [{"name": "llama3.2:3b"}]}), "connected"),
        # a bare name means :latest to Ollama, which isn't llama3.2:1b
        (FakeResponse(200, {"models": [{"name": "llama3.2"}]}), "connected"),
        (FakeResponse(200, ["not", "the", "usual", "shape"]), "connected"),
        (FakeResponse(500, {"error": "internal error"}), "disconnected"),
    ],
    ids=["other-model-pulled", "bare-name-is-latest", "junk-model-list", "tags-500"],
)
def test_health_degraded_when_the_model_cant_be_used(monkeypatch, tags, ollama_state):
    # /ask couldn't run the model in any of these, so not ok
    monkeypatch.setattr(main.requests, "get", lambda *a, **k: tags)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["ollama"] == ollama_state
    assert data["model_pulled"] is False


def test_stats_reports_settings_from_env():
    data = client.get("/stats").json()
    assert data["model"] == "llama3.2:1b"
    assert data["max_results"] == 3
    assert data["confidence_threshold"] == 1.0
    assert data["ollama_url"] == os.environ["OLLAMA_URL"].rstrip("/")


def test_stats_503_when_chromadb_is_broken(monkeypatch):
    use_collection(monkeypatch, broken=True)
    response = client.get("/stats")
    assert response.status_code == 503
    assert response.json() == {"detail": "ChromaDB is not readable"}


# ── /ask: valid requests ───────────────────────────────────────────────────

def test_ask_returns_answer_sources_and_confidence(monkeypatch, ollama):
    use_collection(monkeypatch, rows=ROWS)
    response = ask()
    assert response.status_code == 200
    data = response.json()
    assert_ask_shape(data)
    assert data["answer"] == ANSWER
    # 0.31 is under HIGH_DISTANCE and the next file is 0.31 further away
    assert data["confidence"] == "high"
    assert [s["source"] for s in data["sources"]] == ["chromadb.md", "embeddings.md"]
    assert data["sources"][0]["distance"] == 0.3123


def test_ask_sends_the_question_and_context_to_ollama(monkeypatch, ollama):
    use_collection(monkeypatch, rows=ROWS)
    ask("  How does ChromaDB store embeddings?  ")
    assert len(ollama) == 1
    payload = ollama[0]
    assert payload["model"] == "llama3.2:1b"
    system, user = payload["messages"]
    assert "[Source: chromadb.md]" in system["content"]
    # the validator strips surrounding whitespace before the route sees it
    assert user == {"role": "user", "content": "How does ChromaDB store embeddings?"}


def test_ask_with_nothing_ingested_doesnt_call_ollama(monkeypatch):
    # Ollama is offline (autouse fixture), so a 200 here means it wasn't called
    use_collection(monkeypatch, rows=[])
    response = ask()
    assert response.status_code == 200
    assert response.json() == {"answer": main.EMPTY_DB_ANSWER, "sources": [], "confidence": "low"}


def test_ask_nothing_within_max_distance(monkeypatch):
    use_collection(monkeypatch, rows=[("Unrelated text about bread.", "baking.md", 1.6)])
    response = ask(max_distance=1.0)
    assert response.status_code == 200
    assert response.json() == {"answer": main.NO_MATCH_ANSWER, "sources": [], "confidence": "low"}


@pytest.mark.parametrize(
    ("question", "status"),
    [("x" * 1000, 200), ("  " + "x" * 1000 + "  ", 200), ("x" * 1001, 422)],
    ids=["1000", "1000-plus-spaces", "1001"],
)
def test_ask_question_length_limit(monkeypatch, ollama, question, status):
    # the validator strips before max_length is checked, so padding doesn't count
    use_collection(monkeypatch, rows=ROWS)
    assert ask(question).status_code == status


def test_ask_never_asks_chroma_for_more_than_is_stored(monkeypatch, ollama):
    # chroma errors when n_results is more than the collection holds
    fake = use_collection(monkeypatch, rows=ROWS)
    assert ask(n_results=20).status_code == 200
    assert fake.queried_with == 2


def test_ask_drops_duplicate_and_blank_chunks(monkeypatch, ollama):
    rows = [
        ("Same paragraph.", "a.md", 0.2),
        ("Same paragraph.", "b.md", 0.25),  # same text again, from another file
        ("   ", "c.md", 0.3),               # blank, nothing to answer from
        ("Other paragraph.", None, 0.35),   # stored without any metadata
    ]
    use_collection(monkeypatch, rows=rows)
    # 4, the default of 3 would never fetch the last row
    data = ask(n_results=4).json()
    assert_ask_shape(data)
    # the closer copy is kept, and a chunk with no source still gets a name
    assert [(s["source"], s["distance"]) for s in data["sources"]] == [("a.md", 0.2), ("unknown", 0.35)]


# ── /ask: invalid requests (400, 422) ──────────────────────────────────────

@pytest.mark.parametrize("question", ["", "   ", "\n\t ", "\u200b"])
def test_ask_blank_question_is_422(question):
    # the last one is a zero-width space, which strip() leaves behind
    response = ask(question)
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "question"]
    assert error["msg"] == "Value error, question must not be empty"


@pytest.mark.parametrize(
    "body",
    [
        {},                                       # no question at all
        {"question": 42},                         # not a string
        {"question": "x" * 1001},                 # over max_length
        {"question": "ok", "n_results": 0},       # below ge=1
        {"question": "ok", "n_results": 21},      # above le=20
        {"question": "ok", "n_results": "3"},     # strict, no "3" -> 3
        {"question": "ok", "max_distance": 4.5},  # above le=4
    ],
)
def test_ask_invalid_body_is_422(body):
    response = client.post("/ask", json=body)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_ask_malformed_json_is_422():
    response = post_raw('{"question": ')
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "json_invalid"


@pytest.mark.parametrize(
    "body",
    [
        {"question": "hi", "max_distance": float("nan")},
        {"question": "hi", "max_distance": float("inf")},
        {"question": chr(0xD800)},  # a lone surrogate, half of a UTF-16 pair
    ],
    ids=["nan", "infinity", "lone-surrogate"],
)
def test_ask_input_json_cant_echo_back_is_422_not_500(body):
    # Python's json parser accepts all three, but FastAPI's default 422 handler
    # echoes the input back and can't write any of them out as JSON, so they
    # were a 500. main.py's own handler makes them safe first. json.dumps
    # writes NaN and Infinity as bare words and escapes the surrogate
    response = post_raw(json.dumps(body))
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] in {"max_distance", "question"}


def test_ask_body_not_utf8_is_400():
    # é in latin-1 is the single byte 0xE9, which isn't valid UTF-8
    response = post_raw('{"question": "café"}'.encode("latin-1"))
    assert response.status_code == 400
    assert response.json() == {"detail": "There was an error parsing the body"}


# ── /ask: Ollama or ChromaDB down (502, 503) ───────────────────────────────

@pytest.mark.parametrize(
    ("reply", "detail"),
    [
        (requests.exceptions.ConnectionError("connection refused"), "Ollama unavailable"),
        # requests makes SSLError (and ProxyError) a kind of ConnectionError
        (requests.exceptions.SSLError("certificate verify failed"), "Ollama unavailable"),
        (requests.exceptions.Timeout("read timed out"), "Ollama timed out"),
        (requests.exceptions.TooManyRedirects("exceeded 30 redirects"), "Could not reach Ollama (TooManyRedirects)"),
        (FakeResponse(404, {"error": "model 'llama3.2:1b' not found"}), "Ollama returned 404 for model llama3.2:1b"),
        # what Ollama sends when the model process crashes while loading
        (FakeResponse(500, {"error": "llama-server process has terminated"}),
         "Ollama returned 500 for model llama3.2:1b"),
    ],
    ids=["refused", "ssl", "timeout", "redirects", "model-not-pulled", "model-crashed"],
)
def test_ask_503_when_ollama_fails(monkeypatch, reply, detail):
    use_collection(monkeypatch, rows=ROWS)
    ollama_replies(monkeypatch, reply)
    response = ask()
    # the API itself is fine, so 503 (try again later) rather than a 500
    assert response.status_code == 503
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize(
    ("reply", "detail"),
    [
        (requests.exceptions.ChunkedEncodingError("Connection broken"), "Ollama's response was cut off"),
        (FakeResponse(200, text="<html>Bad Gateway</html>"), "Ollama's response had no answer in it"),
        (FakeResponse(200, {"done": True}), "Ollama's response had no answer in it"),
        (FakeResponse(200, {"message": {"role": "assistant", "content": "   "}}), "Ollama returned an empty answer"),
    ],
    ids=["cut-off", "html-body", "no-message", "blank-answer"],
)
def test_ask_502_when_ollamas_reply_is_unusable(monkeypatch, reply, detail):
    # Ollama was reached and replied, just with nothing usable: a bad gateway
    use_collection(monkeypatch, rows=ROWS)
    ollama_replies(monkeypatch, reply)
    response = ask()
    assert response.status_code == 502
    assert response.json() == {"detail": detail}


def test_ask_503_when_chromadb_is_broken(monkeypatch):
    use_collection(monkeypatch, broken=True)
    response = ask()
    assert response.status_code == 503
    assert response.json() == {"detail": "ChromaDB is not readable"}


def test_ask_503_when_the_search_fails(monkeypatch):
    # the db opens and counts fine, but the query (which embeds the question
    # first) fails, e.g. a missing or broken embedding model
    use_collection(monkeypatch, rows=ROWS, search_broken=True)
    response = ask()
    assert response.status_code == 503
    assert response.json() == {"detail": "ChromaDB search failed"}


# ── confidence ─────────────────────────────────────────────────────────────

def chunk(distance, source="chromadb.md"):
    return {"text": "ChromaDB keeps documents and embeddings in collections.",
            "source": source, "distance": distance}


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(0.2, "high"), (0.4999, "high"), (0.5, "medium"), (0.7999, "medium"), (0.8, "low"), (1.15, "low")],
)
def test_confidence_thresholds(distance, expected):
    assert compute_confidence([chunk(distance)]) == expected


def test_confidence_low_with_no_chunks():
    assert compute_confidence([]) == "low"


def test_confidence_low_when_two_files_nearly_tie():
    # 0.03 apart is under AMBIGUITY_GAP (0.05), the question sits between topics
    assert compute_confidence([chunk(0.30, "chromadb.md"), chunk(0.33, "embeddings.md")]) == "low"


def test_confidence_same_file_close_together_isnt_ambiguous():
    assert compute_confidence([chunk(0.30), chunk(0.31)]) == "high"


def test_confidence_low_when_the_model_says_it_cant_answer():
    answer = "I don't have enough information to answer that."
    assert compute_confidence([chunk(0.1)], answer, "What licence is ChromaDB under?") == "low"


def test_confidence_drops_a_level_for_numbers_not_in_the_sources():
    # 1536 isn't in the question or the chunk, so it came from somewhere else
    assert compute_confidence([chunk(0.1)], "It uses 1536 dimensions.", "How many dimensions?") == "medium"


def test_ask_borderline_match_comes_back_low(monkeypatch, ollama):
    use_collection(monkeypatch, rows=[("Embeddings are lists of numbers.", "embeddings.md", 0.93)])
    data = ask("Tell me about vectors").json()
    assert_ask_shape(data)
    # still answered from it, just flagged
    assert data["confidence"] == "low"
    assert len(data["sources"]) == 1


def test_ask_ambiguous_match_comes_back_low(monkeypatch, ollama):
    rows = [("ChromaDB stores vectors.", "chromadb.md", 0.42),
            ("Embeddings are vectors.", "embeddings.md", 0.45)]
    use_collection(monkeypatch, rows=rows)
    data = ask("What stores vectors?").json()
    assert_ask_shape(data)
    assert data["confidence"] == "low"


# ── /ingest ────────────────────────────────────────────────────────────────

@pytest.fixture
def docs_dir(tmp_path, monkeypatch):
    (tmp_path / "chromadb.md").write_text("# ChromaDB\n\nStores embeddings.\n\nRuns locally.")
    (tmp_path / "notes.txt").write_text("Plain text works too.")
    (tmp_path / "diagram.png").write_bytes(b"\x89PNG")  # not .md or .txt, skipped
    monkeypatch.setattr(main, "DOCS_DIR", str(tmp_path))
    return tmp_path


def test_ingest_loads_md_and_txt_files(monkeypatch, docs_dir):
    # deleted.md_0 is left over from a file that's gone, so it gets removed
    fake = use_collection(monkeypatch, stored_ids=["deleted.md_0"])
    response = client.post("/ingest")
    assert response.status_code == 200
    assert response.json() == {
        "chunks_ingested": 3,
        "files_ingested": 2,
        "chunks_removed": 1,
        "message": "Ingested 3 chunks from 2 files, removed 1 old chunk",
    }
    # the heading is merged into the paragraph under it, not its own chunk
    assert fake.ids == {"chromadb.md_0", "chromadb.md_1", "notes.txt_0"}


def test_ingest_empty_docs_folder_changes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DOCS_DIR", str(tmp_path))
    fake = use_collection(monkeypatch, stored_ids=["chromadb.md_0"])
    response = client.post("/ingest")
    assert response.status_code == 200
    assert response.json()["chunks_ingested"] == 0
    assert fake.ids == {"chromadb.md_0"}


def test_ingest_503_when_chromadb_cant_be_written(monkeypatch, docs_dir):
    use_collection(monkeypatch, broken=True)
    response = client.post("/ingest")
    assert response.status_code == 503
    assert response.json() == {"detail": "Could not write to ChromaDB"}


# ── routing ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("method", "path"), [("get", "/ask"), ("get", "/ingest"), ("post", "/health")])
def test_wrong_method_is_405(method, path):
    assert getattr(client, method)(path).status_code == 405
