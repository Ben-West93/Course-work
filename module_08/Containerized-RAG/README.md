# Containerized RAG Assistant

A question-answering chatbot over a small set of course notes (Python, FastAPI,
Streamlit, Docker and RAG). The FastAPI backend chunks the files in
`backend/docs/`, stores them in ChromaDB, retrieves the closest chunks for a
question and has `llama3.2:1b` (running in Ollama) answer from them. A Streamlit
chat UI sits in front. All three services start with one `docker-compose up`.

## Architecture

```
                         browser
                            │  http://localhost:8501
                            ▼
             ┌──────────────────────────────┐
             │ frontend                     │
             │ Streamlit chat UI            │
             │ history in st.session_state  │
             └──────────────┬───────────────┘
                            │  BACKEND_URL=http://backend:8000
                            │  GET /health, POST /ingest, POST /ask
                            ▼
             ┌──────────────────────────────┐   OLLAMA_URL=http://ollama:11434
             │ backend                      │   POST /api/chat, GET /api/tags
             │ FastAPI + uvicorn            │                       ┌──────────────────────────────┐
             │ main.py    endpoints         ├──────────────────────►│ ollama                       │
             │ rag.py     RAG pipeline      │                       │ llama3.2:1b                  │
             │ config.py  env settings      │                       └──────────────┬───────────────┘
             │ ChromaDB + all-MiniLM-L6-v2  │                                      │
             └──────────────┬───────────────┘                                      │
                            │  /app/rag_db                                         │  /root/.ollama
                            ▼                                                      ▼
             ┌──────────────────────────────┐                       ┌──────────────────────────────┐
             │ volume chroma_data           │                       │ volume ollama_models         │
             │ chunks, embeddings, metadata │                       │ pulled model (~1.3GB)        │
             └──────────────────────────────┘                       └──────────────────────────────┘
```

- The services talk over the default compose network, where each service name
  is a hostname. Inside a container `localhost` is that container, so the
  frontend uses `http://backend:8000` and the backend uses `http://ollama:11434`.
- Both named volumes live outside the containers. `docker-compose down` and
  image rebuilds keep the indexed chunks and the pulled model, only
  `docker-compose down -v` deletes them.
- Start order: Ollama first (backend waits for its healthcheck), then the
  backend, then the frontend. The backend has a healthcheck on `/health` too.
- Published ports are bound to `127.0.0.1`, so nothing else on the network can
  reach them.

## Project structure

```
Containerized-RAG/
├── docker-compose.yml          three services + two named volumes
├── .env.example                documented settings (copy to .env)
├── .github/workflows/ci.yml    tests, docker builds, lint
├── ruff.toml
├── backend/
│   ├── Dockerfile
│   ├── main.py                 FastAPI app: /, /ask, /ingest, /stats, /health
│   ├── rag.py                  chunking, ChromaDB, prompt, Ollama call, confidence
│   ├── config.py               Settings read from environment variables
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── docs/                   the corpus (.txt / .md)
│   └── tests/test_api.py
└── frontend/
    ├── Dockerfile
    ├── app.py                  Streamlit chat UI
    └── requirements.txt
```

The `*.starter.*` files are the course starter code, kept unchanged for
reference. Ruff, pytest and both Docker builds ignore them.

## Setup

Needs Docker Desktop with Compose v2.24 or newer (for the optional `env_file`
and `required: false` in `depends_on`).

```bash
cd module_08/Containerized-RAG
cp .env.example .env            # optional, the defaults work without it
docker-compose up --build -d
```

Pull the model into the Ollama volume. This is only needed once:

```bash
docker-compose exec ollama ollama pull llama3.2:1b
```

Check everything is up:

```bash
docker-compose ps
curl http://localhost:8000/health
```

`/health` should say `"status": "ok"`. If it says `degraded`, `ollama` and
`model_pulled` tell you which part is missing.

If Ollama is already installed and running on your machine it holds port 11434,
and the ollama container can't publish on it. Set `OLLAMA_HOST_PORT=11435` in
`.env`. The backend doesn't care, it reaches Ollama over the compose network.

Stop with `docker-compose down`. Use `docker-compose down -v` only if you want to
wipe the index and the downloaded model.

## Using the UI

Open http://localhost:8501.

1. The sidebar shows whether the backend is reachable, ChromaDB and Ollama
   status, and how many chunks are indexed. It warns when Ollama is down or the
   model isn't pulled.
2. Click **Re-index Documents** the first time (and whenever you change
   `backend/docs/`). It calls `POST /ingest`. Re-indexing is safe to repeat.
3. Type a question in the box at the bottom. Each answer shows a confidence
   badge (green high, orange medium, red low) and a **Sources** expander listing
   the chunks the model was given, with file name and distance.
4. The conversation is kept in `st.session_state` for as long as the tab is
   open. **Clear chat** empties it.

Questions that work well: "How does FastAPI validate request bodies?", "What is
a list comprehension?", "What is session state in Streamlit?", "What is a named
volume?". Something off-topic like "Who won the 2018 World Cup?" gets the
no-match fallback.

To add your own material, drop `.txt` or `.md` files into `backend/docs/`,
rebuild the backend (`docker-compose up -d --build backend`), and re-index.

## Configuration

All settings come from environment variables (`backend/config.py`). Compose
passes `.env` to the backend with `env_file`, and always sets `OLLAMA_URL` and
`CHROMA_PATH` itself. Bad values (`MAX_RESULTS=abc`, `DEBUG=maybe`) stop the
backend at startup with an error naming the variable. A blank value counts as
unset.

| Variable               | Default                  | Used for                                                      |
| ---------------------- | ------------------------ | ------------------------------------------------------------- |
| `OLLAMA_URL`           | `http://localhost:11434` | Ollama server. Compose sets `http://ollama:11434`             |
| `MODEL_NAME`           | `llama3.2:1b`            | model used by `/ask`                                          |
| `CHROMA_PATH`          | `./rag_db`               | ChromaDB folder (relative to `backend/`). Compose: `/app/rag_db` |
| `MAX_RESULTS`          | `3`                      | default `n_results` for `/ask` (1-20)                         |
| `CONFIDENCE_THRESHOLD` | `1.0`                    | top-chunk distance above this makes confidence `low`          |
| `MAX_DISTANCE`         | `1.2`                    | default `max_distance` for `/ask`, chunks past it are dropped  |
| `DEBUG`                | `false`                  | logs retrieval distances and Ollama timings                   |
| `OLLAMA_HOST_PORT`     | `11434`                  | compose only: host port for the ollama container              |
| `BACKEND_URL`          | `http://localhost:8000`  | frontend. Compose sets `http://backend:8000`                  |

`MAX_DISTANCE` isn't in the starter's list. The starter hardcoded
`max_distance = 1.2` on the request model; it's an env var here so nothing in the
pipeline is hardcoded. See [Confidence scoring](#confidence-scoring) for why the
two distance settings are separate.

## API reference

Base URL `http://localhost:8000`. Interactive docs at `/docs` (Swagger UI).

### `GET /`

Liveness only, doesn't touch ChromaDB or Ollama.

```bash
curl http://localhost:8000/
```

```json
{ "message": "RAG API is running", "status": "ok", "docs": "/docs" }
```

### `POST /ask`

Retrieves the closest chunks, sends them to the model as context and returns the
answer with its sources and a confidence level.

Body (JSON object):

| Field          | Type   | Required | Default         | Validation                                                                 |
| -------------- | ------ | -------- | --------------- | -------------------------------------------------------------------------- |
| `question`     | string | yes      |                 | 1-1000 chars after trimming. Must contain a visible character (whitespace, zero-width spaces, BOMs and control characters alone are rejected) |
| `n_results`    | int    | no       | `MAX_RESULTS`   | 1-20. Strict: `"5"`, `2.5` and `true` are rejected                          |
| `max_distance` | float  | no       | `MAX_DISTANCE`  | 0-4, finite. Strict: strings rejected                                      |

Response (`AskResponse`):

| Field        | Type                                | Notes                                                          |
| ------------ | ----------------------------------- | -------------------------------------------------------------- |
| `answer`     | string                              | never empty. A fixed fallback text when there's nothing to answer from |
| `sources`    | list of `{text, source, distance}`  | closest first, each text once, all within `max_distance`. `[]` on fallback. `source` is `"unknown"` if a chunk has no usable file name |
| `confidence` | `"high"` / `"medium"` / `"low"`     | see [Confidence scoring](#confidence-scoring)                  |
| `status`     | `"answered"` / `"no_documents"` / `"no_match"` | why there are or aren't sources               |

If nothing is indexed, or no chunk is within `max_distance`, the model isn't
called at all. That fallback still works with Ollama down.

Answered:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does FastAPI validate request bodies?"}'
```

```json
{
  "answer": "FastAPI automatically validates the request body against the model defined in the Pydantic model. According to the provided context, FastAPI uses Pydantic models for request and response validation.\n\nWhen a request is made to a FastAPI endpoint, the `@app.post(\"/items\")` decorator is used to define a new endpoint. Inside this decorator, the `item` variable is of type `Item` (defined by the `Item` Pydantic model).\n\nFastAPI then checks if the request body conforms to the `Item` model. If it does, the request is processed and the response is generated. If the request body does not conform to the model, a 422 error is returned.\n\nSource: fastapi_basics.txt, Section: Introduction to FastAPI, Section: Request Bodies with Pydantic",
  "sources": [
    {
      "text": "FastAPI automatically validates the request body against the model and returns a 422 error if validation fails.",
      "source": "fastapi_basics.txt",
      "distance": 0.375
    },
    {
      "text": "Introduction to FastAPI\n\nFastAPI is a modern, high-performance web framework for building APIs with Python. It is built on top of Starlette for the web layer and Pydantic for data validation. FastAPI automatically generates interactive API documentation at /docs (Swagger UI) and /redoc.",
      "source": "fastapi_basics.txt",
      "distance": 0.7614
    },
    {
      "text": "Request Bodies with Pydantic\n\nFastAPI uses Pydantic models for request and response validation. Define a model by subclassing BaseModel:\n\n    from pydantic import BaseModel\n\n    class Item(BaseModel):\n        name: str\n        price: float\n        in_stock: bool = True\n\n    @app.post(\"/items\")\n    def create_item(item: Item):\n        return item",
      "source": "fastapi_basics.txt",
      "distance": 0.8113
    }
  ],
  "confidence": "high",
  "status": "answered"
}
```

Borderline match (top chunk between 1.0 and 1.2), answered but flagged. Note the
answer is partly wrong (Pydantic isn't "part of FastAPI"), which is exactly what
the low flag is for:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Pydantic?"}'
```

```json
{
  "answer": "Pydantic is a Python library used for building robust, fast, and scalable data models. It's a part of the FastAPI framework, which is a modern, fast, and secure framework for building APIs with Python 3.7+.\n\nPydantic models are used to define the structure and validation of data. They provide a way to define the shape of data, including the types of fields, their constraints, and their relationships. This makes it easier to validate and sanitize data, reducing the risk of errors and security vulnerabilities.\n\nIn the context of the provided code, the `Item` model is defined using Pydantic, which is used to validate the request body and return a validated response.",
  "sources": [
    {
      "text": "Request Bodies with Pydantic\n\nFastAPI uses Pydantic models for request and response validation. Define a model by subclassing BaseModel:\n\n    from pydantic import BaseModel\n\n    class Item(BaseModel):\n        name: str\n        price: float\n        in_stock: bool = True\n\n    @app.post(\"/items\")\n    def create_item(item: Item):\n        return item",
      "source": "fastapi_basics.txt",
      "distance": 1.1863
    }
  ],
  "confidence": "low",
  "status": "answered"
}
```

Nothing close enough:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who won the 2018 World Cup?"}'
```

```json
{
  "answer": "I couldn't find anything in the documents close enough to this question to answer it. Try rephrasing it, or ask about Python, FastAPI, Streamlit, Docker or RAG.",
  "sources": [],
  "confidence": "low",
  "status": "no_match"
}
```

Nothing indexed yet:

```json
{
  "answer": "No documents have been ingested yet. Click Re-index Documents (POST /ingest) first.",
  "sources": [],
  "confidence": "low",
  "status": "no_documents"
}
```

Validation failure:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "   "}'
```

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "question"],
      "msg": "Value error, question must not be empty",
      "input": "   ",
      "ctx": { "error": {} }
    }
  ]
}
```

Ollama not running (`docker-compose stop ollama`), HTTP 503:

```json
{ "detail": "Ollama unavailable" }
```

### `POST /ingest`

No body. Reads every `.txt` and `.md` file in `backend/docs/`, chunks them and
upserts into ChromaDB. Chunk ids are `<file>_<index>`, so running it again
overwrites instead of duplicating, and chunks whose file was deleted or got
shorter are removed. An empty or missing `docs/` folder changes nothing. Doesn't
need Ollama.

```bash
curl -X POST http://localhost:8000/ingest
```

```json
{
  "chunks_ingested": 56,
  "files_ingested": 5,
  "chunks_removed": 0,
  "message": "Ingested 56 chunks from 5 files"
}
```

### `GET /stats`

```bash
curl http://localhost:8000/stats
```

```json
{
  "document_count": 56,
  "model": "llama3.2:1b",
  "db_path": "/app/rag_db",
  "max_results": 3,
  "confidence_threshold": 1.0,
  "max_distance": 1.2
}
```

`document_count` is the number of chunks, not files.

### `GET /health`

Checks ChromaDB (`collection.count()`) and Ollama (`GET /api/tags`, which is
cheap and lists the pulled models).

| `status`   | HTTP | Meaning                                                                         |
| ---------- | ---- | ------------------------------------------------------------------------------- |
| `ok`       | 200  | ChromaDB readable, Ollama connected, `MODEL_NAME` pulled                         |
| `degraded` | 200  | ChromaDB fine, but Ollama is down or the model isn't pulled. `/ingest` and the fallbacks work, `/ask` can't generate |
| `error`    | 503  | ChromaDB can't be read. `document_count` is `0` and `chromadb` is `"error"`     |

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "chromadb": "ok",
  "ollama": "connected",
  "document_count": 56,
  "model": "llama3.2:1b",
  "model_pulled": true
}
```

With the ollama container stopped:

```json
{
  "status": "degraded",
  "chromadb": "ok",
  "ollama": "disconnected",
  "document_count": 56,
  "model": "llama3.2:1b",
  "model_pulled": false
}
```

### Error response matrix

| Code | Endpoint            | When                                                          | Body                                                                 |
| ---- | ------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------- |
| 200  | all                 | success, including the `no_documents` / `no_match` fallbacks and `degraded` health | the response models above                      |
| 422  | `/ask`              | `question` missing, null, not a string, blank or invisible-only, over 1000 chars | `{"detail": [{"type", "loc", "msg", "input", ...}]}` (pydantic errors) |
| 422  | `/ask`              | `n_results` not an int, or outside 1-20                       | e.g. `"type": "greater_than_equal", "loc": ["body", "n_results"]`    |
| 422  | `/ask`              | `max_distance` not a number, outside 0-4, NaN or Infinity     | e.g. `"type": "finite_number", "loc": ["body", "max_distance"]`      |
| 422  | `/ask`              | body isn't JSON, or is JSON but not an object                 | `"type": "json_invalid"` / `"model_attributes_type"`                 |
| 503  | `/ask`              | Ollama refused / unreachable                                  | `{"detail": "Ollama unavailable"}`                                   |
| 503  | `/ask`              | Ollama took longer than 120s                                  | `{"detail": "Ollama timed out"}`                                     |
| 503  | `/ask`              | model not pulled (Ollama 404) or any other non-200            | `{"detail": "Ollama returned 404 for model llama3.2:1b"}`            |
| 503  | `/ask`              | Ollama replied 200 but with no usable answer                  | `{"detail": "Ollama's response had no answer in it"}` / `{"detail": "Ollama returned an empty answer"}` |
| 503  | `/ask`              | ChromaDB can't be searched                                    | `{"detail": "ChromaDB search failed"}`                               |
| 503  | `/ingest`           | ChromaDB can't be written (safe to retry)                     | `{"detail": "Could not write to ChromaDB"}`                          |
| 503  | `/stats`            | ChromaDB can't be read                                        | `{"detail": "ChromaDB is not readable"}`                             |
| 503  | `/health`           | ChromaDB can't be read                                        | the normal health body with `"status": "error"`                      |

FastAPI also returns `404 {"detail": "Not Found"}` for unknown paths and
`405 {"detail": "Method Not Allowed"}` for e.g. `GET /ask`.

Upstream failures are 503 rather than 500 because the API itself is fine and a
retry can work once the dependency is back. The response only carries the short
reason, the full cause (for example Ollama's own error text) goes to
`docker-compose logs backend`.

## Chunking

`rag.chunk_text()` splits each file on blank lines, then fixes the cases where
that cuts something in the wrong place:

- A heading on its own line ("Python Functions", "What is Ollama?") is joined to
  the paragraph after it. Alone it has nothing to answer from, but it embeds
  very close to short questions and would take up a source slot.
- A lead-in ending in `:` is treated the same way, and indented code blocks are
  attached to the paragraph above, even across blank lines. So "A minimal
  FastAPI app requires only a few lines:" stays with its code.
- A paragraph under 80 characters is joined to the previous one ("This handles
  requests like: GET /search?q=python&limit=5").
- Anything over 1000 characters is split at sentence ends, since
  all-MiniLM-L6-v2 only reads about the first 256 tokens.

Each chunk is stored with metadata `{"source": <file name>, "chunk_index": <n>}`.
The five files give 56 chunks.

## Confidence scoring

ChromaDB's default distance is squared L2. The embeddings are unit vectors, so
0 is identical, 1.0 is cosine similarity 0.5, and 4 is opposite.

`rag.compute_confidence()` looks at the closest chunk and returns:

| Confidence | Rule                                                                         |
| ---------- | ---------------------------------------------------------------------------- |
| `high`     | top chunk < 0.5                                                              |
| `medium`   | top chunk 0.5 to 1.0 (`CONFIDENCE_THRESHOLD`)                                |
| `low`      | no chunks                                                                    |
| `low`      | top chunk > 1.0: borderline, it passed the 1.2 cutoff but is loosely related |
| `low`      | the closest chunk from a different file is within 0.05: retrieval can't tell which topic was meant |
| `low`      | the model says it doesn't have enough information                            |
| `low`      | the answer cites a `.txt`/`.md` file that wasn't in its context               |

The downgrades always go straight to `low`, so a client can treat `low` as
"check the sources before trusting this".

### Where the numbers come from

I indexed the corpus and ran 80-odd probe questions to see where real matches
and noise land:

| Question                                            | Top chunk | Result                    |
| --------------------------------------------------- | --------- | ------------------------- |
| How does FastAPI validate request bodies?           | 0.375     | high, correct             |
| What is a list comprehension?                       | 0.584     | medium, correct           |
| What is a named volume?                             | 0.929     | medium, correct           |
| What are RAG guardrails?                            | 1.062     | low, right section        |
| How does the frontend container reach the backend?  | 1.130     | low, answer was wrong     |
| What is Pydantic?                                   | 1.186     | low, answer partly wrong  |
| How do I store data? (vague)                        | 1.228     | dropped                   |
| Who won the 2018 World Cup?                         | 1.747     | dropped                   |

- Nothing off-topic came in under 1.23, while plenty of answerable questions
  sat between 1.0 and 1.19. A 1.0 cutoff would have thrown away questions like
  "What are RAG guardrails?", so the cutoff (`MAX_DISTANCE`) is the starter's 1.2.
  Short questions about a single identifier still miss ("What does depends_on
  do?" is 1.289), the embedding model just doesn't place them close to prose.
- Between 1.0 and 1.2 roughly one top chunk in five was the wrong section
  ("How do I keep data between restarts?" hit the `--reload` paragraph), and
  even right-section answers like the two above came out wrong. So that band is
  answered but marked `low`.
- Questions about two topics at once still had clear winners (smallest gap
  0.10). Near-ties under 0.05 came from noise, or from genuinely ambiguous
  questions like "What is the chat endpoint?", where Ollama's `/api/chat`
  (0.924) and Streamlit's chat elements (0.968) are 0.044 apart. That one comes
  back `low` with both sources listed. With `"max_distance": 0.93` the Streamlit
  chunk is stripped and it's `medium`.

## Running without Docker

```bash
# backend (needs Ollama on localhost:11434 for /ask)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# frontend, in a second terminal
cd frontend
pip install -r requirements.txt
BACKEND_URL=http://localhost:8000 streamlit run app.py
```

`python rag.py` in `backend/` ingests `docs/` and prints the retrieved chunks
and confidence for a few sample questions.

## Tests

```bash
cd backend
pytest tests/ -v
```

The suite (74 tests) needs neither Ollama nor Docker. It points ChromaDB at a
temporary folder and `OLLAMA_URL` at a closed port, and patches
`requests.post` where it needs Ollama to answer or fail in a specific way.
Covered:

- `test_root`, `test_health`, `test_stats`: status codes, keys and types
- `/health` in the ok, degraded (Ollama down, model missing) and error (ChromaDB broken) states
- `/ingest` counts, re-ingest without duplicates, stale chunk removal, empty `docs/`
- `/ask` answered, both fallbacks, `max_distance` stripping sources, the borderline case
- 18 invalid bodies and 5 unparseable ones (including NaN and lone surrogates) all give 422, not 500
- every Ollama failure mode giving 503 with the right detail
- every confidence rule, plus chunking and retrieval edge cases (CRLF, bad bytes,
  hidden files, odd metadata, duplicate text)

Lint with `ruff check backend/ frontend/` from the project folder.

## CI

`.github/workflows/ci.yml` runs three jobs in parallel on every push and pull
request to `main`: the pytest suite on Python 3.11, both Docker image builds
plus a `docker compose config` check, and ruff. In the Course-work repo GitHub
only reads workflows from the repository root, so the same jobs run from
`.github/workflows/containerized-rag-ci.yml` there, limited to changes in this
folder.

## Troubleshooting

| Symptom                                           | Fix                                                                  |
| ------------------------------------------------- | -------------------------------------------------------------------- |
| `ports are not available: exposing port TCP 127.0.0.1:11434 ... address already in use` | Ollama is running on the host. Set `OLLAMA_HOST_PORT=11435` in `.env` |
| `/ask` 503 `Ollama returned 404 for model ...`    | `docker-compose exec ollama ollama pull llama3.2:1b`                 |
| `/ask` 503 `Ollama unavailable`                   | `docker-compose start ollama`                                        |
| first answer is slow                              | the model is loading into memory, later ones are faster              |
| sidebar shows 0 documents                         | click Re-index Documents                                             |
| answers seem to ignore a new file in `docs/`      | `docker-compose up -d --build backend`, then re-index                |
