# RAG Application CI

[![RAG Application CI](https://github.com/Ben-West93/Course-work/actions/workflows/rag-application-ci.yml/badge.svg)](https://github.com/Ben-West93/Course-work/actions/workflows/rag-application-ci.yml)

GitHub Actions CI for the RAG app from the containerization exercise (FastAPI backend, Streamlit frontend). Every push and pull request to `main` runs three jobs: the pytest suite, a build of both Docker images, and ruff.

```
RAG-Application-CI/
├── .github/
│   └── workflows/
│       ├── ci.yml                 the workflow: test, docker, lint
│       └── ci.starter.yml         course starter, kept as handed out
├── backend/
│   ├── Dockerfile                 built by the docker job
│   ├── main.py                    FastAPI RAG API
│   ├── config.py                  Settings class, reads env vars and .env
│   ├── requirements.txt           fastapi, uvicorn, chromadb, requests, pydantic, python-dotenv
│   ├── pytest.ini                 puts backend/ on sys.path, skips the starter
│   ├── docs/                      7 .md/.txt files, 32 chunks once ingested
│   └── tests/
│       ├── __init__.py
│       ├── test_api.py            68 tests, no Ollama or embedding model needed
│       └── test_api.starter.py    course starter, kept as handed out
├── frontend/
│   ├── Dockerfile                 built by the docker job
│   ├── app.py                     Streamlit UI
│   └── requirements.txt           streamlit, requests
├── ruff.toml                      lint settings for the local run and the lint job
├── .env.example                   every setting, with the values CI uses
└── README.md
```

## Where the workflow runs from

GitHub only runs workflow files from `.github/workflows/` at the root of a repository. This project is a folder inside `Course-work`, so `.github/workflows/ci.yml` here is never picked up on its own. The copy that runs is [`/.github/workflows/rag-application-ci.yml`](../../.github/workflows/rag-application-ci.yml) at the repo root. It's the same workflow with these differences:

| | `ci.yml` (this folder) | `rag-application-ci.yml` (repo root) |
|---|---|---|
| Default working directory | repo root (= this folder if it were its own repo) | `module_08/RAG-Application-CI`, set once for the whole workflow |
| `test` job working directory | `./backend` | `module_08/RAG-Application-CI/backend` |
| pip cache key file | `backend/requirements.txt` | `module_08/RAG-Application-CI/backend/requirements.txt` |
| Triggers | every push/PR to `main` | push/PR to `main` that touches this folder or the workflow file |

Any change to one file has to go into the other too. If the project ever moves to its own repo, `ci.yml` works as it is, but `ci.starter.yml` would have to come out of `.github/workflows/` first. GitHub reads every `.yml` in that folder, and the starter's `name: # TODO` and `run: # TODO` lines parse as null, which makes it an invalid workflow.

## The pipeline

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
```

A push to `main` triggers a run, and so does opening or updating a PR that targets `main`. The three jobs have no `needs:` between them, so they start together on separate fresh runners and one failing doesn't cancel the others. The workflow token only gets `contents: read`.

| Job | Runner | Working directory | Steps | Timeout |
|---|---|---|---|---|
| `test` | `ubuntu-latest` | `./backend` | checkout, Python 3.11, `pip install -r requirements.txt` + `pytest httpx`, `pytest tests/ -v --tb=short` | 10 min |
| `docker` | `ubuntu-latest` | project root | checkout, `docker build -t rag-backend ./backend`, `docker build -t rag-frontend ./frontend`, check both images exist | 20 min |
| `lint` | `ubuntu-latest` | project root | checkout, Python 3.11, `pip install ruff==0.16.10`, `ruff check backend/ frontend/` | 5 min |

The timeouts are there because GitHub's default is 6 hours, so a test stuck on a network call would hold a runner that long.

### test

- `defaults.run.working-directory: ./backend` only applies to `run:` steps. `uses:` steps ignore it, which is why `cache-dependency-path` is written from the repo root.
- `httpx` isn't in `requirements.txt` because the image doesn't need it, but `fastapi.testclient` imports it. Without the extra `pip install pytest httpx` the suite fails on import.
- The run step sets two env vars:

  | Variable | Value | Why |
  |---|---|---|
  | `OLLAMA_URL` | `http://localhost:11434` | `config.py` refuses a URL without `http(s)://`. Nothing listens there in CI, and every Ollama call in the tests is faked anyway |
  | `CHROMA_PATH` | `./test_chroma` | the real collection the tests open (relative to `backend/`), kept apart from the `rag_db/` uvicorn uses |

- With `httpx` the run prints one warning, `Using httpx with starlette.testclient is deprecated; install httpx2 instead`. The tests still pass. The local venv has `httpx2` installed, so the warning doesn't show up locally.

### docker

Both builds use the Dockerfiles from the containerization exercise. The backend build is the slow one, since it installs chromadb's dependencies and downloads the embedding model into the image. `backend/.dockerignore` keeps `tests/`, `pytest.ini`, `test_chroma/` and `rag_db/` out of the image, which ends up holding only `main.py`, `config.py`, `requirements.txt` and `docs/`.

The verify step runs `docker images rag-backend && docker images rag-frontend` to print the two images, then `docker image inspect rag-backend rag-frontend`. That second command matters because `docker images <name>` exits 0 even when nothing matches, so on its own it can't fail the step. `inspect` exits 1 if either image is missing.

### lint

ruff's default rule set keeps growing between releases. 0.16.10 already turns on import sorting (`I`), blind `except` (`BLE`), pytest style (`PT`) and a lot more. An unpinned `pip install ruff` could fail CI on code that passes locally, so the job installs the same version as the venv. Settings are in `ruff.toml`:

- `target-version = "py311"`, the version CI and both images run
- `src = ["backend", "frontend"]`, so `config` and `main` count as first-party imports when sorting
- `extend-exclude = ["*.starter.py"]`. The starter test file has an unused import and placeholder `pass` lines, 5 errors that would fail the job

### Dependency caching

| What | Cached? | How |
|---|---|---|
| pip downloads in `test` | yes | `actions/setup-python` with `cache: pip`, keyed on the hash of `backend/requirements.txt`. A requirements change makes a new cache |
| ruff in `lint` | no | a single wheel, not worth a cache entry |
| Docker layers in `docker` | no | every run starts on a fresh runner with an empty Docker cache, so both images build from scratch |

## Running the checks locally

From the project folder, with the repo's venv (it sits at the repo root, two folders up):

```bash
cd backend
../../../.venv/bin/pytest tests/ -v
```

pytest finds `backend/pytest.ini` from whatever path it's given, so `pytest backend/tests` from the project folder and `.venv/bin/pytest module_08/RAG-Application-CI/backend/tests` from the repo root run the same tests.

Or with any environment that has the backend requirements:

```bash
cd backend
pip install -r requirements.txt pytest httpx
pytest tests/ -v
```

`tests/test_api.py` sets `OLLAMA_URL` and `CHROMA_PATH` to the CI values when they aren't already set, and pins the other settings to `config.py`'s defaults, so a local run behaves like CI even with a `.env` around. A single test or group:

```bash
pytest tests/test_api.py::test_health -v
pytest tests/ -v -k confidence
```

Lint, from the project folder:

```bash
../../.venv/bin/ruff check backend/ frontend/
```

The docker job, from the project folder (Docker Desktop has to be running):

```bash
docker build -t rag-backend ./backend && docker build -t rag-frontend ./frontend && docker image inspect rag-backend rag-frontend > /dev/null && echo ok
```

The venv is Python 3.13 and CI is 3.11. To run the suite on 3.11 with a fresh install, like the test job does:

```bash
docker run --rm -v "$PWD/backend:/src:ro" python:3.11-slim sh -c 'cp -r /src /work && cd /work && pip install -q -r requirements.txt pytest httpx && OLLAMA_URL=http://localhost:11434 CHROMA_PATH=./test_chroma pytest tests/ -v --tb=short'
```

## Checking a run on GitHub

1. Push to `main` (or open a PR into it) with a change inside `module_08/RAG-Application-CI/`.
2. Open the repo's **Actions** tab and pick **RAG Application CI** on the left.
3. Click the run for your commit. The summary shows `test`, `docker` and `lint`, each with a green check or a red cross.
4. Click a job to see each step's log. A failed step is expanded automatically, and `pytest --tb=short` keeps the failure output short.
5. **Re-run jobs → Re-run failed jobs** reruns only the ones that failed.

The same from the terminal:

```bash
gh run list --workflow "RAG Application CI" --limit 5
```

```bash
gh run watch --exit-status
```

```bash
gh run view --log-failed
```

## Notes and known limits

- **Node 20 warning.** Every job gets an annotation saying `actions/checkout@v4` and `actions/setup-python@v5` target Node 20, which GitHub has deprecated, so they're forced onto Node 24. They still work. The exercise asks for these versions. `checkout@v5` and `setup-python@v6` are the Node 24 releases if the warning needs to go.
- **`ubuntu-latest` is moving to Ubuntu 26** from October 19, 2026 (a notice on every job). `setup-python` already has a Python 3.11.17 build for 26.04, so the test and lint jobs should carry on as they are, but the runner's Docker version and system packages will change with it. `runs-on: ubuntu-24.04` would hold the runner image still if that ever matters.
- **Formatting isn't checked.** The lint step keeps the starter's name, "Check code formatting", but `ruff check` is the linter. `ruff format --check backend/ frontend/` would currently want to reformat 4 files.
- **The images are built, not started.** A Dockerfile with a broken `CMD` would still pass the docker job. Both images do start: run locally, the backend's `/health` answers 200 (`degraded`, there's no Ollama in the container) and the frontend's `/_stcore/health` answers `ok`. To check by hand after building the images locally (each needs a few seconds before it answers, a `curl` straight after `docker run -d` gets an empty reply):

  ```bash
  docker run -d --rm --name smoke-backend -p 127.0.0.1:18000:8000 rag-backend
  docker run -d --rm --name smoke-frontend -p 127.0.0.1:18501:8501 rag-frontend
  sleep 5
  curl http://localhost:18000/health
  curl http://localhost:18501/_stcore/health
  docker stop smoke-backend smoke-frontend
  ```

- **The backend build needs the network.** It downloads Chroma's embedding model during `docker build`. If the docker job fails on that `RUN` line, look for a download error in the log before looking at the code, and re-run the job.
- **`pytest` and `httpx` aren't pinned.** The test job installs the newest of each. The first run got pytest 9.1.1 and httpx 0.28.1, the same as the venv, but a new major release of either could fail the job with no code change. `pip install pytest==9.1.1 httpx==0.28.1` would stop that.
- **The paths filter and required checks.** The root workflow only runs when something in this folder (or the workflow file) changes. If `test`, `docker` or `lint` were ever made required status checks on `main`, a PR that doesn't touch this folder would wait forever for checks that never start.

## The tests

`backend/tests/test_api.py` has 68 tests (counting parametrized cases) and runs in under a second. The idea is to test the API layer only, so nothing in it needs Ollama, the internet or the embedding model:

- `TestClient(app)` calls the app in-process through httpx. Each request still goes through routing, pydantic validation, the route and the exception handlers, just without uvicorn or a port.
- An autouse fixture replaces `requests.get` and `requests.post` in every test with a function that raises `ConnectionError` straight away. Nothing reaches a real Ollama, even when one is running on the machine, and nothing hangs on a timeout in CI. Tests that need Ollama up use the `ollama` fixture, which returns canned `/api/tags` and `/api/chat` replies and records what was sent.
- `/ask` and `/ingest` tests swap `main.collection` for a `FakeCollection` that returns scripted rows. A real chroma query would first download the 79MB embedding model. The three exercise tests (`test_root`, `test_health`, `test_stats`) use the real, empty `test_chroma` collection, which checks that chroma itself starts in CI. Chroma's anonymized telemetry is on by default, but its `capture()` does nothing in 1.5.9, so opening the collection doesn't reach the network either.
- Error replies from Ollama go through `ollama_replies(monkeypatch, reply)`, where `reply` is either a `FakeResponse` for `/api/chat` to return or an exception for `requests.post` to raise.

| Group | Tests | Checks |
|---|---|---|
| Exercise | `test_root`, `test_health`, `test_stats` | 200s, `status`/`chromadb` in `/health`, `document_count` in `/stats` |
| `/health`, `/stats` | 9 | `ok` with Ollama up; `degraded` (still 200) with Ollama down, another model pulled, only a bare `llama3.2` (that's `:latest`), a junk model list, or a 500 from `/api/tags`; 503 + `document_count: null` with chroma broken; settings come from env; `/stats` 503 |
| `/ask` answers | 9 | exact response shape (no nulls, distances rounded to 4 places and sorted), confidence `high`, payload sent to Ollama, empty db and no-match answers that skip Ollama, question length 1000 and 1000-plus-spaces pass while 1001 fails, `n_results` capped at what's stored, duplicate and blank chunks dropped, missing metadata becomes `"unknown"` |
| `/ask` 400, 422 | 16 | blank, whitespace, newline/tab and zero-width-space questions get the custom validator message; missing field, wrong type, too long, `n_results` 0/21/`"3"`, `max_distance` 4.5, malformed JSON; NaN, Infinity and a lone surrogate come back 422 instead of 500; a body that isn't UTF-8 is 400 |
| `/ask` 502, 503 | 12 | Ollama refused, SSL error, timed out, too many redirects, model not pulled (404), model crashed (500); reply cut off, HTML instead of JSON, no message, blank answer (502); chroma unreadable; chroma search failing |
| Confidence | 13 | distance thresholds, no chunks, two files nearly tied, refusal answer, unsupported number, plus borderline and ambiguous retrievals through `/ask` coming back `low` |
| `/ingest` | 3 | `.md` and `.txt` loaded, other files skipped, headings merged, stale chunks removed; empty folder changes nothing; 503 when chroma can't be written |
| Routing | 3 | wrong method is 405 |

The suite was also run against ten deliberately broken copies of `main.py`: borderline cutoff raised to 1.2, the Ollama 503 turned into a 500, the blank-question validator removed, the ambiguity check switched off, `document_count` renamed, the custom 422 handler removed, the `n_results` cap removed, the duplicate check removed, the empty-answer 502 turned into a 503, and the model-pulled check skipped. Each one made at least one test fail.

## API reference

Start the API from `backend/` (needs Ollama for `/ask`, everything else works without it):

```bash
cd backend
../../../.venv/bin/uvicorn main:app --port 8000
```

Swagger UI is at http://localhost:8000/docs. The examples below are real responses from that server with the default settings and the 7 files in `docs/` ingested.

| Method | Path | Body | Success | Errors |
|---|---|---|---|---|
| GET | `/` | none | 200 | none |
| GET | `/health` | none | 200 (`ok` or `degraded`) | 503 |
| GET | `/stats` | none | 200 | 503 |
| POST | `/ingest` | none | 200 | 503 |
| POST | `/ask` | JSON, see below | 200 | 400, 422, 502, 503 |

None of the endpoints take query parameters. A path that doesn't exist is a 404 `{"detail": "Not Found"}`, and the wrong method on a real path (e.g. `GET /ask`) is a 405 `{"detail": "Method Not Allowed"}`.

### GET /

Says the API is up. Doesn't touch ChromaDB or Ollama.

```bash
curl http://localhost:8000/
```

```json
{
  "message": "RAG API is running",
  "docs": "/docs"
}
```

### GET /health

Checks ChromaDB (`collection.count()`), Ollama (`GET /api/tags`, 3 second timeout) and whether `MODEL_NAME` is pulled.

| Field | Type | Values |
|---|---|---|
| `status` | string | `ok`: everything works. `degraded`: ChromaDB works but Ollama is down or doesn't have the model, `/ingest` and retrieval still work. `error`: ChromaDB can't be read |
| `chromadb` | string | `ok`, `error` |
| `ollama` | string | `connected`, `disconnected` |
| `document_count` | int or null | chunks stored, `null` when ChromaDB can't be read |
| `model` | string | `MODEL_NAME` |
| `model_pulled` | bool | `false` whenever Ollama can't be reached |

`ok` and `degraded` are 200, `error` is 503 with the same body.

```bash
curl -i http://localhost:8000/health
```

200, everything up:

```json
{
  "status": "ok",
  "chromadb": "ok",
  "ollama": "connected",
  "document_count": 32,
  "model": "llama3.2:1b",
  "model_pulled": true
}
```

200, Ollama not running:

```json
{
  "status": "degraded",
  "chromadb": "ok",
  "ollama": "disconnected",
  "document_count": 32,
  "model": "llama3.2:1b",
  "model_pulled": false
}
```

503, ChromaDB unreadable (from `test_health_503_when_chromadb_is_broken`):

```json
{
  "status": "error",
  "chromadb": "error",
  "ollama": "disconnected",
  "document_count": null,
  "model": "llama3.2:1b",
  "model_pulled": false
}
```

### GET /stats

How many chunks are stored, plus the settings the API loaded. Handy for checking a `.env` change took effect.

```bash
curl http://localhost:8000/stats
```

```json
{
  "document_count": 32,
  "model": "llama3.2:1b",
  "db_path": "/path/to/RAG-Application-CI/backend/rag_db",
  "ollama_url": "http://localhost:11434",
  "max_results": 3,
  "confidence_threshold": 1.0,
  "debug": false
}
```

(`db_path` shortened here.) It's always absolute: a relative `CHROMA_PATH` is resolved from `backend/`, wherever uvicorn was started. 503 when ChromaDB can't be read:

```json
{"detail": "ChromaDB is not readable"}
```

### POST /ingest

Splits every `.txt` and `.md` file in `backend/docs/` into paragraph chunks and upserts them. A heading on its own line is merged into the paragraph under it. Chunk ids are `<file>_<index>`, so calling it again overwrites instead of duplicating, and chunks whose file was deleted or got shorter are removed. An empty or missing `docs/` changes nothing. No body.

```bash
curl -X POST http://localhost:8000/ingest
```

```json
{
  "chunks_ingested": 32,
  "files_ingested": 7,
  "chunks_removed": 0,
  "message": "Ingested 32 chunks from 7 files"
}
```

503 when ChromaDB can't be written. Every step is safe to repeat, so calling it again once ChromaDB is back finishes the job:

```json
{"detail": "Could not write to ChromaDB"}
```

### POST /ask

Embeds the question, fetches the closest `n_results` chunks, drops any further than `max_distance`, and sends the rest to the model as context. When nothing is left it returns a fixed answer without calling Ollama, so that case works with Ollama down.

Body (`Content-Type: application/json`):

| Field | Type | Required | Default | Validation |
|---|---|---|---|---|
| `question` | string | yes | | Surrounding whitespace is stripped first (custom `mode="before"` validator). Has to contain at least one visible character, so `""`, `"   "`, `"\n\t"` a lone zero-width space and a lone surrogate are rejected with `question must not be empty`. 1 to 1000 characters after stripping. A non-string is a type error |
| `n_results` | int | no | `MAX_RESULTS` (3) | 1 to 20. Strict, so `"3"` and `true` are rejected instead of converted |
| `max_distance` | float | no | `CONFIDENCE_THRESHOLD` (1.0) | 0 to 4, squared L2 distance (0 identical, ~1.2 loosely related). Strict, and `NaN`/`Infinity` are rejected |

Response:

| Field | Type | Notes |
|---|---|---|
| `answer` | string | the model's answer, or a fixed message when there was nothing to answer from |
| `sources` | list | closest first, each `{text, source, distance}`. `distance` rounded to 4 places, `source` is `"unknown"` if the chunk has none. Empty when nothing was stored or nothing was within `max_distance` |
| `confidence` | string | `high`, `medium` or `low`, see below |

How `confidence` is worked out (`compute_confidence` in `main.py`):

| Situation | Confidence |
|---|---|
| no sources | `low` |
| the answer says it doesn't have enough information | `low` |
| closest source at 0.8 or further (borderline) | `low` |
| two different files within 0.05 of each other at the top (ambiguous) | `low` |
| closest source under 0.5 | `high` |
| closest source from 0.5 to 0.8 | `medium` |
| answer has a number, acronym or link host that isn't in the question or the sources | one level lower |

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does ChromaDB store embeddings?"}'
```

200, close match:

```json
{
  "answer": "ChromaDB stores embeddings using the all-MiniLM-L6-v2 sentence-transformer model. According to the source, it produces 384-dimensional vectors.",
  "sources": [
    {
      "text": "# ChromaDB\n\nChromaDB is an open-source vector database. It stores documents, their embeddings, metadata and ids in collections, and handles the embedding step automatically if you pass raw text.",
      "source": "chromadb.md",
      "distance": 0.4948
    },
    {
      "text": "ChromaDB uses the all-MiniLM-L6-v2 sentence-transformer model by default. It produces 384-dimensional vectors and runs locally on CPU, so no API key is needed.",
      "source": "embeddings.md",
      "distance": 0.8273
    },
    {
      "text": "chromadb.Client() keeps everything in memory and loses it when the script exits. chromadb.PersistentClient(path=\"./rag_db\") writes the data to disk in that folder so it survives restarts.",
      "source": "chromadb.md",
      "distance": 0.9856
    }
  ],
  "confidence": "high"
}
```

200, borderline match. The only source is 0.94 away, so it's `low`, and the answer shows why: most of it isn't in the source at all.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can attackers trick the model?"}'
```

```json
{
  "answer": "Yes, attackers can potentially trick a model into producing incorrect or misleading results. This can be done through various means, such as:\n\n* Manipulating input data to elicit a specific response\n* Using adversarial examples to alter the model's output\n* Exploiting vulnerabilities in the model's architecture or training process\n\nAs stated in the paper \"No single defence is complete\" by the authors, ...",
  "sources": [
    {
      "text": "No single defence is complete. Layering several checks and limiting what the model is allowed to do with its output is the practical approach.",
      "source": "prompt_injection.md",
      "distance": 0.9376
    }
  ],
  "confidence": "low"
}
```

200, nothing within `max_distance` (Ollama isn't called):

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a good sourdough hydration?", "max_distance": 1.0}'
```

```json
{
  "answer": "No relevant documents found for this question. Try rephrasing it or raising max_distance.",
  "sources": [],
  "confidence": "low"
}
```

With nothing ingested yet the answer is `No documents have been ingested yet. Call POST /ingest first.`, also with no sources and `low`.

422, blank question:

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
      "ctx": {"error": {}}
    }
  ]
}
```

422, `n_results` out of range:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "n_results": 50}'
```

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "n_results"],
      "msg": "Input should be less than or equal to 20",
      "input": 50,
      "ctx": {"le": 20}
    }
  ]
}
```

422, body isn't valid JSON:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": '
```

```json
{
  "detail": [
    {
      "type": "json_invalid",
      "loc": ["body", 13],
      "msg": "JSON decode error",
      "input": {},
      "ctx": {"error": "Expecting value"}
    }
  ]
}
```

503, Ollama not running:

```json
{"detail": "Ollama unavailable"}
```

## Error responses

Every error body is JSON. App errors are `{"detail": "<message>"}`. 422s are `{"detail": [...]}` with one entry per problem (`type`, `loc`, `msg`, `input`, `ctx`).

| Status | Endpoint | When | Body |
|---|---|---|---|
| 200 | all | success. `/health` is also 200 when `degraded` | see each endpoint |
| 400 | `/ask` | body isn't valid UTF-8 | `{"detail": "There was an error parsing the body"}` |
| 404 | any | unknown path | `{"detail": "Not Found"}` |
| 405 | any | wrong method, e.g. `GET /ask` | `{"detail": "Method Not Allowed"}` |
| 422 | `/ask` | blank or invisible-only question (whitespace, zero-width space, a lone surrogate like `"\ud800"`) | `msg`: `Value error, question must not be empty` |
| 422 | `/ask` | `question` missing | `type`: `missing` |
| 422 | `/ask` | wrong type, e.g. `"question": 42`, `"n_results": "3"` or `"max_distance": "1.0"` | `type`: `string_type` / `int_type` / `float_type` |
| 422 | `/ask` | out of range: question over 1000 chars, `n_results` outside 1-20, `max_distance` outside 0-4 | `type`: `string_too_long` / `greater_than_equal` / `less_than_equal` |
| 422 | `/ask` | `max_distance` is `NaN` or `Infinity` | `type`: `finite_number` |
| 422 | `/ask` | malformed JSON | `type`: `json_invalid` |
| 502 | `/ask` | Ollama's reply was cut off part way | `{"detail": "Ollama's response was cut off"}` |
| 502 | `/ask` | Ollama's reply isn't JSON, or has no `message.content` | `{"detail": "Ollama's response had no answer in it"}` |
| 502 | `/ask` | the answer text is blank | `{"detail": "Ollama returned an empty answer"}` |
| 503 | `/ask` | Ollama refused the connection. Also SSL and proxy errors, which requests treats as connection errors | `{"detail": "Ollama unavailable"}` |
| 503 | `/ask` | Ollama took longer than 120s | `{"detail": "Ollama timed out"}` |
| 503 | `/ask` | Ollama returned an error status: 404 when the model isn't pulled, 500 when the model process crashed while loading | `{"detail": "Ollama returned 404 for model llama3.2:1b"}` |
| 503 | `/ask` | any other request error, e.g. too many redirects | `{"detail": "Could not reach Ollama (TooManyRedirects)"}` |
| 503 | `/ask`, `/stats` | ChromaDB can't be read | `{"detail": "ChromaDB is not readable"}` |
| 503 | `/ask` | ChromaDB search (or embedding the question) failed | `{"detail": "ChromaDB search failed"}` |
| 503 | `/ingest` | ChromaDB can't be written | `{"detail": "Could not write to ChromaDB"}` |
| 503 | `/health` | ChromaDB can't be read | full health body with `"status": "error"` and `"document_count": null` |

503 rather than 500 for the Ollama and ChromaDB failures: the API itself is fine, the service behind it isn't, and the same request can be retried once it's back. The cause the API saw (for example Ollama's own error text) goes to the uvicorn log, not to the client.

Python's JSON parser accepts `NaN`, `Infinity` and lone surrogates, but none of them can be written back out as JSON. FastAPI's default 422 handler echoes the bad input in the response, so those requests used to end in a 500. `main.py` registers its own handler that makes the echo safe first, and `test_ask_input_json_cant_echo_back_is_422_not_500` keeps it that way.

`Ollama returned 500 for model ...` means the API reached Ollama but Ollama couldn't run the model, for example because its model process crashed while loading (on a Mac this can be a Metal/GPU error). Ollama's own error text is in the uvicorn log. If the same model works in the `ollama/ollama` container, the problem is the local Ollama install rather than the API.

## What would break CI, and what handles it

| Problem | What would happen | Handled by |
|---|---|---|
| workflow only in the project's `.github/` | no run at all | root copy `rag-application-ci.yml` |
| `httpx` not installed | `ImportError` from `fastapi.testclient` | `pip install pytest httpx` |
| `from main import app` can't be found | `ModuleNotFoundError: main` | `tests/__init__.py` + `pythonpath = .` in `pytest.ini` |
| `test_api.starter.py` collected (matches `test_*.py`) | `No module named 'tests.test_api.starter'` | `addopts = --ignore-glob=*.starter.py` |
| starter linted | 5 ruff errors | `extend-exclude` in `ruff.toml` |
| unpinned ruff | new default rules fail the job | `ruff==0.16.10` |
| real call to Ollama | 3s or 120s timeouts, or different results depending on the machine | autouse fixture + fakes |
| real chroma query | 79MB model download on every run | `FakeCollection` |
| `python-dotenv` missing from requirements | `ImportError` in `config.py` | kept in `requirements.txt` |
| `docker images` used as the check | missing image still passes | `docker image inspect` |
| a hung step | runner held for 6 hours | `timeout-minutes` on each job |
