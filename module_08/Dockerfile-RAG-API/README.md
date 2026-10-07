# Dockerfile-RAG-API

The RAG FastAPI backend from `module_08/my_rag_api` built into a Docker image. This folder is the whole build context: `my_rag_api.py`, `requirements.txt` and `docs/` are all here.

`Dockerfile.starter` and `dockerignore.starter` are the original scaffolds, `Dockerfile` and `.dockerignore` are the finished versions.

Changes to `my_rag_api.py` compared with `my_rag_api/`:

- `OLLAMA_URL` comes from an environment variable (default `http://localhost:11434`) so the container can reach Ollama on the host
- markdown headings are merged into the paragraph below them instead of being their own chunk (32 chunks instead of 37, see [Confidence](#confidence))
- `/ingest` deletes chunks whose file was removed or got shorter
- request limits on `/ask`, typed response models for every endpoint, and 502/503 errors for Ollama problems
- confidence also looks at the answer, not just the retrieval distance
- Ollama and ChromaDB failures are logged with the actual cause

To run it without Docker, from this folder (Ollama on the default port works as is):

```
pip install -r requirements.txt
uvicorn my_rag_api:app --reload --port 8000
```

## Configuration

| Variable | Default | What it's for |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama is. In Docker use `http://host.docker.internal:11434` |
| `RAG_DB_PATH` | `rag_db/` next to `my_rag_api.py` | Where ChromaDB stores its files. The tests point it at a temp folder |

The model (`llama3.2:1b`) is set in the code as `MODEL`. It has to be pulled first with `ollama pull llama3.2:1b`, otherwise `/ask` returns 503 and the log says `model "llama3.2:1b" not found, try pulling it first`.

## Build

```
docker build -t my-rag-api .                                  # from this folder
docker build -t my-rag-api module_08/Dockerfile-RAG-API/      # from the repo root
```

The image is 1.28GB. About 435MB of that is `build-essential`: on this Mac (arm64) every package in `requirements.txt` installs from a prebuilt wheel, so the compiler never gets used, and the same Dockerfile without that line builds an 845MB image that works the same. It's kept because the exercise asks for it and a platform without wheels would need it. A multi-stage build (compile in one stage, copy the installed packages into a clean `python:3.11-slim`) would get the size back without losing that.

## Run

```
docker run -p 8000:8000 my-rag-api
docker run -d -p 8000:8000 --name rag-api-container my-rag-api
```

- `-p 8000:8000` maps host port 8000 to container port 8000
- `-d` runs it in the background, `--name` lets you use `rag-api-container` instead of the ID

Inside the container `localhost` is the container, not the Mac, so `/ask` can't reach Ollama with the default URL. To use the LLM:

```
docker run -d -p 8000:8000 -e OLLAMA_URL=http://host.docker.internal:11434 --name rag-api-container my-rag-api
```

`rag_db/` is in `.dockerignore`, so every new container starts with an empty collection. Run `POST /ingest` first. The first ingest in a container also downloads Chroma's embedding model (all-MiniLM-L6-v2, 79MB), so it can take 30s or so.

To keep the ingested data when the container is removed, put `rag_db` on a named volume. To change the docs without rebuilding, mount a folder over `docs/` and call `/ingest` again:

```
docker run -d -p 8000:8000 -e OLLAMA_URL=http://host.docker.internal:11434 \
    -v rag-data:/app/rag_db -v "$PWD/docs":/app/docs:ro --name rag-api-container my-rag-api
```

Both were tested: the volume kept all 32 chunks across `docker rm` and a new container, and editing the mounted files then re-ingesting removed the chunks from a shortened and a deleted file. The one thing `/ingest` won't do is empty the collection, because an empty or missing `docs/` is treated as a mistake. To start from nothing, remove the volume (`docker volume rm rag-data`) or the container.

When `/ask` fails, `docker logs rag-api-container` has a `WARNING` line with the reason, e.g. Ollama's own error text, which the 502/503 response leaves out.

Swagger UI is at http://localhost:8000/docs and ReDoc at http://localhost:8000/redoc. Both are generated from the same models as below, including field limits, error responses and an example for `/ask`.

## API reference

All request and response bodies are JSON. Errors raised by the app are `{"detail": "<message>"}`. Validation errors (422) are FastAPI's standard format, a list under `detail`:

```json
{"detail": [{"type": "greater_than_equal", "loc": ["body", "n_results"],
             "msg": "Input should be greater than or equal to 1", "input": 0, "ctx": {"ge": 1}}]}
```

### POST /ingest

Splits every `.txt` and `.md` file in `docs/` (extension case doesn't matter) into chunks on blank lines and upserts them into ChromaDB. No request body; anything sent is ignored.

- Chunk ids are `<filename>_<index>`, so calling it again overwrites instead of duplicating
- Chunks left over from a file that was removed or got shorter are deleted (`chunks_removed`)
- A markdown heading is joined to the paragraph after it. Headings with no text after them, empty files and other file types are skipped
- If `docs/` is missing or has nothing usable, the collection is left as it is and all counts are 0

| Status | When |
|---|---|
| 200 | Always, including when nothing was found to ingest |

```
curl -X POST http://localhost:8000/ingest
```

```json
{"chunks_ingested": 32, "files_ingested": 7, "chunks_removed": 0, "message": "Ingested 32 chunks from 7 files"}
```

When old chunks are cleaned up the message ends in `, removed N old chunks` (singular when it's 1, like `Ingested 1 chunk from 1 file`). With an empty `docs/` it's `No .txt or .md content found in /app/docs, nothing changed`.

### POST /ask

Embeds the question, fetches the `n_results` closest chunks, drops any with a distance over `max_distance`, and sends what's left to llama3.2:1b as context.

| Field | Type | Required | Default | Limits |
|---|---|---|---|---|
| `question` | string | yes | | 1 to 1000 characters, can't be only whitespace. Surrounding whitespace is stripped |
| `n_results` | integer | no | 3 | 1 to 20 |
| `max_distance` | number | no | 1.2 | 0 to 4 |

Distances are squared L2 between normalised embeddings, so 0 is identical text, around 1.2 is loosely related and 4 is the maximum. Unknown fields are ignored.

Types are strict: `n_results` has to be a JSON integer (`true`, `"5"` and `5.0` are rejected) and `max_distance` a JSON number (`"0.9"`, `true`, `NaN` and `Infinity` are rejected). Without that, pydantic quietly turned `true` into 1.

| Status | When | Body |
|---|---|---|
| 200 | Answered, or nothing to answer from (see below) | `answer`, `sources`, `confidence` |
| 400 | The body can't be read at all: not valid UTF-8, or JSON nested tens of thousands of levels deep | `There was an error parsing the body` |
| 422 | Missing/blank/too long `question`, a value out of range or the wrong type, or the body isn't a JSON object | FastAPI validation error |
| 502 | Ollama replied but there was no usable answer: not JSON, no `message.content`, empty text, or the reply was cut off part way | `Ollama's response had no answer in it` / `Ollama returned an empty answer` / `Ollama's response was cut off` |
| 503 | Ollama is down, timed out, or returned a non-200 (e.g. 404 when the model isn't pulled) | `Ollama service is unavailable` / `Ollama timed out` / `Ollama returned 404 for model llama3.2:1b` |

Response fields:

- `answer`: the model's answer with surrounding whitespace removed, never empty
- `sources`: the chunks used as context, closest first. Each has `text` (never blank), `source` (the file name, or `"unknown"` for a chunk that has none) and `distance` (rounded to 4 places, never more than `max_distance`)
- `confidence`: always one of `high`, `medium`, `low`

```
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How does ChromaDB store embeddings?", "n_results": 2}'
```

```json
{
  "answer": "ChromaDB stores embeddings automatically if you pass raw text. According to the source, ChromaDB uses the all-MiniLM-L6-v2 sentence-transformer model, which produces 384-dimensional vectors. This information is mentioned in the embeddings.md document.",
  "sources": [
    {"text": "# ChromaDB\n\nChromaDB is an open-source vector database. It stores documents, their embeddings, metadata and ids in collections, and handles the embedding step automatically if you pass raw text.",
     "source": "chromadb.md", "distance": 0.4948},
    {"text": "ChromaDB uses the all-MiniLM-L6-v2 sentence-transformer model by default. It produces 384-dimensional vectors and runs locally on CPU, so no API key is needed.",
     "source": "embeddings.md", "distance": 0.8273}
  ],
  "confidence": "high"
}
```

When there's nothing to answer from, the model isn't called (so this works with Ollama down) and the response is 200 with `sources: []` and `confidence: "low"`. The answer says which case it was:

| Case | `answer` |
|---|---|
| Nothing ingested yet | `No documents have been ingested yet. Call POST /ingest first.` |
| No chunk within `max_distance` | `No relevant documents found for this question. Try rephrasing it or raising max_distance.` |

Off-topic and vague questions ("Give me a pancake recipe", "How does it work?", "Tell me more") all land above 1.2, so they get the second message instead of a made-up answer.

### GET /stats

| Status | When | Body |
|---|---|---|
| 200 | | `{"document_count": 32, "model": "llama3.2:1b", "db_path": "/app/rag_db"}` |
| 503 | ChromaDB can't be read | `{"detail": "ChromaDB is not readable"}` |

### GET /health

Checks ChromaDB (by counting the collection) and Ollama (`GET /api/tags`, 3s timeout).

| `status` | HTTP | Meaning |
|---|---|---|
| `ok` | 200 | Both work |
| `degraded` | 200 | ChromaDB works, Ollama doesn't. `/ingest` and retrieval work, `/ask` returns 503 whenever it has context to send |
| `error` | 503 | ChromaDB can't be read. `document_count` is `null` |

```json
{"status": "ok", "chromadb": "ok", "ollama": "connected", "document_count": 32, "model": "llama3.2:1b"}
```

### Other routes

Wrong method on a route gives 405, unknown paths give 404. CORS allows any origin.

## Confidence

It starts from the distance of the closest source:

| Best distance | Level |
|---|---|
| under 0.5 | high |
| 0.5 to under 0.8 | medium |
| 0.8 and up | low |

Then it looks at the answer:

1. If the model says it doesn't have enough information (the wording the system prompt asks for), confidence is `low` no matter how close the sources were
2. If the answer has a number with 2+ digits or an acronym (`MIT`, `NER`, `1,024`) that isn't in the question, the sources or the source file names, it drops one level. Numbers are compared without commas and acronyms as whole words, so `LLMs` matches `LLM` but `MIT` doesn't match `limit`

No sources means `low`.

Why the extra checks: these were all real answers from llama3.2:1b with sources retrieved, before the change:

| Question | Answer said | Before | After |
|---|---|---|---|
| What license is ChromaDB released under? | MIT License (not in the docs) | medium | low |
| How many parameters does llama3.2:1b have? | 1,024 parameters, citing ollama.txt | low | low |
| How do I defend against prompt injection? | made-up NER/POS filtering | medium | low |
| How does fixed-size chunking work? | made-up 100 to 200 character sizes | medium | low |
| What version of ChromaDB is current? | I don't have enough information... | medium | low |
| ChromaDB (one word) | | high, from a chunk that was just `# ChromaDB` | medium |

The last row is why headings are merged now. A heading on its own was 10 to 38 characters, embedded very close to short questions, and filled a source slot with nothing in it.

Known limits:

- Correct answers can still come out `low`. "What are the three stages of a RAG pipeline?" scores 0.961 because the chunk that answers it never says "RAG", only its heading does. "What is semantic search?" scores 0.809, just over the line
- A made-up answer with no numbers or acronyms isn't caught. Asked "What is prompt injection?", the model added a paragraph about "prompt poisoning" and still got `high`
- Some paraphrases miss entirely: "how do I save vectors to disk" scores 1.49 against the PersistentClient chunk, so it's filtered out
- The 0.5 / 0.8 thresholds were only checked against these 7 docs, a bigger corpus would need them rechecked

To look for false positives, the specifics check was run over two sweeps of real answers, about 50 in total, mostly to questions the docs do cover. It flagged 10 answers and every one had something made up in it: the 4 in the table above plus MiniLM being "128-dimensional", a 2019 Google study, a 2018 journal article, BERT and NLP (not in the docs), and HTTP methods with a wrong expansion of REST. No correct answer got flagged, but it's a regex, so an answer that writes a normal word in capitals for emphasis would be marked down.

### Why max_distance defaults to 1.2

The best-match distance of questions in three groups:

| Group | Examples | Best distance |
|---|---|---|
| Answer is in the docs, but missed | "How big are the vectors from all-MiniLM-L6-v2?", "Does the embedding model run on CPU?", "how do I save vectors to disk" | 1.25 to 1.63 |
| Answer isn't in the docs | "What is pgvector?", "Who created Ollama?", "How do I fine-tune llama3?" | 0.99 to 1.56 |
| Off-topic or vague | "Give me a pancake recipe", "How does it work?", "asdf qwer zxcv" | 1.55 to 1.75 |

The first two overlap, so no cutoff separates them. Raising it to 1.3 would catch 2 more real questions but also let 3 uncovered ones through, and the model would be answering those from loosely related chunks. 1.2 keeps all off-topic questions out and errs on the side of "No relevant documents found". Pass a higher `max_distance` per request to trade the other way. The embedding model is also weak on numbers: "Which port is 11434?" scores 1.63 even though `ollama.txt` has that exact port.

## Tests

`tests/` has 176 tests that run in-process with FastAPI's TestClient, a throwaway ChromaDB and a fake Ollama, so they need neither Docker nor Ollama and take about 5 seconds. `tests/test_live.py` has 12 more that hit a running container with the real model, and they're skipped unless `RAG_API_URL` is set.

Run them from this folder, `pytest.ini` is what tells pytest where `my_rag_api` is. A plain `pytest` from the repo root fails with `ModuleNotFoundError`; `pytest module_08/Dockerfile-RAG-API` from the root works.

```
pip install -r requirements.txt -r requirements-dev.txt
pytest                                                     # 176 passed, 12 skipped
RAG_API_URL=http://localhost:8000 pytest tests/test_live.py   # needs the container running with OLLAMA_URL set
```

Because Ollama is faked, the in-process tests check how answers are handled (status codes, fields, confidence rules), not whether the model's answers are any good. The live tests only check the shape of real answers and which doc was retrieved. Answer quality was checked by hand, see [Confidence](#confidence).

| File | Covers |
|---|---|
| `test_ask.py` | each doc retrieved by a matching question, response fields and formatting, the prompt sent to Ollama, `n_results`/`max_distance` behaviour, boundary values, empty collection, off-topic and vague questions, 21 invalid fields and 5 malformed bodies (422), NaN/Infinity/lone surrogates (422, used to be 500), unreadable bodies (400), Ollama down/timeout/404/500 (503), bad, empty or cut-off Ollama replies (502), failure causes logged, chunks with no source or blank text |
| `test_confidence.py` | every threshold and its boundary, refusal phrasing, unsupported numbers and acronyms (and things that must not be flagged), end-to-end through `/ask`, heading-only chunk regression |
| `test_ingest.py` | real docs (32 chunks / 7 files), repeat ingests, cleanup after a file is shortened or deleted, empty and missing folders, file types, CRLF, invalid UTF-8, heading merging, singular/plural in the message |
| `test_health_stats.py` | `/stats` and `/health` normal, Ollama down 5 different ways (degraded), ChromaDB unreadable (503, and logged) |
| `test_docs_and_routing.py` | every route and schema field documented in OpenAPI, limits and enums in the schema, examples valid, documented error examples match the real responses, 405/404, CORS |
| `test_live.py` | the same endpoints against the real container and model, plus 30 concurrent `/health` and 3 concurrent `/ask` |

Line and branch coverage of `my_rag_api.py` is 100%. To check the tests actually catch things, I broke the code 23 different ways (turned off heading merging, stale cleanup, the 502 checks, the source fallback, the confidence downgrade, the safe 422 handler, strict types, the logging, changed a threshold from `<` to `<=`, and so on) and every one made at least one test fail.

Things that were checked by hand rather than in `tests/`, since they need Docker or a real server:

- a fake Ollama over real HTTP returning HTML, empty answers, 404, 500, a dropped connection and a truncated body, with the container giving the documented 502/503 each time
- 130 requests from 40 threads at once (10 `/ingest` alongside 60 `/ask`, plus `/health` and `/stats`): all 200, no answer saw a half-ingested collection
- a 50MB request body (accepted, see below), null bytes, emoji-only and punctuation-only questions
- the whole suite inside the image (Python 3.11) and in a fresh `python:3.11-slim` with only the committed files, as well as locally on Python 3.13


## Layer caching

`requirements.txt` is copied and installed before the app code, so changing `my_rag_api.py` or `docs/` doesn't redo the pip install.

To check: build once, change a comment in `my_rag_api.py`, then build again.

```
docker build -t my-rag-api .
```

Results:

| Build | Time | apt-get | pip install |
|---|---|---|---|
| first build (`--no-cache`) | 44s | 10.3s | 25.3s |
| after editing a comment in `my_rag_api.py` | 0.27s | CACHED | CACHED |
| after editing a file in `docs/` | 0.17s | CACHED | CACHED, only `COPY docs/` reruns |
| after editing `requirements.txt` | 31s | CACHED | reruns (24.4s) |

Rebuild output after the comment change:

```
#7 [3/7] RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
#7 CACHED
#8 [4/7] COPY requirements.txt .
#8 CACHED
#9 [5/7] RUN pip install --no-cache-dir -r requirements.txt
#9 CACHED
#10 [6/7] COPY my_rag_api.py .
#10 DONE 0.0s
#11 [7/7] COPY docs/ ./docs
#11 DONE 0.0s
```

With a single `COPY . .` before `pip install`, any edit at all would reinstall chromadb.

## Checking the container

```
docker ps                                 # Up, 0.0.0.0:8000->8000/tcp
docker logs rag-api-container             # Uvicorn running on http://0.0.0.0:8000 + request log
docker top rag-api-container              # uvicorn my_rag_api:app --host 0.0.0.0 --port 8000
docker run --rm my-rag-api ls /app        # docs, my_rag_api.py, requirements.txt only
docker stop rag-api-container             # exits 0, uvicorn shuts down cleanly on SIGTERM
docker rm rag-api-container
```

Ingested data survives `docker restart` and `docker stop` / `docker start`, and is gone after `docker rm`, since it lives in the container's writable layer.

## Things that can go wrong

- Port 8000 already in use on the host: `docker run` fails with "Bind for 0.0.0.0:8000 failed: port is already allocated". Stop whatever is using it, or map another port (`-p 8001:8000`).
- `--name rag-api-container` already exists (even stopped): `docker rm -f rag-api-container` first.
- No `--host 0.0.0.0`: uvicorn listens on the container's 127.0.0.1, `-p` forwards the connection but nothing answers (curl gets a reset).
- Missing `requirements.txt` or `docs/` in the build context: the `COPY` step fails.
- Ollama not reachable: `/health` says degraded and `/ask` returns 503. Retrieval and `/ingest` still work, and it recovers by itself once Ollama is back.
- `pytest` from the repo root: `ModuleNotFoundError`. Run it from this folder.

## Before using this for real

This is set up for running locally. Things that would need changing first:

- No authentication, anyone who can reach port 8000 can call `/ingest` and `/ask`
- CORS allows every origin with credentials
- The container runs as root (the `python:3.11-slim` default)
- No limit on request size: a 50MB body was accepted and parsed. A reverse proxy in front would normally cap this
- No rate limiting. Each `/ask` holds one of FastAPI's 40 worker threads for as long as Ollama takes (up to the 120s timeout), so 40 slow requests at once would make the rest wait
- `/stats` shows the db path inside the container
