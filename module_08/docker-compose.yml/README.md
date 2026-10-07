# Multi-Service RAG Stack (Docker Compose)

The RAG FastAPI backend from `Dockerfile-RAG-API` and Ollama running as two containers on one Compose network. This folder is the whole build context: `Dockerfile`, `.dockerignore`, `my_rag_api.py`, `requirements.txt` and `docs/`. `tests/` isn't copied into the image.

`docker-compose.starter.yml` is the original scaffold, `docker-compose.yml` is the finished version.

Changes to `my_rag_api.py` compared with `Dockerfile-RAG-API/`:

- the ChromaDB path is read from `DB_PATH` instead of `RAG_DB_PATH`, so the value set in compose is actually used
- `GET /` returns `{"message": "RAG API is running", "docs": "/docs"}` (the exercise checks that the root endpoint responds)
- ChromaDB failing during `/ask` or `/ingest` returns a JSON 503 instead of a plain-text 500
- a `source` stored as a number or bool no longer crashes `/ask` with a 500, and blank or padded source names are cleaned up
- the same chunk text is only returned (and sent to the model) once

The `Dockerfile` also downloads Chroma's embedding model at build time (see [Volumes](#volumes)).

## Running it

From this folder:

```
docker-compose up --build -d                         # build the backend, start both services
docker-compose exec ollama ollama pull llama3.2:1b   # once, the model goes into the ollama_models volume
curl localhost:8000/health
curl -X POST localhost:8000/ingest
docker-compose down                                  # removes containers + network, keeps volumes
```

Leave out `-d` to keep the logs in the terminal, or run `docker-compose logs -f backend` to follow them. Swagger UI is at http://localhost:8000/docs.

If Ollama is already running on the Mac, publish the container's Ollama on another host port instead (see [Networking](#networking)):

```
OLLAMA_HOST_PORT=11435 docker-compose up -d
```

The model has to be pulled before `/ask` will work. Without it `/ask` returns `503 {"detail": "Ollama returned 404 for model llama3.2:1b"}`, and `docker-compose logs backend` shows Ollama's own `model 'llama3.2:1b' not found`.

Compose v2 and later prints `the attribute version is obsolete` for `version: "3.8"`. It's only a warning. The top-level `name: rag-stack` is there because the project name would otherwise come from the folder name, `docker-compose.yml`, which becomes `docker-composeyml`.

## Networking

Compose creates a `rag-stack_default` network and puts both services on it. Docker's DNS resolves each service name to its container, so the backend reaches Ollama with:

```
OLLAMA_URL=http://ollama:11434
```

Inside the backend container `localhost` is the backend itself. Checked from inside the running container (`docker-compose exec backend ...`):

```
getent hosts ollama                          -> 172.18.0.2  ollama
requests.get("http://ollama:11434/api/tags")    -> 200 ['llama3.2:1b']
requests.get("http://localhost:11434/api/tags") -> ConnectionError
```

| Service | Ports | Notes |
|---|---|---|
| `backend` | `8000:8000` | built from `./Dockerfile`, starts once `ollama` is healthy |
| `ollama` | `11434:11434` (`OLLAMA_HOST_PORT` changes the host side) | `ollama/ollama` image. The backend doesn't need this port published. It's there so the Mac can reach the container's Ollama too |

### Start order

A plain `depends_on: [ollama]` only waits for the Ollama container to start, not for the server inside it to listen, so `/health` could report `"ollama": "disconnected"` for the first few seconds. The `ollama` service has a healthcheck (`ollama list`, which fails until the server answers) and the backend uses `condition: service_healthy`, so Compose holds the backend back until Ollama is up:

```
Container rag-stack-ollama-1 Started
Container rag-stack-ollama-1 Waiting
Container rag-stack-ollama-1 Healthy
Container rag-stack-backend-1 Started
```

The very first `/health` response after `up -d` already says `connected`. While starting, the check runs every second (`start_interval`). After that it runs every 30s, so it isn't running `ollama list` constantly.

### Port 11434 with Ollama on the Mac

Running Ollama on the Mac (`ollama serve` or the Ollama app) doesn't stop the container from starting. Ollama on the Mac listens on `127.0.0.1:11434` and Docker publishes on `*:11434`, so both run and the Mac ends up with two Ollamas on one port:

```
127.0.0.1:11434 -> 0.35.1   Mac's Ollama, also what the ollama CLI uses
localhost:11434 -> 0.40.0   container (curl tried ::1 first)
```

Nothing errors, but an `ollama pull` on the Mac goes to the Mac's own Ollama, not the container's. The backend isn't affected since it uses `http://ollama:11434` on the Compose network. Either stop Ollama on the Mac first, or start the stack with `OLLAMA_HOST_PORT=11435` so the container is on its own port. Without the variable the mapping is `11434:11434` as the exercise asks.

Port 8000 needs to be free too, e.g. no old `rag-api-container` or local uvicorn.

## Volumes

| Volume | Mounted at | Holds | Size after the test |
|---|---|---|---|
| `chroma_data` | `/app/rag_db` (`DB_PATH`) | ChromaDB collection | 516KB, 32 chunks |
| `ollama_models` | `/root/.ollama` | pulled models | 1.32GB |

Both are named volumes declared under the top-level `volumes:`, so Docker stores them outside any container. On disk they're `rag-stack_chroma_data` and `rag-stack_ollama_models` (`docker volume ls`).

- `docker-compose down` removes the containers and the network, and leaves the volumes alone
- `docker-compose down -v` deletes the volumes too, so the chunks and the 1.3GB model are gone

### Chroma's embedding model

Chroma downloads its embedding model (all-MiniLM-L6-v2) the first time it embeds anything. Left to do that at runtime, it went into each container's own filesystem. Every new backend container downloaded it again on its first `/ingest` or `/ask` (175MB per container, and it needed internet access). The `Dockerfile` now runs one embedding at build time, so the model is part of the image. The 79MB archive is deleted in the same step, because Chroma only checks for the extracted files.

| | Before | After |
|---|---|---|
| backend container's own layer after `/ingest` + `/ask` | 175MB | 274KB |
| image | 1.28GB | 1.46GB (the model layer is 92MB, Docker counts the compressed copy too) |
| embedding with `--network none` | had to download | works |

The model is in a layer after `pip install` and before `COPY my_rag_api.py`, so editing the app doesn't rerun the download.

## Persistence test

```
curl localhost:8000/stats                      # note document_count
docker-compose down
docker volume ls --filter name=rag-stack       # both volumes still listed
docker-compose up -d
curl localhost:8000/stats                      # same document_count
docker-compose exec ollama ollama list         # llama3.2:1b still there
```

Results:

| Check | Before `down` | After `down` + `up -d` |
|---|---|---|
| `/stats` `document_count` | 32 | 32 |
| `/health` | `ok`, `connected` | `ok`, `connected` |
| `ollama list` | `llama3.2:1b 1.3 GB` | `llama3.2:1b 1.3 GB` |
| `POST /ask` | 200 | 200, no re-pull or re-ingest |

`down` removed both containers and `rag-stack_default`, and neither volume was touched. Rebuilding only the backend (`docker-compose up --build -d` after editing `my_rag_api.py`) also kept all 32 chunks, and the ollama container kept running. The same checks passed again after adding the healthcheck and the embedding model step, with the stack on `OLLAMA_HOST_PORT=11435` next to Ollama running on the Mac.

## API reference

Swagger UI (http://localhost:8000/docs) and ReDoc (`/redoc`) are generated from the same models as below: every field has a description, the limits are in the schema, and every error code has an example. The examples here are real responses from the stack.

All request and response bodies are JSON. Errors raised by the app are `{"detail": "<message>"}`. Validation errors (422) are FastAPI's standard format, a list under `detail`:

```json
{"detail": [{"type": "greater_than_equal", "loc": ["body", "n_results"],
             "msg": "Input should be greater than or equal to 1", "input": 0, "ctx": {"ge": 1}}]}
```

| Method | Path | Success | Errors |
|---|---|---|---|
| GET | `/` | 200 | |
| POST | `/ingest` | 200 | 503 |
| POST | `/ask` | 200 | 400, 422, 502, 503 |
| GET | `/stats` | 200 | 503 |
| GET | `/health` | 200 (`ok` or `degraded`) | 503 (`error`) |

### GET /

Shows the API process is up. It doesn't touch ChromaDB or Ollama, so it still answers with both of them down. `/health` is the one that checks them. Query strings are ignored.

```
curl http://localhost:8000/
```

```json
{"message": "RAG API is running", "docs": "/docs"}
```

### POST /ingest

Splits every `.txt` and `.md` file in `docs/` (extension case doesn't matter) into chunks on blank lines and upserts them into ChromaDB. No request body, anything sent is ignored.

- Chunk ids are `<filename>_<index>`, so calling it again overwrites instead of duplicating
- Chunks left over from a file that was removed or got shorter are deleted (`chunks_removed`)
- A markdown heading is joined to the paragraph after it. Headings with no text after them, empty files and other file types are skipped
- If `docs/` is missing or has nothing usable, the collection is left as it is and all counts are 0

| Status | When | Body |
|---|---|---|
| 200 | Ingested, or nothing found to ingest | counts and `message` |
| 503 | ChromaDB couldn't be written to (db locked, disk full, embedding model failing) | `Could not write to ChromaDB` |

Every step can be repeated, so after a 503, calling `/ingest` again once ChromaDB is back finishes the job, old-chunk cleanup included. The cause of the failure is in `docker-compose logs backend`.

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

Types are strict: `n_results` has to be a JSON integer (`true`, `"5"` and `5.0` are rejected) and `max_distance` a JSON number (`"0.9"`, `true`, `NaN` and `Infinity` are rejected).

| Status | When | `detail` |
|---|---|---|
| 200 | Answered, or nothing to answer from (see below) | |
| 400 | The body can't be read at all: not valid UTF-8, or JSON nested tens of thousands of levels deep | `There was an error parsing the body` |
| 422 | Missing, blank or too long `question`, a value out of range or the wrong type, or the body isn't a JSON object | FastAPI validation error |
| 502 | Ollama replied but there was no usable answer: not JSON, no `message.content`, empty text, or cut off part way | `Ollama's response had no answer in it` / `Ollama returned an empty answer` / `Ollama's response was cut off` |
| 503 | Ollama is down, timed out, or returned a non-200 (e.g. 404 when the model isn't pulled) | `Ollama service is unavailable` / `Ollama timed out` / `Ollama returned 404 for model llama3.2:1b` |
| 503 | ChromaDB can't be read, or the search fails (e.g. the embedding model can't load) | `ChromaDB is not readable` / `ChromaDB search failed` |

For every 502 and 503 the actual cause (Ollama's own error text, the ChromaDB exception) is logged as a `WARNING` in `docker-compose logs backend`, and the response only has the short message.

Response fields, always all three:

- `answer`: the model's answer with surrounding whitespace removed, never empty
- `sources`: the chunks used as context, closest first. Can be fewer than `n_results`
  - `text`: trimmed, never blank. Each text appears once: if the same paragraph is stored twice (two files sharing it, or the same text under another id) only the closest copy is kept, so it doesn't take two slots or go to the model twice
  - `source`: the file name, trimmed. `"unknown"` when the chunk has no source or a blank one. A number stored as the source comes back as text (`"123"`), `true`/`false` count as no source
  - `distance`: rounded to 4 places, never more than `max_distance` (a chunk exactly at `max_distance` is kept)
- `confidence`: always one of `high`, `medium`, `low`, see [Confidence](#confidence)

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
| No usable chunk within `max_distance` | `No relevant documents found for this question. Try rephrasing it or raising max_distance.` |

Off-topic and vague questions ("Give me a pancake recipe", "Is it safe?", "How do I run it locally?") all land above 1.2, so they get the second message instead of a made-up answer.

### GET /stats

| Status | When | Body |
|---|---|---|
| 200 | | `{"document_count": 32, "model": "llama3.2:1b", "db_path": "/app/rag_db"}` |
| 503 | ChromaDB can't be read | `{"detail": "ChromaDB is not readable"}` |

`db_path` is `DB_PATH` from `docker-compose.yml`, the mount point of the `chroma_data` volume.

### GET /health

Checks ChromaDB (by counting the collection) and Ollama (`GET http://ollama:11434/api/tags`, 3s timeout).

| `status` | HTTP | Meaning |
|---|---|---|
| `ok` | 200 | Both work |
| `degraded` | 200 | ChromaDB works, Ollama doesn't. `/ingest` and retrieval work, `/ask` returns 503 whenever it has context to send |
| `error` | 503 | ChromaDB can't be read. `document_count` is `null` |

```json
{"status": "ok", "chromadb": "ok", "ollama": "connected", "document_count": 32, "model": "llama3.2:1b"}
```

### Other routes

A wrong method on a route gives 405 (`{"detail": "Method Not Allowed"}`), unknown paths give 404. CORS allows any origin.

## Confidence

It starts from the distance of the closest source:

| Best distance | Level |
|---|---|
| under 0.5 | high |
| 0.5 to under 0.8 | medium |
| 0.8 and up | low |

Then it looks at the answer:

1. If the model says it doesn't have enough information (the wording the system prompt asks for), confidence is `low` no matter how close the sources were
2. If the answer has a number with 2+ digits or an acronym (`MIT`, `NER`, `1,024`) that isn't in the question, any of the sources or the source file names, it drops one level. Numbers are compared without commas and acronyms as whole words, so `LLMs` matches `LLM` but `MIT` doesn't match `limit`

No sources means `low`. The rest of the sources don't change the level: weaker extra chunks don't pull a close match down, several sources tied at one distance score the same as one, and a chunk labelled `"unknown"` scores the same as a named one (the label only matters for citing).

### Checking confidence against real answers

To see how the levels hold up on ambiguous and borderline questions, 51 questions went through the stack with the defaults: ambiguous or vague (9), borderline (8), not answered by the docs (8), two-part like "Compare X and Y" (8), covered by the docs (14) and 4 controls. 12 had no source within 1.2 and got "No relevant documents found". The other 39 answers were graded by hand against `docs/`:

| Confidence | Answers | Correct | Partly made up | Wrong or not from the docs | Refused |
|---|---|---|---|---|---|
| high | 4 | 4 | | | |
| medium | 10 | 8 | 2 | | |
| low | 25 | 6 | 7 | 6 | 6 |

"Partly made up" means a right core with an invented detail, e.g. `ollama list` showing URLs, or NER/POS filtering as a prompt-injection defence.

- `high` was right every time, and no answer that was wrong or not from the docs got above `low`
- All 8 questions the docs don't answer ("Who created Ollama?", "What license is ChromaDB released under?") came out `low`. The model refused 6 of them, and the other 2 (a made-up MIT License, a made-up delete API) had best matches past 0.8
- Borderline matches: 22 answers had a best distance between 0.8 and 1.2, and only 6 of those were right ("three stages of a RAG pipeline" at 0.961, "What port does Ollama use?" at 0.947, ...). The others were refusals, partly made up or wrong. A correct answer from that band still comes out `low`, since nothing in the answer reliably tells the 6 apart from the 16
- Two-part questions are where it's weakest. The closest chunk usually covers one half, and confidence is based on that half. "What's the difference between chunking and embedding?" matched the chunking paragraph at 0.45, and the model filled in the embedding half from its training data (Word2Vec, GloVe, BERT). It only dropped from `high` to `medium` because BERT and NLP are acronyms. With the 2 ambiguous ones that also had two halves, that's 10 two-part questions: 1 came out `high` (correct), 4 `medium` (3 correct) and 5 `low`

Two more checks were tried on the same 39 answers and left out because the data didn't support them:

- Word overlap between the answer and its sources: correct paraphrases scored as low as half-invented answers (0.47 for a right "What is prompt injection?", 0.49 for the half-made-up chunking/embedding one), so no cutoff separates them
- Flagging tool and product names that aren't in the sources (Word2Vec, GloVe, RoBERTa): it only fired on the answer the acronym check already caught, 0 new catches

These cases are in `tests/test_confidence.py` with the real answers and distances, so a change to the rules shows up as a failing test.

Known limits:

- Correct answers from borderline matches stay `low` (above). The 0.5 / 0.8 thresholds were only checked against these 7 docs, a bigger corpus would need them rechecked
- A two-part question is only marked down for its made-up half when that half has a number or an acronym in it
- Example numbers count as specifics. "What is recall?" used 10 and 100 in a worked example and dropped from `medium` to `low` (that answer also mixed recall up with precision@k, so `low` wasn't wrong there)
- A made-up answer with no numbers or acronyms isn't caught at all

### Why max_distance defaults to 1.2

Best-match distances measured against these docs:

| Group | Examples | Best distance |
|---|---|---|
| Answer is in the docs, but missed | "how do I save vectors to disk", "What is topic blurring?", "Does the embedding model need a GPU?" (6 questions) | 1.25 to 1.54 |
| Answer isn't in the docs | "Who created Ollama?", "What is pgvector?", "How do I fine-tune llama3?" (6) | 0.99 to 1.25 |
| Off-topic or vague | "Give me a pancake recipe", "Is it safe?", "How do I run it locally?" (8) | 1.34 to 1.75 |

The first two groups overlap, so no cutoff separates them. 1.2 keeps every off-topic and vague question away from the model. 5 of the 6 unanswerable questions still get through, which is what the refusal check and the `low` band are for. Raising the cutoff to 1.3 would let 2 of the missed questions through, plus 1 more unanswerable one. Pass a higher `max_distance` per request to trade the other way.

## Tests

```
pip install -r requirements.txt -r requirements-dev.txt
pytest                                                        # 231 passed, 15 skipped
RAG_API_URL=http://localhost:8000 pytest tests/test_live.py   # with the stack up and the model pulled
```

Run them from this folder, `pytest.ini` is what tells pytest where `my_rag_api` is. The in-process tests use FastAPI's TestClient, a throwaway ChromaDB (via `DB_PATH`, the same variable compose sets) and a fake Ollama, so they need neither Docker running nor Ollama, and take about 6 seconds.

| File | Tests | Covers |
|---|---|---|
| `test_ask.py` | 99 | each doc retrieved by a matching question, response fields and formatting, the prompt sent to Ollama, `n_results`/`max_distance` including a chunk exactly at the limit, 21 invalid fields and 5 malformed bodies (422), NaN/Infinity/lone surrogates (422), unreadable bodies (400), Ollama down/timeout/404/500 (503), bad, empty or cut-off Ollama replies (502), ChromaDB unreadable or search failing (503), sources with no name, blank, padded, numeric or bool names, blank or `None` text, duplicate text |
| `test_confidence.py` | 49 | every threshold and its boundary, refusal phrasing, unsupported numbers and acronyms (and things that must not be flagged), end-to-end through `/ask`, and the real answers above: a two-part question, a made-up detail, a borderline correct answer, a refusal, tied and `unknown` sources |
| `test_ingest.py` | 18 | real docs (32 chunks from 7 files), repeat ingests, cleanup after a file is shortened or deleted, empty and missing folders, file types, CRLF, invalid UTF-8, heading merging, singular/plural, ChromaDB write failure (503) and a retry after a half-finished ingest |
| `test_health_stats.py` | 14 | `/stats` and `/health` normal, Ollama down 5 different ways (degraded), ChromaDB unreadable (503, logged), `DB_PATH` from the environment |
| `test_docs_and_routing.py` | 32 | `GET /` (also with ChromaDB and Ollama both down), every route, field and error code in the OpenAPI schema, documented error examples match the real responses, 405/404, CORS |
| `test_compose.py` | 19 | `docker-compose.yml` as Compose parses it: services, ports and the `OLLAMA_HOST_PORT` override, `OLLAMA_URL` using the service name, `DB_PATH` matching the volume mount, `service_healthy`, the healthcheck, volumes, the app reading every env var compose sets, the Dockerfile port and layer order. Needs `docker-compose` installed (not running), skipped otherwise |
| `test_live.py` | 15 | the endpoints against the running stack and the real model, two questions the docs don't answer coming back `low`, 30 concurrent `/health` and 3 concurrent `/ask` |

Because Ollama is faked, the in-process tests check how answers are handled, not whether they're any good. Answer quality was graded by hand, see [Confidence](#confidence).

Checks on the tests themselves:

- line and branch coverage of `my_rag_api.py` is 100% (`coverage run --branch -m pytest`)
- the new edge-case tests were run against the previous `my_rag_api.py`, and 14 failed, one or more for each bug fixed here
- `test_compose.py` was run against three broken copies of the setup (`OLLAMA_URL` pointing at localhost, the app reading `RAG_DB_PATH`, a plain `depends_on` list), and each one failed the matching test
- inside the backend image (Python 3.11): 212 passed, with the 34 compose and live tests skipped. Locally it's Python 3.13
- the live tests pass against the rebuilt stack
