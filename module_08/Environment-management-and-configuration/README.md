# Configurable RAG Application

The Compose RAG stack from `../docker-compose.yml/`, with every setting moved out of the code and into environment variables. `config.py` reads them into a single `Settings` object, `.env` holds local values, and `docker-compose.yml` passes the same file into the backend container with `env_file`.

| File | |
|---|---|
| `config.py` | `Settings` class and the `settings` instance everything imports |
| `starter.py` | the original scaffold for `config.py`, unchanged |
| `.env.example` | committed, every setting with its default and a comment |
| `.env` | local values, ignored by git and docker |
| `my_rag_api.py` | the FastAPI app, now reads `settings.*` instead of its own constants |
| `docker-compose.yml`, `Dockerfile` | `env_file: .env` on both services, `config.py` copied into the image |

## Setup

```
cp .env.example .env
python config.py                      # prints the settings it loaded
OLLAMA_HOST_PORT=11435 docker-compose up --build -d   # leave the prefix off if Ollama isn't running on the Mac
```

## How config is loaded

```
shell / compose env var   >   .env   >   default in config.py
```

When `config` is first imported, `load_dotenv()` reads the `.env` next to `config.py` (not the current directory, so running from the repo root works). It never overwrites a variable that's already set, so the shell always wins. Then `Settings()` reads each variable with `os.environ.get` and falls back to the default. `settings = Settings()` runs once at import, and Python caches modules, so every `from config import settings` gets the same object. A bad value stops the app at startup instead of on the first request.

| Variable | Type | Default | Used for |
|---|---|---|---|
| `OLLAMA_URL` | str | `http://localhost:11434` | Ollama endpoint for `/ask` and `/health` |
| `MODEL_NAME` | str | `llama3.2:1b` | model sent to Ollama, shown in `/stats` and `/health` |
| `CHROMA_PATH` | str | `./rag_db` | ChromaDB folder, a relative path is taken from the app folder |
| `MAX_RESULTS` | int | `3` | default `n_results` on `/ask` |
| `CONFIDENCE_THRESHOLD` | float | `1.0` | default `max_distance` on `/ask`, chunks further away are dropped |
| `DEBUG` | bool | `false` | debug logging (loaded settings, retrieved chunks, Ollama timings) and tracebacks on 500s |

`/stats` returns all six, so you can check what the running API actually loaded.

## Type casting rules

Every env var is a string, so the non-string settings get converted, and anything that doesn't convert raises a `ValueError` that names the variable:

| | Accepted | Rejected |
|---|---|---|
| `MAX_RESULTS` | `int()` of the value, 1 to 20 (same limits as `n_results`) | `three`, `2.5`, `0`, `21` |
| `CONFIDENCE_THRESHOLD` | `float()` of the value, 0 to 4 (same as `max_distance`) | `high`, `-0.1`, `nan`, `inf` |
| `DEBUG` | `true/1/yes/on` and `false/0/no/off`, any case | anything else, e.g. `ture` |
| `OLLAMA_URL` | must start with `http://` or `https://`, trailing `/` removed | `localhost:11434` |

- A blank value (`MODEL_NAME=` in `.env`) counts as not set and gets the default. `os.environ.get` alone would return `""`.
- Surrounding whitespace is stripped, so `MAX_RESULTS= 4 ` is `4`.
- `nan` would get past a plain `float()`. Every comparison with it is False, so it would silently drop every chunk. The range check catches it.
- The starter's `.lower() == "true"` would turn `DEBUG=1` into False without warning, which is why `DEBUG` accepts the usual spellings and rejects anything else.

```
$ MAX_RESULTS=three python config.py
ValueError: MAX_RESULTS must be a whole number, got 'three'
$ CONFIDENCE_THRESHOLD=nan python config.py
ValueError: CONFIDENCE_THRESHOLD must be between 0.0 and 4.0, got 'nan'
$ DEBUG=ture python config.py
ValueError: DEBUG must be true or false, got 'ture'
```

## .env vs .env.example

- `.env` is listed in this folder's `.gitignore` and `.dockerignore`, and in the repo's root `.gitignore`. `git check-ignore -v .env` shows which rule matches.
- Keeping it out of the build context means it can't end up in an image layer, where anyone with the image could read it. Compose passes the values in at run time instead.
- `.env.example` is committed and has every setting set to the default from `config.py`, with a comment for each (a test checks they match). There are no secrets in this app yet. If one gets added, it goes in `.env.example` as an empty placeholder and `config.py` should raise when it's missing.

## Docker Compose

```yaml
backend:
  env_file: .env
  environment:
    - OLLAMA_URL=http://ollama:11434
    - CHROMA_PATH=/app/rag_db
ollama:
  env_file: .env
```

- `env_file` turns every line of `.env` into an env var in the backend container. The image has no `.env` in it, so `load_dotenv` finds nothing and `config.py` just reads what Compose set.
- `environment:` wins over `env_file`. `.env` has `localhost` and `./rag_db` for running on the Mac. Inside the container `localhost` is the container itself, so these two are fixed in the compose file. `docker-compose config` shows the merged result.
- Compose only reads `.env` when it creates a container. After editing it, run `docker-compose up -d`, which recreates both containers. `docker-compose restart` keeps the old values.
- A variable set in the shell doesn't override `env_file` inside the container. Shell variables only fill `${...}` in the YAML, like `OLLAMA_HOST_PORT`.
- Without a `.env`, `docker-compose up` stops with `env file .../.env not found`. Run `cp .env.example .env` first.
- `ollama` gets the same `env_file`, so `.env` configures the whole stack. Ollama ignores the app's settings (it reads `OLLAMA_HOST`, `OLLAMA_KEEP_ALIVE` and so on), but any of those put in `.env` would reach it. It has no `environment:` block, so it sees `.env` exactly as written. Because of that, any edit to `.env` also restarts Ollama, and the backend waits for its healthcheck again.
- The project name is still `rag-stack`, so this reuses the previous exercise's volumes, including the pulled model.

## API reference

Swagger UI (http://localhost:8000/docs) and ReDoc (`/redoc`) are generated from the same models as below. Every field has a description, the limits and defaults are in the schema, and every error code has an example (`test_docs_and_routing.py` checks all of this, including that the error examples match the real responses). The defaults shown there are whatever `MAX_RESULTS` and `CONFIDENCE_THRESHOLD` were loaded at startup. The examples here are real responses from the stack.

All bodies are JSON. Errors raised by the app are `{"detail": "<message>"}`. Validation errors (422) use FastAPI's list format. For every 502 and 503 the real cause (Ollama's error text, the ChromaDB exception) is logged as a `WARNING` in `docker-compose logs backend`, and the response only has the short message.

| Method | Path | Success | Errors |
|---|---|---|---|
| GET | `/` | 200 | |
| POST | `/ingest` | 200 | 503 |
| POST | `/ask` | 200 | 400, 422, 502, 503 |
| GET | `/stats` | 200 | 503 |
| GET | `/health` | 200 (`ok` or `degraded`) | 503 (`error`) |

On any route, a wrong method gives 405 `{"detail": "Method Not Allowed"}` and an unknown path gives 404 `{"detail": "Not Found"}`. CORS allows any origin. An unexpected exception would be a plain-text 500 `Internal Server Error`, or the traceback with `DEBUG=true`, but every known failure below has its own status.

### GET /

Shows the process is up. It doesn't touch ChromaDB or Ollama, so it answers even with both down.

```
$ curl -s localhost:8000/
{"message":"RAG API is running","docs":"/docs"}
```

### POST /ingest

No request body. Splits every `.txt` and `.md` file in `docs/` into paragraph chunks and upserts them into ChromaDB.

- ids are `<filename>_<index>`, so calling it again overwrites instead of duplicating
- chunks from a file that was removed or got shorter are deleted (`chunks_removed`)
- a heading is joined to the paragraph under it. Empty files, headings with nothing under them and other file types are skipped
- a missing or empty `docs/` leaves the collection alone and returns all zeros (`No .txt or .md content found in /app/docs, nothing changed`)

| Status | When | Body |
|---|---|---|
| 200 | Ingested, or nothing to ingest | counts and `message` |
| 503 | ChromaDB can't be written (locked, disk full, embedding model failing) | `{"detail": "Could not write to ChromaDB"}`. Every step can be repeated, so calling it again finishes the job |

```
$ curl -s -X POST localhost:8000/ingest
{"chunks_ingested":32,"files_ingested":7,"chunks_removed":0,"message":"Ingested 32 chunks from 7 files"}
```

### POST /ask

Embeds the question, fetches the `n_results` closest chunks, drops any further than `max_distance`, and sends the rest to `MODEL_NAME` as context.

| Field | Type | Required | Default | Limits |
|---|---|---|---|---|
| `question` | string | yes | | 1 to 1000 characters, not only whitespace. Surrounding whitespace is stripped |
| `n_results` | integer | no | `MAX_RESULTS` (3) | 1 to 20 |
| `max_distance` | number | no | `CONFIDENCE_THRESHOLD` (1.0) | 0 to 4 |

Distances are squared L2 between normalised embeddings: 0 is identical text, around 1.0 to 1.2 is loosely related, 4 is the maximum. Types are strict, so `n_results` has to be a JSON integer (`true`, `"5"` and `5.0` are rejected) and `max_distance` a JSON number (`"0.9"`, `true`, `NaN` and `Infinity` are rejected). Unknown fields are ignored.

| Status | When | `detail` |
|---|---|---|
| 200 | Answered, or nothing to answer from (below) | |
| 400 | Body isn't valid UTF-8, or JSON nested tens of thousands of levels deep | `There was an error parsing the body` |
| 422 | Missing, blank or too long `question`, a value out of range or of the wrong type, body not a JSON object | FastAPI validation list |
| 502 | Ollama replied with nothing usable: not JSON, no `message.content`, empty text, cut off part way | `Ollama's response had no answer in it` / `Ollama returned an empty answer` / `Ollama's response was cut off` |
| 503 | Ollama down, timed out, or returned a non-200, e.g. the model isn't pulled | `Ollama service is unavailable` / `Ollama timed out` / `Ollama returned 404 for model llama3.2:3b` |
| 503 | ChromaDB can't be read or searched | `ChromaDB is not readable` / `ChromaDB search failed` |

```
$ curl -s -X POST localhost:8000/ask -H Content-Type:application/json -d '{"question": "hi", "n_results": 0}'
{"detail":[{"type":"greater_than_equal","loc":["body","n_results"],"msg":"Input should be greater than or equal to 1","input":0,"ctx":{"ge":1}}]}
$ curl -s -X POST localhost:8000/ask -H Content-Type:application/json -d '{"question": "   "}'
{"detail":[{"type":"value_error","loc":["body","question"],"msg":"Value error, question must not be empty","input":"   ","ctx":{"error":{}}}]}
$ curl -s -X POST localhost:8000/ask -H Content-Type:application/json -d '{"question": "hi", "max_distance": NaN}'
{"detail":[{"type":"finite_number","loc":["body","max_distance"],"msg":"Input should be a finite number","input":"nan"}]}
```

The response always has all three fields:

- `answer`: the model's answer, trimmed, never empty
- `sources`: the chunks sent as context, closest first, at most `n_results`
  - `text`: trimmed, never blank. Blank chunks are skipped, and when the same text is stored twice only the closest copy is kept, so it doesn't take two slots or go to the model twice
  - `source`: the file name, trimmed. `"unknown"` when the chunk has no source or a blank one. A number comes back as text (`"123"`), `true`/`false` count as no source
  - `distance`: rounded to 4 places, never more than `max_distance` (a chunk exactly at it is kept)
- `confidence`: `high`, `medium` or `low`, see [Confidence](#confidence)

```
$ curl -s -X POST localhost:8000/ask -H Content-Type:application/json \
       -d '{"question": "How does ChromaDB store embeddings?", "n_results": 2}'
```

```json
{
  "answer": "ChromaDB stores embeddings automatically if you pass raw text. According to the source, ChromaDB uses the all-MiniLM-L6-v2 sentence-transformer model, which produces 384-dimensional vectors. This information is mentioned in the embeddings.md file, specifically in the section about the model used by ChromaDB.",
  "sources": [
    {"text": "# ChromaDB\n\nChromaDB is an open-source vector database. It stores documents, their embeddings, metadata and ids in collections, and handles the embedding step automatically if you pass raw text.",
     "source": "chromadb.md", "distance": 0.4948},
    {"text": "ChromaDB uses the all-MiniLM-L6-v2 sentence-transformer model by default. It produces 384-dimensional vectors and runs locally on CPU, so no API key is needed.",
     "source": "embeddings.md", "distance": 0.8273}
  ],
  "confidence": "high"
}
```

When there's nothing to answer from, the model isn't called (so this works with Ollama down) and the response is a 200 with `sources: []` and `confidence: "low"`:

| Case | `answer` |
|---|---|
| Nothing ingested yet | `No documents have been ingested yet. Call POST /ingest first.` |
| No usable chunk within `max_distance` | `No relevant documents found for this question. Try rephrasing it or raising max_distance.` |

### GET /stats

The chunk count plus the settings the running API loaded, so a change to `.env` can be checked without calling `/ask`. `db_path` is the resolved `CHROMA_PATH`. A `~` is expanded, and a relative path is taken from the app folder.

| Status | When | Body |
|---|---|---|
| 200 | | below (with my `.env`) |
| 503 | ChromaDB can't be read | `{"detail": "ChromaDB is not readable"}` |

```
$ curl -s localhost:8000/stats
{"document_count":32,"model":"llama3.2:1b","db_path":"/app/rag_db","ollama_url":"http://ollama:11434","max_results":5,"confidence_threshold":1.2,"debug":true}
```

### GET /health

Counts the collection, and calls Ollama's `/api/tags` (3s timeout), which also lists the pulled models. `model_pulled` says whether `MODEL_NAME` is one of them. A name with no tag counts as `:latest`, the way Ollama treats it. It's `null` when Ollama can't be reached, and `false` when the list can't be read.

| `status` | HTTP | Meaning |
|---|---|---|
| `ok` | 200 | ChromaDB works, Ollama answers and the model is pulled |
| `degraded` | 200 | ChromaDB works, but Ollama is down or doesn't have the model. `/ingest` and retrieval work, but `/ask` returns 503 whenever it has context to send |
| `error` | 503 | ChromaDB can't be read, `document_count` is `null` |

```
$ curl -s localhost:8000/health
{"status":"ok","chromadb":"ok","ollama":"connected","document_count":32,"model":"llama3.2:1b","model_pulled":true}
```

Before `model_pulled` was added, setting `MODEL_NAME` to a model that isn't pulled still gave `"status":"ok"` while every `/ask` was a 503 (see verification step 4).

## Confidence

It starts from the distance of the closest source. The rest of the sources don't change the level.

| Best distance | Level |
|---|---|
| under 0.5 | high |
| 0.5 to under 0.8 | medium |
| 0.8 and up | low |

Then it looks at the answer:

1. If the model says it doesn't have enough information (the wording the system prompt asks for), the level is `low`
2. If the answer has something specific that isn't in the question, the sources or their file names, the level drops by one. That covers a number with 2+ digits, an acronym (`MIT`, `CUDA`), or a link whose host isn't in the sources (`https://github.com/...`)

No sources means `low`. `CONFIDENCE_THRESHOLD` doesn't move these bands, despite its name. It only decides which chunks reach the model, so the same chunks get the same level whatever the cutoff (tested at 0.5, 1.0, 1.2 and 4). The bands and the first two checks were tuned on 51 graded answers in `../docker-compose.yml/README.md`.

### Cutoff check: 1.0 vs 1.2

The new 1.0 default changes which chunks reach the model, so I asked 26 questions through the stack at `max_distance` 1.0 and 1.2 (`n_results` 3). The questions were 12 the docs answer, 6 they don't, 4 vague and 4 two-part. I picked them to land near the cutoff.

- **Confidence was the same at both cutoffs for all 26.** The closest chunk decides the level, and if it's between 1.0 and 1.2 the level is `low` either way.
- **5 questions with the closest chunk between 1.0 and 1.2 no longer reach the model.** At 1.2 they got 3 refusals ("Who created Ollama?", "How much RAM does llama3.1:8b need?", "What distance threshold should I use?"), a made-up `search_path` parameter for restricting a search to one file, and a Client/PersistentClient comparison that was half invented. All were `low`. At 1.0 they get the fixed "No relevant documents found" answer.
- **One correct answer was lost.** For "What port does Ollama use?", the closest chunk (0.9469) is about starting the server, and the port is in the next one at 1.0525. At 1.2 the model answered 11434. At 1.0 it only got the first chunk and said it didn't have enough information. `low` both times.

Answers at the 1.0 default, graded against `docs/`:

| Confidence | Answers | Correct | Partly made up | Wrong | Refused |
|---|---|---|---|---|---|
| high | 0 | | | | |
| medium | 3 | | 3 | | |
| low | 11 | 2 | 1 | 5 | 3 |

The other 12 got "No relevant documents found". No wrong answer came out above `low`. The wrong ones include CUDA support for ChromaDB, an MIT license, streaming with a plain `requests.get` to the server root, and "RAG uses prompt injection".

The run turned up one gap, now fixed. "What happens if I query with a different embedding model than I indexed with?" got a correct answer at 0.5354, but it cited `embeddings.md` with a made-up GitHub link, and came out `medium`. Links are now checked like numbers and acronyms, so it's `low`. A rerun with the fix changed no other level.

Known limits:

- `medium` isn't the same as correct. All 3 here were partly made up: a made-up code sample for `collection.add` around a correct core, the right quote but the wrong reason for "Why do small models make things up?", and the embedding half of "chunking vs embedding" filled in from training data. Made-up code, reasoning and prose without a number, acronym or link isn't caught.
- A right answer can need a chunk that's past the cutoff, like the port. `max_distance: 1.2` on the request or `CONFIDENCE_THRESHOLD=1.2` gets the 1.2 behaviour back.
- Temperature 0 isn't fully repeatable. 3 answers changed wording between the two runs. "How do RAG and prompt injection relate?" got different answers at 1.0 and 1.2 with the same 3 sources, in both runs. No level changed, but the answer checks only ever see one answer.

The real distances and answers from this run are in `tests/test_confidence.py`.

## Verification

All from the repo root with the project venv. No `MODEL_NAME` etc. set in the shell.

**1. No .env, defaults**

```
$ .venv/bin/python module_08/Environment-management-and-configuration/config.py
Settings(
  ollama_url           = 'http://localhost:11434' (str)
  model_name           = 'llama3.2:1b' (str)
  chroma_path          = './rag_db' (str)
  max_results          = 3 (int)
  confidence_threshold = 1.0 (float)
  debug                = False (bool)
)
```

**2. With .env** (`MAX_RESULTS=5`, `CONFIDENCE_THRESHOLD=1.2`, `DEBUG=true`)

```
  max_results          = 5 (int)
  confidence_threshold = 1.2 (float)
  debug                = True (bool)
```

**3. Shell beats .env**

```
$ MODEL_NAME="llama3.2:3b" .venv/bin/python module_08/Environment-management-and-configuration/config.py
  model_name           = 'llama3.2:3b' (str)
  max_results          = 5 (int)          <- the rest still from .env
```

**4. Compose: change `MODEL_NAME` in .env, no code changes**

```
$ curl -s localhost:8000/stats
{"document_count":32,"model":"llama3.2:1b","db_path":"/app/rag_db","ollama_url":"http://ollama:11434","max_results":5,"confidence_threshold":1.2,"debug":true}

$ sed -i '' 's/^MODEL_NAME=.*/MODEL_NAME=llama3.2:3b/' .env
$ docker-compose restart backend && curl -s localhost:8000/stats
{..."model":"llama3.2:1b"...}                   <- restart keeps the old env

$ docker-compose up -d
 Container rag-stack-ollama-1 Recreated
 Container rag-stack-backend-1 Recreated
 Container rag-stack-ollama-1 Healthy
 Container rag-stack-backend-1 Started
$ curl -s localhost:8000/stats
{"document_count":32,"model":"llama3.2:3b","db_path":"/app/rag_db","ollama_url":"http://ollama:11434","max_results":5,"confidence_threshold":1.2,"debug":true}
$ curl -s localhost:8000/health
{"status":"degraded","chromadb":"ok","ollama":"connected","document_count":32,"model":"llama3.2:3b","model_pulled":false}
```

Both containers were recreated from the same images, nothing was rebuilt, and `md5sum` of `config.py` and `my_rag_api.py` inside the container matches the files on disk. `ollama_url` and `db_path` show the `environment:` values beating `.env`. `/health` is `degraded` because that model isn't pulled, and `/ask` returns `503 {"detail": "Ollama returned 404 for model llama3.2:3b"}` until it is (`docker-compose exec ollama ollama pull llama3.2:3b`). Setting it back to `llama3.2:1b` and running `up -d` again restores the original model.

## Tests

```
pip install -r requirements.txt -r requirements-dev.txt
pytest                                                        # 372 passed, 16 skipped
RAG_API_URL=http://localhost:8000 pytest tests/test_live.py   # 16 passed, stack up
```

`conftest.py` sets all six variables before the app is imported, so a local `.env` (with `DEBUG=true`, for example) can't change what the tests see. New or changed for this exercise:

- `test_config.py` (80): defaults and types, every casting rule above, blank values, `repr`, a single shared `settings`, and `config.py` run as a script in a temp folder (no `.env`, with `.env`, shell beating `.env`, a bad value in `.env`, started from another directory). It also checks the ignore files and that `.env.example` matches the defaults. Two more run the app with `DEBUG=true` and `false` in a separate process: logger level, the settings logged at startup, and the traceback (or not) in a 500
- `test_compose.py` (25): `env_file` on both services, `.env` values reaching the containers, `environment:` beating them, every setting `config.py` reads reaching the backend, the Dockerfile copying `config.py`. Skipped without a `.env`
- `test_health_stats.py` (40): `model_pulled` for pulled, missing, untagged, case-different and registry model names, 7 unreadable `/api/tags` bodies, settings changes reaching `/stats`, `/health` and the Ollama call, `/stats` types, `CHROMA_PATH` resolution
- `test_ask.py` (104): `n_results` and `max_distance` defaulting to `MAX_RESULTS` and `CONFIDENCE_THRESHOLD` (a chunk at 1.0 kept, at 1.0001 dropped), debug log lines, none when `DEBUG` is off
- `test_confidence.py` (72): made-up links, the level not depending on the cutoff, off-topic questions at `max_distance` 4 staying `low`, and the real cases from the cutoff check
- `test_live.py` (16): `/stats` and `/health` match `.env` on the running stack, and the model is pulled
- `test_ingest.py` and `test_docs_and_routing.py` came over unchanged apart from the request defaults now coming from settings. They cover `/ingest` (repeats, cleanup, empty or missing folders, CRLF, bad UTF-8, write failures and retries) and every route, field and error code in the OpenAPI schema

Line and branch coverage of `my_rag_api.py` and `config.py` is 99% (`coverage run --branch -m pytest`). The only lines it doesn't count are the `DEBUG=true` setup and `print(settings)`, which the subprocess tests run.

Running the suite against broken copies of this folder makes the matching tests fail. The broken versions were: the starter's `== "true"` DEBUG check, `.env` missing from `.gitignore`, no `COPY config.py`, `CHROMA_PATH` dropped from `environment:`, `/health` ignoring the model list, no `~` expansion, a hardcoded 1.2 default, the app ignoring `DEBUG`, and no link check.

## Changes from the previous exercise

- `max_distance` now defaults to `CONFIDENCE_THRESHOLD`, which is 1.0 per the assignment. The previous exercise used 1.2. My `.env` sets `CONFIDENCE_THRESHOLD=1.2` to keep that behaviour, but without a `.env` the API is stricter and drops chunks between 1.0 and 1.2. See the [cutoff check](#cutoff-check-10-vs-12) for what that changes
- `n_results` defaults to `MAX_RESULTS`, and the model, Ollama URL and db path come from settings
- `/stats` returns every setting
- `/health` has `model_pulled`, and is `degraded` when the model isn't pulled
- confidence drops a level for a link whose host isn't in the sources
- `CHROMA_PATH` expands `~`
