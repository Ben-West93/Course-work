# Configurable RAG Application

The Compose RAG stack from `../docker-compose.yml/`, with every setting moved out of the code and into environment variables. `config.py` reads them into a single `Settings` object, `.env` holds local values, and `docker-compose.yml` passes the same file into the backend container with `env_file`.

| File | |
|---|---|
| `config.py` | `Settings` class and the `settings` instance everything imports |
| `starter.py` | the original scaffold for `config.py`, unchanged |
| `.env.example` | committed, every setting with its default and a comment |
| `.env` | local values, ignored by git and docker |
| `my_rag_api.py` | the FastAPI app, now reads `settings.*` instead of its own constants |
| `docker-compose.yml`, `Dockerfile` | `env_file: .env` on the backend, `config.py` copied into the image |

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
```

- `env_file` turns every line of `.env` into an env var in the backend container. The image has no `.env` in it, so `load_dotenv` finds nothing and `config.py` just reads what Compose set.
- `environment:` wins over `env_file`. `.env` has `localhost` and `./rag_db` for running on the Mac. Inside the container `localhost` is the container itself, so these two are fixed in the compose file. `docker-compose config` shows the merged result.
- Compose only reads `.env` when it creates a container. After editing it, run `docker-compose up -d` (which recreates the backend). `docker-compose restart` keeps the old values.
- A variable set in the shell doesn't override `env_file` inside the container. Shell variables only fill `${...}` in the YAML, like `OLLAMA_HOST_PORT`.
- Without a `.env`, `docker-compose up` stops with `env file .../.env not found`. Run `cp .env.example .env` first.
- `ollama` doesn't get `env_file`. It doesn't read any of these settings, and with `env_file` every edit to `.env` would also restart Ollama.
- The project name is still `rag-stack`, so this reuses the previous exercise's volumes, including the pulled model.

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
 Container rag-stack-ollama-1 Running
 Container rag-stack-backend-1 Recreated
$ curl -s localhost:8000/stats
{"document_count":32,"model":"llama3.2:3b","db_path":"/app/rag_db","ollama_url":"http://ollama:11434","max_results":5,"confidence_threshold":1.2,"debug":true}
$ curl -s localhost:8000/health
{"status":"ok","chromadb":"ok","ollama":"connected","document_count":32,"model":"llama3.2:3b"}
```

Only the backend was recreated, from the same image, and `md5sum` of `config.py` and `my_rag_api.py` inside the container matches the files on disk. `ollama_url` and `db_path` show the `environment:` values beating `.env`. `/ask` then returns `503 {"detail": "Ollama returned 404 for model llama3.2:3b"}` until that model is pulled (`docker-compose exec ollama ollama pull llama3.2:3b`). Setting it back to `llama3.2:1b` and running `up -d` again restores the original model.

## Tests

```
pip install -r requirements.txt -r requirements-dev.txt
pytest                                                        # 316 passed, 16 skipped
RAG_API_URL=http://localhost:8000 pytest tests/test_live.py   # 16 passed, stack up
```

`conftest.py` sets all six variables before the app is imported, so a local `.env` (with `DEBUG=true`, for example) can't change what the tests see. New or changed for this exercise:

- `test_config.py` (78): defaults and types, every casting rule above, blank values, `repr`, a single shared `settings`, and `config.py` run as a script in a temp folder (no `.env`, with `.env`, shell beating `.env`, a bad value in `.env`, started from another directory). It also checks the ignore files and that `.env.example` matches the defaults
- `test_compose.py` (25): `env_file` on the backend only, `.env` values reaching the container, `environment:` beating them, every setting `config.py` reads reaching the backend, the Dockerfile copying `config.py`. Skipped without a `.env`
- `test_live.py`: `/stats` and `/health` match `.env` on the running stack

Running the suite against broken copies of this folder (the starter's `== "true"` DEBUG check, `.env` missing from `.gitignore`, no `COPY config.py`, `CHROMA_PATH` dropped from `environment:`) makes the matching tests fail.

## Change from the previous exercise

`max_distance` now defaults to `CONFIDENCE_THRESHOLD`, which is 1.0 per the assignment. The previous exercise used 1.2, picked after grading 51 questions (see `../docker-compose.yml/README.md`). My `.env` sets `CONFIDENCE_THRESHOLD=1.2` to keep that behaviour, but without a `.env` the API is stricter and drops chunks between 1.0 and 1.2.
