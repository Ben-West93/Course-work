# Multi-Service RAG Stack (Docker Compose)

The RAG FastAPI backend from `Dockerfile-RAG-API` and Ollama running as two containers on one Compose network. This folder is the whole build context: `Dockerfile`, `.dockerignore`, `my_rag_api.py`, `requirements.txt` and `docs/`.

`docker-compose.starter.yml` is the original scaffold, `docker-compose.yml` is the finished version.

Changes to `my_rag_api.py` compared with `Dockerfile-RAG-API/`:

- the ChromaDB path is read from `DB_PATH` instead of `RAG_DB_PATH`, so the value set in compose is actually used
- `GET /` returns `{"message": "RAG API is running", "docs": "/docs"}` (the exercise checks that the root endpoint responds)

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
| `backend` | `8000:8000` | built from `./Dockerfile`, `depends_on: ollama` |
| `ollama` | `11434:11434` | `ollama/ollama` image. The backend doesn't need this port published. It's there so the Mac can reach the container's Ollama too |

`depends_on` only sets the start order. Compose starts the Ollama container first but doesn't wait for it to listen. `/health` checks Ollama on every call, so if the backend comes up first it just reports `"ollama": "disconnected"` until Ollama is ready. After a restart, `/health` reported `connected` within 2s.

Port 11434 can't already be in use on the Mac. If Ollama is running locally (`ollama serve` or the Ollama app), stop it before `docker-compose up` or the ollama container won't start. Port 8000 has the same issue with an old `rag-api-container` or a local uvicorn.

## Volumes

| Volume | Mounted at | Holds | Size after the test |
|---|---|---|---|
| `chroma_data` | `/app/rag_db` (`DB_PATH`) | ChromaDB collection | 516KB, 32 chunks |
| `ollama_models` | `/root/.ollama` | pulled models | 1.32GB |

Both are named volumes declared under the top-level `volumes:`, so Docker stores them outside any container. On disk they're `rag-stack_chroma_data` and `rag-stack_ollama_models` (`docker volume ls`).

- `docker-compose down` removes the containers and the network, and leaves the volumes alone
- `docker-compose down -v` deletes the volumes too, so the chunks and the 1.3GB model are gone

Chroma's embedding model (all-MiniLM-L6-v2, 79MB) is cached in the backend container's own filesystem, not a volume. Every new backend container downloads it again on the first `/ingest` or `/ask`, which needs internet access and adds a few seconds to that first call.

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

`down` removed both containers and `rag-stack_default`, and neither volume was touched. Rebuilding only the backend (`docker-compose up --build -d` after editing `my_rag_api.py`) also kept all 32 chunks, and the ollama container kept running.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | `{"message": "RAG API is running", "docs": "/docs"}` |
| GET | `/health` | ChromaDB and Ollama status, `ok` / `degraded` / `error` |
| GET | `/stats` | `document_count`, `model`, `db_path` |
| POST | `/ingest` | loads `docs/` into ChromaDB (32 chunks from 7 files) |
| POST | `/ask` | `{"question": "..."}` -> answer, sources, confidence |

The full request/response reference is in `Dockerfile-RAG-API/README.md`. The API is the same apart from `GET /`.
