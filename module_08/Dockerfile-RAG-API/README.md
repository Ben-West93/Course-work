# Dockerfile-RAG-API

The RAG FastAPI backend from `module_08/my_rag_api` built into a Docker image. This folder is the whole build context: `my_rag_api.py`, `requirements.txt` and `docs/` are all here.

`Dockerfile.starter` and `dockerignore.starter` are the original scaffolds, `Dockerfile` and `.dockerignore` are the finished versions.

`my_rag_api.py` is the same as in `my_rag_api/` except `OLLAMA_URL` is read from an environment variable (default `http://localhost:11434`), so the container can be pointed at Ollama on the host.

## Build

```
docker build -t my-rag-api .                                  # from this folder
docker build -t my-rag-api module_08/Dockerfile-RAG-API/      # from the repo root
```

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

`rag_db/` is in `.dockerignore`, so every new container starts with an empty collection. Run `POST /ingest` first. The first ingest in a container also downloads Chroma's embedding model (all-MiniLM-L6-v2, 79MB), so it's slow.

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

Rebuild output:

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

Only the changed file and the step after it ran again. If `requirements.txt` changes, pip and everything after it reruns. With a single `COPY . .` before `pip install`, any edit at all would reinstall chromadb.

## Endpoint tests

Tested with curl against the running container. Swagger UI is at http://localhost:8000/docs.

| Request | Result |
|---|---|
| `GET /docs` | 200 |
| `GET /health` (no `OLLAMA_URL`) | 200, `"status": "degraded"`, `"ollama": "disconnected"` |
| `GET /health` (with `OLLAMA_URL`) | 200, `"status": "ok"`, `"ollama": "connected"` |
| `POST /ask` before `/ingest` | 200, "No relevant documents found. Try /ingest first..." |
| `POST /ingest` | 200, `Ingested 37 chunks from 7 files` (31s first time, model download) |
| `POST /ask` "What is RAG?" (no `OLLAMA_URL`) | 503, `Ollama service is unavailable` |
| `POST /ask` "What is RAG?" (with `OLLAMA_URL`) | 200, confidence medium, sources `rag_overview.md` (0.661, 0.718), `prompt_injection.md` (0.986) |
| `POST /ask` off-topic, `max_distance` 0.8 | 200, no sources, LLM not called |
| `POST /ask` with `"question": "   "` | 422 |
| `GET /stats` | 200, 37 documents, `db_path` `/app/rag_db` |

```
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ingest
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What is RAG?"}'
```

The right chunks came back for "What is RAG?", but llama3.2:1b still expanded the acronym wrong in its answer. That's the 1B model, not the container.

## Checking the container

```
docker ps                                 # Up, 0.0.0.0:8000->8000/tcp
docker logs rag-api-container             # Uvicorn running on http://0.0.0.0:8000 + request log
docker top rag-api-container              # uvicorn my_rag_api:app --host 0.0.0.0 --port 8000
docker run --rm my-rag-api ls /app        # docs, my_rag_api.py, requirements.txt only
docker stop rag-api-container
docker rm rag-api-container
```

## Things that can go wrong

- Port 8000 already in use on the host: `docker run` fails with "port is already allocated". Stop whatever is using it, or map another port (`-p 8001:8000`).
- `--name rag-api-container` already exists (even stopped): `docker rm -f rag-api-container` first.
- No `--host 0.0.0.0`: uvicorn listens on the container's 127.0.0.1, `-p` forwards the connection but nothing answers.
- Missing `requirements.txt` or `docs/` in the build context: the `COPY` step fails.
- Ollama not reachable: `/health` says degraded and `/ask` returns 503. Retrieval and `/ingest` still work.
