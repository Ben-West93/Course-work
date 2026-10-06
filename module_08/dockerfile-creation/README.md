# dockerfile-creation

The `my_rag_api` FastAPI backend packaged as a Docker image. This folder has everything the build needs (`my_rag_api.py`, `docs/`, `requirements.txt`), so it can be used as the build context on its own.

`Dockerfile.starter` and `dockerignore.starter` are the original scaffolds. `Dockerfile` and `.dockerignore` are the finished versions.

The only change to `my_rag_api.py` from `module_08/my_rag_api` is that `OLLAMA_URL` now comes from an environment variable. The default is still `http://localhost:11434`, but inside a container `localhost` means the container itself, so it has to be pointed at the host.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | `python:3.11-slim`, build-essential, pip install, app + docs, uvicorn on `0.0.0.0:8000` |
| `.dockerignore` | keeps caches, venvs, `.git`, `.env`, `rag_db/` etc. out of the build context |
| `requirements.txt` | fastapi, uvicorn, chromadb, requests, pydantic (pinned) |
| `my_rag_api.py` | the RAG API (`/ask`, `/ingest`, `/stats`, `/health`) |
| `docs/` | 7 files, 37 chunks after `/ingest` |

## Build

```
docker build -t my-rag-api .                                  # from this folder
docker build -t my-rag-api module_08/dockerfile-creation/     # from the repo root
```

## Run

```
docker run -d -p 8000:8000 --name rag-container my-rag-api
```

- `-d` runs it in the background
- `-p 8000:8000` maps host port 8000 to container port 8000
- `--name` gives it a name, so the management commands below don't need the container ID

To let `/ask` reach Ollama running on the Mac:

```
docker run -d -p 8000:8000 -e OLLAMA_URL=http://host.docker.internal:11434 --name rag-container my-rag-api
```

Without that, `/health` reports `"status": "degraded"` and `/ask` returns 503. Retrieval and `/ingest` still work.

## Testing

```
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/docs     # 200
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ingest                                # 37 chunks from 7 files
```

`rag_db/` is excluded from the image, so the collection starts empty in every new container and `/ingest` has to be run first. The first `/ingest` also downloads Chroma's default embedding model (about 80MB) inside the container, so that request is slow.

## Layer caching

`requirements.txt` is copied and installed before the app code. To check that it works, change a comment in `my_rag_api.py` and rebuild:

```
docker build -t my-rag-api .
```

The `apt-get` and `pip install` steps should show `CACHED`, and only `COPY my_rag_api.py` and the steps after it run again. If `requirements.txt` changes, pip runs again, along with every step after it.

If the Dockerfile did `COPY . .` before `pip install`, any edit to any file would invalidate the cache and reinstall chromadb every time.

## Managing containers

```
docker ps                     # running containers
docker ps -a                  # including stopped ones
docker images                 # local images
docker logs rag-container     # uvicorn output
docker stop rag-container
docker rm rag-container       # has to be stopped first (or use rm -f)
```

`docker run --name rag-container` fails if a container with that name already exists, even a stopped one, so `docker rm` it before running again.

## Docker First Steps

Warm-up commands from the First Steps exercise:

```
docker run hello-world
docker run -it python:3.11-slim bash
  python --version
  pip install requests
  exit
```

### Questions

**1. What happens when you `docker run` the same image twice?**

You get two separate containers. Each one has its own ID, name, filesystem layer and process, even though both start from the same read-only image. Running them both with `-d` works, but they can't both publish the same host port, so the second `-p 8000:8000` fails with "port is already allocated". Giving them the same `--name` fails too.

**2. What happens to files created inside a container when it stops?**

They're kept in the container's writable layer while the container exists, so `docker stop` followed by `docker start` brings them back. `docker rm` deletes that layer and the files go with it. They never go into the image, so a new `docker run` starts clean. Anything that has to survive needs a volume or bind mount (`-v`). Here that would be `rag_db/`, which is why `/ingest` has to be rerun in each new container.

**3. How is the `-p` flag used to map ports?**

`-p HOST:CONTAINER`. `-p 8000:8000` forwards port 8000 on the Mac to port 8000 in the container. The two numbers don't have to match: `-p 9000:8000` would serve the same app at `localhost:9000`. `EXPOSE` in the Dockerfile only documents the port and doesn't publish anything. The app also has to listen on `0.0.0.0` inside the container, because traffic forwarded by `-p` doesn't arrive on the container's `127.0.0.1`.
