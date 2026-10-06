# docker-first-steps

The guided FastAPI example from the Docker First Steps exercise. `app.py` has two routes, `/` and `/health`, and the `Dockerfile` installs fastapi and uvicorn on `python:3.11-slim` and runs uvicorn on port 8000.

The full RAG API Dockerfile is in `../dockerfile-creation`.

## Build and run

```
docker build -t my-first-api .
docker run -d -p 8000:8000 --name first-api my-first-api
```

```
curl http://localhost:8000/          # {"message":"Hello from Docker!","status":"running"}
curl http://localhost:8000/health    # {"status":"healthy"}
```

`/docs` also works, since FastAPI generates it.

## Warm-up

```
docker run hello-world
docker run -it python:3.11-slim bash
  python --version        # Python 3.11.17
  pip install requests
  exit
```

## Management commands

```
docker ps                # running containers
docker ps -a             # including stopped
docker images            # local images
docker stop first-api
docker rm first-api
```

`stop` and `rm` also take the container ID from `docker ps`. A container has to be stopped before `docker rm` will remove it, unless you use `rm -f`.

## Questions

**1. What happens when you `docker run` the same image twice?**

You get two separate containers from the same read-only image, each with its own ID, name and writable filesystem. They can't publish the same host port, though. Starting a second one with `-p 8000:8000` fails with `Bind for 0.0.0.0:8000 failed: port is already allocated`, so it needs a different host port such as `-p 8001:8000`.

**2. What happens to files created inside a container when it stops?**

They stay in that container's writable layer, so `docker stop` then `docker start` still has them. `docker rm` deletes the layer along with the files. They are never saved into the image, so a new `docker run` starts from a clean copy. To keep data, mount a volume with `-v`.

**3. How is the `-p` flag used to map ports?**

`-p HOST:CONTAINER`. `-p 8000:8000` sends traffic on the Mac's port 8000 to port 8000 in the container, and `-p 8001:8000` would serve the same app at `localhost:8001`. The app inside has to listen on `0.0.0.0` (hence `--host 0.0.0.0` in the CMD). Forwarded traffic doesn't arrive on the container's `127.0.0.1`.
