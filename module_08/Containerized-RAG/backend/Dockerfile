# backend/Dockerfile - FastAPI RAG API (build context: ./backend)
#
# Three stages:
#   builder   installs the runtime packages into /opt/venv and downloads the embedding model
#   test      builder + pytest, runs the test suite. Only built when asked for:
#               docker build --target test ./backend
#   runtime   what compose runs: the venv, the model and the app code, nothing else
#
# The last stage is the default target, so `docker-compose up --build` never
# builds the test stage and stays as quick as a single-stage build.

# one place for the base image. The venv is copied from builder into runtime and
# only works on the exact interpreter it was created with
ARG PYTHON_IMAGE=python:3.11-slim


# ── builder ────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS builder

# a venv keeps everything installed under one folder, so runtime can take it
# with a single COPY. --without-pip because the base image already has pip, a
# second copy inside the venv would be 15MB of dead weight in the final image
RUN python -m venv --without-pip /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# requirements before the code, so editing the app doesn't invalidate this
# layer. --python points the base image's pip at the venv
COPY requirements.txt .
RUN pip --python /opt/venv/bin/python install --no-cache-dir -r requirements.txt

# chroma downloads its embedding model (all-MiniLM-L6-v2) the first time it
# embeds anything. Doing it at build time bakes it in, so containers start
# without a download. The archive isn't needed once it's extracted
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm up'])" \
    && rm -f /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz


# ── test ───────────────────────────────────────────────────────────────────
FROM builder AS test

COPY requirements-dev.txt .
RUN pip --python /opt/venv/bin/python install --no-cache-dir -r requirements-dev.txt

# the whole context, tests/ and pytest.ini included
COPY . .

# any failing test fails the build, and -W error fails it on warnings too.
# --network=none proves the suite needs no Ollama or internet: it uses a temp
# db, a closed Ollama port and the model baked in above
RUN --network=none pytest tests/ -q -W error -p no:cacheprovider


# ── runtime ────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS runtime

# venv first on PATH, so uvicorn and the compose healthcheck's `python` both use it
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/chroma /root/.cache/chroma

# only the app. Tests, pytest.ini and the dev requirements never get here
COPY config.py main.py rag.py ./
COPY docs/ docs/

EXPOSE 8000

# 0.0.0.0, uvicorn's default 127.0.0.1 would only accept connections from inside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
