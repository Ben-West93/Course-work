# backend/Dockerfile - JobFit API (build context: ./backend)
#
#   builder   runtime packages in /opt/venv, plus the embedding model
#   test      builder + pytest, runs the suite. Only built when asked for:
#               docker build --target test ./backend
#   runtime   what compose runs: the venv, the model and the app code
#
# runtime is last, so it's the default target and `docker-compose up --build`
# never builds the test stage

# the venv copied from builder into runtime only works on the interpreter it was made with
ARG PYTHON_IMAGE=python:3.11-slim


# ── builder ────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS builder

RUN python -m venv --without-pip /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# constraints.txt pins everything requirements.txt pulls in, so a later
# rebuild installs the same versions this one did
COPY requirements.txt constraints.txt ./
RUN pip --python /opt/venv/bin/python install --no-cache-dir -r requirements.txt -c constraints.txt

# chroma downloads all-MiniLM-L6-v2 the first time it embeds anything. Doing
# it here bakes it into the image, so saving and searching postings work
# with no internet and while Ollama is still pulling its model
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm up'])" \
    && rm -f /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz


# ── test ───────────────────────────────────────────────────────────────────
FROM builder AS test

COPY requirements-dev.txt .
RUN pip --python /opt/venv/bin/python install --no-cache-dir -r requirements-dev.txt -c constraints.txt

COPY . .

# --network=none proves the suite needs no Postgres, Ollama or internet:
# SQLite and Chroma in a temp folder, a closed Ollama port, the baked model
RUN --network=none pytest tests/ -q -W error -p no:cacheprovider


# ── runtime ────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS runtime

ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# the API runs as app, not root. /app/chroma is made here so a new
# chroma_data volume starts out owned by app: Docker copies the folder's
# owner into an empty volume the first time it's mounted
RUN useradd --uid 10001 --create-home app && mkdir /app/chroma && chown app:app /app/chroma

COPY --from=builder /opt/venv /opt/venv
# ChromaDB looks for the embedding model under the home folder of whoever runs it
COPY --from=builder --chown=app:app /root/.cache/chroma /home/app/.cache/chroma

# owned by root, so the app can read its code but not change it
COPY auth.py config.py database.py main.py models.py rag_pipeline.py schemas.py ./
# docker-compose exec backend python walkthrough.py, see the README
COPY walkthrough.py ./
COPY seed/ seed/

USER app
EXPOSE 8000

# 0.0.0.0, uvicorn's default 127.0.0.1 only accepts connections from inside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
