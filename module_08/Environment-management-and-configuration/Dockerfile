# Dockerfile for RAG FastAPI Backend
# Built by the backend service in docker-compose.yml (build: .)
# Run:    docker-compose up --build -d
# Test:   open http://localhost:8000/docs

FROM python:3.11-slim

WORKDIR /app

# gcc/g++ for chromadb's native deps in case pip can't find a wheel.
# apt's package lists are deleted in the same RUN so they never get saved
# into this layer (removing them in a later RUN wouldn't shrink the image)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# requirements.txt is copied on its own, before the app code. Docker only
# reruns a step when its inputs change, so edits to the .py files or docs/
# leave the pip install layer cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# chroma downloads its embedding model (all-MiniLM-L6-v2) the first time it
# embeds anything. Doing that here puts it in the image, so new containers
# don't fetch it again and /ingest works without internet. Chroma only checks
# for the extracted files, so the 79MB archive can go in the same RUN
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm up'])" \
    && rm /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz

COPY config.py my_rag_api.py ./
COPY docs/ ./docs

# just documents the port, ports: in docker-compose.yml is what publishes it
EXPOSE 8000

# uvicorn defaults to 127.0.0.1, which inside a container only accepts
# connections from the container itself. 0.0.0.0 lets traffic coming in
# through the port mapping reach the app
CMD ["uvicorn", "my_rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
