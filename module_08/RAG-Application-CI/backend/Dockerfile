# backend/Dockerfile
# Build context: ./backend (build: in docker-compose.yml)
# Runs the FastAPI RAG API via uvicorn

FROM python:3.11-slim

WORKDIR /app

# requirements.txt goes in on its own before the code. Docker only reruns a
# step when its inputs change, so editing main.py or docs/ keeps the pip
# install layer cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# chroma downloads its embedding model (all-MiniLM-L6-v2) the first time it
# embeds anything. Doing it here puts it in the image, so a recreated
# container doesn't fetch it again. Chroma only checks for the extracted
# files, so the 79MB archive can go in the same RUN
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm up'])" \
    && rm /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz

# main.py, config.py and docs/. .dockerignore keeps out the starters,
# a local rag_db/ and caches
COPY . .

EXPOSE 8000

# uvicorn defaults to 127.0.0.1, which inside a container only accepts
# connections from the container itself
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
