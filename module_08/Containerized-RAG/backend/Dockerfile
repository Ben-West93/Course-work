# backend/Dockerfile - FastAPI RAG API (build context: ./backend)

FROM python:3.11-slim

WORKDIR /app

# requirements first, so editing the code or docs/ keeps the pip layer cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# chroma downloads its embedding model (all-MiniLM-L6-v2, ~80MB) the first time
# it embeds anything. Doing it at build time bakes it into the image instead of
# re-downloading it every time the container is recreated
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm up'])" \
    && rm -f /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz

# main.py, rag.py, config.py and docs/ (.dockerignore keeps tests and caches out)
COPY . .

EXPOSE 8000

# 0.0.0.0, uvicorn's default 127.0.0.1 would only accept connections from inside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
