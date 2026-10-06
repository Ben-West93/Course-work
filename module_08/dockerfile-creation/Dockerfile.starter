# Dockerfile for RAG FastAPI Backend  (STARTER)
# ================================================
# Build:  docker build -t my-rag-api .
# Run:    docker run -p 8000:8000 my-rag-api
# Test:   open http://localhost:8000/docs

# TODO: Choose a base image — use python:3.11-slim for a small footprint
FROM # TODO

# TODO: Set the working directory inside the container to /app
WORKDIR # TODO

# TODO: (Optional) Install system build dependencies if chromadb needs them
#   RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# --- Layer caching trick ---
# Copy requirements FIRST (before any application code).
# Docker caches each layer. If requirements.txt hasn't changed,
# the pip install step is skipped on rebuilds — saving minutes.

# TODO: COPY requirements.txt .
# TODO: RUN pip install --no-cache-dir -r requirements.txt

# --- Application code ---
# TODO: COPY your main application file(s) into /app
# TODO: COPY the docs/ folder into /app/docs

# TODO: EXPOSE the port uvicorn will listen on (8000)

# TODO: CMD to run uvicorn:
#   - module:app  (e.g. my_rag_api:app)
#   - --host 0.0.0.0   (IMPORTANT: binds to all interfaces, not just localhost)
#   - --port 8000
CMD # TODO
