# Dockerfile for RAG FastAPI Backend
# Build:  docker build -t my-rag-api .
# Run:    docker run -d -p 8000:8000 --name rag-container my-rag-api
# Test:   open http://localhost:8000/docs

FROM python:3.11-slim

WORKDIR /app

# chromadb pulls in a few packages that may need to compile (hnswlib etc.)
# when there's no wheel for the platform. Clearing the apt lists in the same
# RUN keeps them out of the layer, otherwise they'd still add ~20MB to the image
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# requirements.txt goes in on its own before any app code. Each instruction is
# a cached layer, and a layer is only rebuilt if its inputs changed, so editing
# my_rag_api.py or docs/ reuses the pip install below instead of redoing it
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY my_rag_api.py .
COPY docs/ ./docs

# documentation only, the actual mapping is done with -p at docker run
EXPOSE 8000

# 0.0.0.0, not the default 127.0.0.1. Inside the container 127.0.0.1 only
# accepts connections from the container itself, so the -p port mapping from
# the host would connect and then get nothing back
CMD ["uvicorn", "my_rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
