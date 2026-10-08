# frontend/Dockerfile
# Build context: ./frontend (build: in docker-compose.yml)
# Runs the Streamlit UI

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# no browser to open in a container, no usage stats sent anywhere, and no
# Deploy button (that's for Streamlit's own cloud)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_TOOLBAR_MODE=minimal

COPY . .

EXPOSE 8501

# Streamlit needs --server.address=0.0.0.0 to be reachable from outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
