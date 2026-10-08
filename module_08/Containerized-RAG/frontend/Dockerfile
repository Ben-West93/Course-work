# frontend/Dockerfile - Streamlit chat UI (build context: ./frontend)
#
# Single stage on purpose. Streamlit and requests install from prebuilt wheels,
# so there are no build tools or test packages to leave behind. A multi-stage
# version measured 810MB against 825MB, the difference being pip's own bytecode
# cache, which wasn't worth a second stage (see the README)

# same base as the backend
FROM python:3.11-slim

WORKDIR /app

# requirements before the code, so editing app.py doesn't rerun pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# no browser to open in a container, no usage stats sent anywhere, and no
# Deploy button (that's for Streamlit's own cloud)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_TOOLBAR_MODE=minimal

# only the app, the starter is kept out by .dockerignore
COPY app.py .

EXPOSE 8501

# --server.address=0.0.0.0 so the published port can reach it from outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
