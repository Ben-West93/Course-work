# frontend/Dockerfile - Streamlit UI (build context: ./frontend)
#
# Single stage: streamlit and requests install from prebuilt wheels, there
# are no build tools or test packages to leave behind

# same base as the backend
FROM python:3.11-slim

WORKDIR /app

# requirements before the code, so editing app.py doesn't rerun pip.
# constraints.txt pins everything they pull in, so a later rebuild installs
# the same versions this one did
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# no browser to open in a container, no usage stats, no Deploy button
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_TOOLBAR_MODE=minimal

COPY app.py api_client.py ./

# runs as app, not root. Streamlit keeps its settings under the home folder
RUN useradd --uid 10001 --create-home app
USER app

EXPOSE 8501

# 0.0.0.0 so the published port can reach it from outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
