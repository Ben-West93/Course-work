# Checks docker-compose.yml as Compose itself reads it (docker-compose config),
# so the YAML is parsed the same way `up` would. Needs docker-compose installed
# but not running, and a .env (cp .env.example .env). Skipped without them.

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent.parent
COMPOSE = shutil.which("docker-compose")
pytestmark = [
    pytest.mark.skipif(not COMPOSE, reason="docker-compose not installed"),
    # env_file: .env makes compose refuse to run without it
    pytest.mark.skipif(not (HERE / ".env").exists(), reason="no .env, cp .env.example .env"),
]


def compose_config(**env_overrides):
    env = {k: v for k, v in os.environ.items() if k != "OLLAMA_HOST_PORT"}
    env.update(env_overrides)
    # --env-file replaces the file compose reads for ${...} in the YAML, so an
    # OLLAMA_HOST_PORT=11435 in .env doesn't change the ports checked here.
    # The services' env_file: .env is separate and still read
    out = subprocess.run(
        [COMPOSE, "--env-file", os.devnull, "config", "--format", "json"], cwd=HERE, env=env,
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def config():
    return compose_config()


@pytest.fixture(scope="module")
def backend_svc(config):
    return config["services"]["backend"]


@pytest.fixture(scope="module")
def ollama_svc(config):
    return config["services"]["ollama"]


def ports(service):
    return [(p["published"], p["target"]) for p in service["ports"]]


def mounts(service):
    return {v["source"]: v["target"] for v in service["volumes"] if v["type"] == "volume"}


def test_two_services_and_a_fixed_project_name(config):
    assert set(config["services"]) == {"backend", "ollama"}
    # the folder name would otherwise turn into "docker-composeyml"
    assert config["name"] == "rag-stack"


def test_backend_is_built_from_this_folder(backend_svc):
    assert backend_svc["build"] == {"context": str(HERE), "dockerfile": "Dockerfile"}


def test_ollama_uses_the_official_image(ollama_svc):
    assert ollama_svc["image"] == "ollama/ollama"
    assert "build" not in ollama_svc


# ── ports ──────────────────────────────────────────────────────────────────

def test_ports_match_the_exercise_by_default(backend_svc, ollama_svc):
    assert ports(backend_svc) == [("8000", 8000)]
    assert ports(ollama_svc) == [("11434", 11434)]


def test_ports_are_only_published_on_loopback(backend_svc, ollama_svc):
    # without a host_ip Docker listens on 0.0.0.0, i.e. the whole network
    for service in (backend_svc, ollama_svc):
        assert [p.get("host_ip") for p in service["ports"]] == ["127.0.0.1"]


@pytest.mark.parametrize("host_port", ["11435", "21434"])
def test_ollama_host_port_can_be_moved(host_port):
    services = compose_config(OLLAMA_HOST_PORT=host_port)["services"]
    # only the Mac side moves, the backend still talks to 11434 on the network
    assert ports(services["ollama"]) == [(host_port, 11434)]
    assert services["ollama"]["ports"][0]["host_ip"] == "127.0.0.1"
    assert services["backend"]["environment"]["OLLAMA_URL"] == "http://ollama:11434"


# ── backend -> ollama ──────────────────────────────────────────────────────

def test_backend_reaches_ollama_by_service_name(config, backend_svc, ollama_svc):
    url = urlparse(backend_svc["environment"]["OLLAMA_URL"])
    assert url.hostname == "ollama"
    assert url.hostname in config["services"]
    assert url.port == ollama_svc["ports"][0]["target"]


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"])
def test_ollama_url_doesnt_point_outside_the_compose_network(backend_svc, host):
    assert host not in backend_svc["environment"]["OLLAMA_URL"]


def test_backend_waits_for_ollama_to_be_healthy(backend_svc):
    assert backend_svc["depends_on"]["ollama"]["condition"] == "service_healthy"


def test_ollama_healthcheck(ollama_svc):
    check = ollama_svc["healthcheck"]
    assert check["test"] == ["CMD", "ollama", "list"]
    # fast while starting so the backend isn't held up, slow after that
    assert check["start_interval"] == "1s"
    assert check["interval"] == "30s"


# ── volumes ────────────────────────────────────────────────────────────────

def test_named_volumes_are_declared(config, backend_svc, ollama_svc):
    assert set(config["volumes"]) == {"chroma_data", "ollama_models"}
    assert set(mounts(backend_svc)) | set(mounts(ollama_svc)) == set(config["volumes"])


def test_chroma_volume_is_where_the_app_writes(backend_svc):
    # if these drift apart chroma writes outside the volume and the
    # document count resets on every down/up
    assert mounts(backend_svc) == {"chroma_data": "/app/rag_db"}
    assert backend_svc["environment"]["CHROMA_PATH"] == "/app/rag_db"


def test_ollama_models_volume(ollama_svc):
    assert mounts(ollama_svc) == {"ollama_models": "/root/.ollama"}


# ── compose, Dockerfile and app agree ──────────────────────────────────────

def read_by_config():
    return set(re.findall(r'env_\w+\("(\w+)"', (HERE / "config.py").read_text()))


def test_app_reads_every_env_var_compose_sets():
    # a name that's slightly off (DB_PATH vs CHROMA_PATH) would be silently
    # ignored. Checked as written, since the merged environment also has
    # anything else in .env, like OLLAMA_HOST_PORT
    raw = yaml.safe_load((HERE / "docker-compose.yml").read_text())
    names = {item.split("=", 1)[0] for item in raw["services"]["backend"]["environment"]}
    assert names == {"OLLAMA_URL", "CHROMA_PATH"}
    assert names <= read_by_config()


# ── env_file ───────────────────────────────────────────────────────────────

def test_both_services_load_env_file():
    # `config` folds env_file into environment, so check the file as written
    raw = yaml.safe_load((HERE / "docker-compose.yml").read_text())
    assert raw["services"]["backend"]["env_file"] == ".env"
    assert raw["services"]["ollama"]["env_file"] == ".env"


def test_values_from_env_file_reach_the_backend(backend_svc):
    env = dotenv_values(HERE / ".env")
    for key in ("MODEL_NAME", "MAX_RESULTS", "CONFIDENCE_THRESHOLD", "DEBUG"):
        if env.get(key):
            assert backend_svc["environment"][key] == env[key]


def test_environment_beats_env_file(backend_svc):
    # .env points at localhost for running on the Mac, the container must not
    assert backend_svc["environment"]["OLLAMA_URL"] == "http://ollama:11434"
    assert backend_svc["environment"]["CHROMA_PATH"] == "/app/rag_db"


def test_backend_gets_every_setting(backend_svc):
    assert read_by_config() <= set(backend_svc["environment"])


def test_ollama_gets_env_file_as_is(ollama_svc):
    # no environment: block on ollama, so nothing overrides the file there
    env = {k: v for k, v in dotenv_values(HERE / ".env").items() if v}
    assert {k: ollama_svc["environment"].get(k) for k in env} == env


def test_dockerfile_serves_on_the_mapped_port(backend_svc):
    dockerfile = (HERE / "Dockerfile").read_text()
    target = backend_svc["ports"][0]["target"]
    assert f"EXPOSE {target}" in dockerfile
    assert f'"--host", "0.0.0.0", "--port", "{target}"' in dockerfile


def test_embedding_model_is_downloaded_before_the_app_is_copied():
    # after pip install so chromadb is there, before COPY my_rag_api.py so
    # editing the app doesn't download the model again
    dockerfile = (HERE / "Dockerfile").read_text()
    pip = dockerfile.index("pip install")
    warm_up = dockerfile.index("DefaultEmbeddingFunction()")
    copy_app = dockerfile.index("COPY config.py my_rag_api.py")
    assert pip < warm_up < copy_app


def test_dockerfile_copies_config():
    # my_rag_api imports config, without it the container dies on startup
    copies = [line for line in (HERE / "Dockerfile").read_text().splitlines() if line.startswith("COPY")]
    assert any("config.py" in line.split() for line in copies)
