import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import config
from config import Settings

HERE = Path(__file__).resolve().parent.parent
KEYS = ["OLLAMA_URL", "MODEL_NAME", "CHROMA_PATH", "MAX_RESULTS", "CONFIDENCE_THRESHOLD", "DEBUG"]
DEFAULTS = {
    "ollama_url": "http://localhost:11434",
    "model_name": "llama3.2:1b",
    "chroma_path": "./rag_db",
    "max_results": 3,
    "confidence_threshold": 1.0,
    "debug": False,
}


@pytest.fixture
def env(monkeypatch):
    # Settings() only reads os.environ (the .env is loaded once at import),
    # so clearing these gives the same result as no .env and nothing exported
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)

    def set_vars(**values):
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        return Settings()
    return set_vars


# ── defaults and types ─────────────────────────────────────────────────────

def test_defaults(env):
    s = env()
    assert vars(s) == DEFAULTS
    assert type(s.max_results) is int
    assert type(s.confidence_threshold) is float


def test_env_vars_override_defaults(env):
    s = env(OLLAMA_URL="http://ollama:11434", MODEL_NAME="llama3.2:3b", CHROMA_PATH="/app/rag_db",
            MAX_RESULTS="5", CONFIDENCE_THRESHOLD="0.8", DEBUG="true")
    assert vars(s) == {
        "ollama_url": "http://ollama:11434",
        "model_name": "llama3.2:3b",
        "chroma_path": "/app/rag_db",
        "max_results": 5,
        "confidence_threshold": 0.8,
        "debug": True,
    }


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_value_uses_the_default(env, key, blank):
    assert vars(env(**{key: blank})) == DEFAULTS


def test_surrounding_whitespace_is_ignored(env):
    s = env(MODEL_NAME="  llama3.2:3b ", MAX_RESULTS=" 4 ", CONFIDENCE_THRESHOLD=" 0.9 ")
    assert (s.model_name, s.max_results, s.confidence_threshold) == ("llama3.2:3b", 4, 0.9)


# ── MAX_RESULTS ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [("1", 1), ("20", 20), ("07", 7), ("+3", 3)])
def test_max_results_valid(env, raw, expected):
    assert env(MAX_RESULTS=raw).max_results == expected


@pytest.mark.parametrize("raw", ["three", "2.5", "3e1", "0x10", "3 results"])
def test_max_results_not_a_whole_number(env, raw):
    with pytest.raises(ValueError, match=f"MAX_RESULTS must be a whole number, got '{raw}'"):
        env(MAX_RESULTS=raw)


@pytest.mark.parametrize("raw", ["0", "-1", "21", "1000"])
def test_max_results_out_of_range(env, raw):
    with pytest.raises(ValueError, match="MAX_RESULTS must be between 1 and 20"):
        env(MAX_RESULTS=raw)


# ── CONFIDENCE_THRESHOLD ───────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [("0", 0.0), ("4", 4.0), ("1.2", 1.2), (".5", 0.5), ("1e0", 1.0)])
def test_threshold_valid(env, raw, expected):
    assert env(CONFIDENCE_THRESHOLD=raw).confidence_threshold == expected


@pytest.mark.parametrize("raw", ["high", "1,2", "0.8.1"])
def test_threshold_not_a_number(env, raw):
    with pytest.raises(ValueError, match=f"CONFIDENCE_THRESHOLD must be a number, got '{raw}'"):
        env(CONFIDENCE_THRESHOLD=raw)


@pytest.mark.parametrize("raw", ["-0.1", "4.01", "nan", "NaN", "inf", "-inf", "Infinity"])
def test_threshold_out_of_range_or_not_finite(env, raw):
    with pytest.raises(ValueError, match="CONFIDENCE_THRESHOLD must be between 0.0 and 4.0"):
        env(CONFIDENCE_THRESHOLD=raw)


# ── DEBUG ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", ["true", "True", "TRUE", " true ", "1", "yes", "Yes", "on"])
def test_debug_true(env, raw):
    assert env(DEBUG=raw).debug is True


@pytest.mark.parametrize("raw", ["false", "False", "0", "no", "off", "OFF"])
def test_debug_false(env, raw):
    assert env(DEBUG=raw).debug is False


@pytest.mark.parametrize("raw", ["ture", "enabled", "2", "y"])
def test_debug_unknown_value_fails(env, raw):
    with pytest.raises(ValueError, match=f"DEBUG must be true or false, got '{raw}'"):
        env(DEBUG=raw)


# ── OLLAMA_URL ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", ["http://ollama:11434/", "http://ollama:11434//"])
def test_ollama_url_trailing_slash_is_dropped(env, raw):
    assert env(OLLAMA_URL=raw).ollama_url == "http://ollama:11434"


@pytest.mark.parametrize("raw", ["localhost:11434", "ollama", "ftp://ollama:11434"])
def test_ollama_url_needs_a_scheme(env, raw):
    with pytest.raises(ValueError, match="OLLAMA_URL must start with http:// or https://"):
        env(OLLAMA_URL=raw)


# ── repr and the singleton ─────────────────────────────────────────────────

def test_repr_lists_every_setting_with_its_type(env):
    text = repr(env(MAX_RESULTS="5"))
    assert text.startswith("Settings(\n") and text.endswith("\n)")
    assert "max_results          = 5 (int)" in text
    assert "confidence_threshold = 1.0 (float)" in text
    assert "debug                = False (bool)" in text
    assert "model_name           = 'llama3.2:1b' (str)" in text
    assert len(text.splitlines()) == len(KEYS) + 2


def test_settings_is_one_shared_instance():
    import my_rag_api
    assert my_rag_api.settings is config.settings


# ── running config.py ──────────────────────────────────────────────────────
# these run the real script in a copy of the folder, so the .env loading
# (which only happens on import) is checked too

@pytest.fixture
def run_config(tmp_path):
    shutil.copy(HERE / "config.py", tmp_path / "config.py")
    clean_env = {k: v for k, v in os.environ.items() if k not in KEYS}

    def run(dotenv=None, **shell):
        if dotenv is not None:
            (tmp_path / ".env").write_text(dotenv)
        out = subprocess.run([sys.executable, "config.py"], cwd=tmp_path, env={**clean_env, **shell},
                             capture_output=True, text=True)
        return out
    return run


def test_script_without_env_file_uses_defaults(run_config):
    out = run_config()
    assert out.returncode == 0
    assert "model_name           = 'llama3.2:1b' (str)" in out.stdout
    assert "debug                = False (bool)" in out.stdout


def test_script_reads_env_file(run_config):
    out = run_config("# comment\nMODEL_NAME=llama3.2:3b\nMAX_RESULTS='5'\nDEBUG=true\n")
    assert "model_name           = 'llama3.2:3b' (str)" in out.stdout
    assert "max_results          = 5 (int)" in out.stdout
    assert "debug                = True (bool)" in out.stdout


def test_shell_beats_env_file(run_config):
    out = run_config("MODEL_NAME=llama3.2:3b\n", MODEL_NAME="qwen2.5:0.5b")
    assert "model_name           = 'qwen2.5:0.5b' (str)" in out.stdout


def test_bad_value_in_env_file_stops_the_script(run_config):
    out = run_config("MAX_RESULTS=three\n")
    assert out.returncode != 0
    assert "ValueError: MAX_RESULTS must be a whole number, got 'three'" in out.stderr


def test_env_file_is_found_from_another_directory(run_config, tmp_path):
    (tmp_path / ".env").write_text("MODEL_NAME=llama3.2:3b\n")
    out = subprocess.run([sys.executable, str(tmp_path / "config.py")], cwd=tmp_path.parent,
                         env={k: v for k, v in os.environ.items() if k not in KEYS},
                         capture_output=True, text=True)
    assert "'llama3.2:3b'" in out.stdout


# ── DEBUG in the app ───────────────────────────────────────────────────────
# app and logger are set up on import, so this needs its own process

APP_CHECK = """
import logging, sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(message)s")
from fastapi.testclient import TestClient
import my_rag_api as api

@api.app.get("/boom")
def boom():
    raise RuntimeError("something broke")

r = TestClient(api.app, raise_server_exceptions=False).get("/boom")
print("debug", api.app.debug, logging.getLogger("uvicorn.error").level)
print("status", r.status_code)
print("body", r.text.replace(chr(10), " "))
"""


@pytest.mark.parametrize("debug", ["true", "false"])
def test_debug_in_the_app(tmp_path, debug):
    for name in ("config.py", "my_rag_api.py"):
        shutil.copy(HERE / name, tmp_path / name)
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    env.update(DEBUG=debug, CHROMA_PATH=str(tmp_path / "db"))
    out = subprocess.run([sys.executable, "-c", APP_CHECK], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120).stdout

    assert "status 500" in out
    if debug == "true":
        assert "debug True 10" in out                        # logging.DEBUG
        assert "Loaded Settings(" in out and "debug                = True (bool)" in out
        assert "RuntimeError: something broke" in out       # traceback in the response
    else:
        assert "debug False 0" in out
        assert "Loaded Settings(" not in out
        assert "body Internal Server Error" in out
        assert "something broke" not in out.split("body", 1)[1]


# ── .env hygiene ───────────────────────────────────────────────────────────

def lines(name):
    return [line.strip() for line in (HERE / name).read_text().splitlines()]


@pytest.mark.parametrize("name", [".gitignore", ".dockerignore"])
def test_env_file_is_ignored(name):
    assert ".env" in lines(name)


def test_env_example_is_not_ignored():
    # .env.example is the one that gets committed
    for name in (".gitignore", ".dockerignore"):
        assert not {".env*", ".env.*", "*.example"} & set(lines(name))


def test_env_example_documents_every_setting():
    keys = {line.split("=", 1)[0] for line in lines(".env.example") if line and not line.startswith("#")}
    assert keys == set(KEYS)


def test_env_example_values_are_the_defaults(env):
    # copying .env.example to .env has to give exactly what you get without one
    from dotenv import dotenv_values
    assert vars(env(**dotenv_values(HERE / ".env.example"))) == DEFAULTS
