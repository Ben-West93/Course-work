"""
Backend settings
================
Run with:
    python config.py        # prints current settings

Every setting is read from an environment variable and falls back to a
default when it isn't set. The project's .env is loaded first, but anything
already set in the shell (or by docker-compose) wins over it.

Usage pattern in other modules:
    from config import settings

    client = chromadb.PersistentClient(path=settings.chroma_path)
    MODEL = settings.model_name
"""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

# load_dotenv doesn't override by default, so .env only fills in variables that
# aren't set yet and `MODEL_NAME=llama3.2:3b python config.py` still beats the
# file. .env lives at the project root next to docker-compose.yml, one folder
# up from this file, so a local `uvicorn main:app` reads the same one compose
# does. A missing .env is fine, the defaults below are used. Inside the image
# there's no .env at all (parent of /app is /), compose passes the values in
# as real env vars instead
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

NAMES = ("OLLAMA_URL", "MODEL_NAME", "CHROMA_PATH", "MAX_RESULTS", "CONFIDENCE_THRESHOLD", "DEBUG")
TRUE_VALUES = {"true", "1", "yes", "on"}
FALSE_VALUES = {"false", "0", "no", "off"}


def env_str(name: str, default: str) -> str:
    # os.environ.get only falls back when the variable is missing altogether.
    # `MODEL_NAME=` in .env sets it to "", which is never usable, so a blank
    # value counts as not set
    value = os.environ.get(name, "").strip()
    return value or default


# env vars are always strings, so anything else has to be converted. int() and
# float() raise a ValueError that doesn't say which variable was wrong, so the
# helpers below re-raise it with the name. Better to stop at startup than to
# hit a bad value on the first request

def env_int(name: str, default: str, low: int, high: int) -> int:
    raw = env_str(name, default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from None
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")
    return value


def env_float(name: str, default: str, low: float, high: float) -> float:
    raw = env_str(name, default)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    # float() happily parses "nan" and "inf". Every comparison with nan is
    # False, so a nan cutoff would drop every chunk without an error. The range
    # check catches both since neither is between low and high
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {raw!r}")
    return value


def env_bool(name: str, default: str) -> bool:
    raw = env_str(name, default)
    # the usual `.lower() == "true"` turns DEBUG=1, DEBUG=yes or a typo like
    # DEBUG=ture into False without saying anything
    if raw.lower() in TRUE_VALUES:
        return True
    if raw.lower() in FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def warn_case_mismatches() -> None:
    # env var names are case sensitive, so model_name=... is a separate variable
    # and would be ignored without a word. Only names that match one of ours
    # apart from case are flagged: .env also holds names meant for compose and
    # ollama, so an unknown name on its own isn't a mistake
    for key in os.environ:
        if key not in NAMES and key.upper() in NAMES:
            warnings.warn(f"{key} is set, but config.py reads {key.upper()} (names are case sensitive), "
                          f"so it's ignored", stacklevel=3)


class Settings:
    """
    Central configuration class. All values read from environment variables.
    Import as: from config import settings
    """

    def __init__(self):
        # a trailing slash would turn every request into //api/chat
        self.ollama_url = env_str("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model_name = env_str("MODEL_NAME", "llama3.2:1b")
        self.chroma_path = env_str("CHROMA_PATH", "./rag_db")
        # these two are the defaults for n_results and max_distance on /ask,
        # so they get the same limits the endpoint puts on them
        self.max_results = env_int("MAX_RESULTS", "3", 1, 20)
        self.confidence_threshold = env_float("CONFIDENCE_THRESHOLD", "1.0", 0.0, 4.0)
        self.debug = env_bool("DEBUG", "false")

        # without the scheme requests fails on every call, and /health would
        # only ever say "disconnected"
        if not self.ollama_url.startswith(("http://", "https://")):
            raise ValueError(f"OLLAMA_URL must start with http:// or https://, got {self.ollama_url!r}")
        warn_case_mismatches()

    def __repr__(self) -> str:
        """Print all settings for inspection."""
        # the type next to each value shows the conversion worked, 3 (int)
        # rather than '3' (str)
        width = max(len(name) for name in vars(self))
        lines = [f"  {name:<{width}} = {value!r} ({type(value).__name__})"
                 for name, value in vars(self).items()]
        return "Settings(\n" + "\n".join(lines) + "\n)"


# Module-level singleton — import this everywhere
# Python only runs a module once and caches it, so every `from config import
# settings` gets this same object and the env is only read at startup
settings = Settings()


if __name__ == "__main__":
    print(settings)
