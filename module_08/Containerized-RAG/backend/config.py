"""
Backend settings, read from environment variables.

    python config.py      # prints what was loaded

Nothing here reads .env directly. docker-compose passes it to the backend
container through env_file, and outside Docker the defaults below point at
localhost, so `uvicorn main:app --reload` works without any setup.
"""

import os


def env_str(name: str, default: str) -> str:
    # `MODEL_NAME=` in .env sets the variable to "", which is never a usable
    # value, so blank counts as not set
    return os.environ.get(name, "").strip() or default


# os.environ only holds strings. int()/float() errors don't say which variable
# was wrong, so these re-raise with the name. Failing at startup beats failing
# on the first request

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
    # float() accepts "nan" and "inf", the range check rules both out
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {raw!r}")
    return value


def env_bool(name: str, default: str) -> bool:
    raw = env_str(name, default).lower()
    # plain `== "true"` would turn DEBUG=1 or a typo like DEBUG=ture into False
    if raw in ("true", "1", "yes", "on"):
        return True
    if raw in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


class Settings:
    """All backend configuration. Import the `settings` instance, not this class."""

    def __init__(self):
        # a trailing slash would turn every request into //api/chat
        self.ollama_url = env_str("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model_name = env_str("MODEL_NAME", "llama3.2:1b")
        self.chroma_path = env_str("CHROMA_PATH", "./rag_db")
        # default n_results on /ask, same limits as the endpoint
        self.max_results = env_int("MAX_RESULTS", "3", 1, 20)
        # top-chunk distance above this makes the answer "low" confidence
        self.confidence_threshold = env_float("CONFIDENCE_THRESHOLD", "1.0", 0.0, 4.0)
        # default max_distance on /ask: chunks further than this are dropped.
        # Between the two is the borderline band, used as context but flagged low
        self.max_distance = env_float("MAX_DISTANCE", "1.2", 0.0, 4.0)
        self.debug = env_bool("DEBUG", "false")

        if not self.ollama_url.startswith(("http://", "https://")):
            raise ValueError(f"OLLAMA_URL must start with http:// or https://, got {self.ollama_url!r}")

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"Settings({fields})"


# modules are only executed once, so every `from config import settings`
# shares this object and the environment is read a single time at startup
settings = Settings()


if __name__ == "__main__":
    print(settings)
