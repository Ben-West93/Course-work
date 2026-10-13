"""
Backend settings, read from environment variables.

    python config.py      # prints what was loaded, secret hidden

Outside Docker the defaults use SQLite and a ./chroma folder next to this
file and expect Ollama on localhost:11434, so `uvicorn main:app --reload`
runs with no setup. docker-compose.yml sets the Postgres, Ollama and Chroma
values for the container.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def env_str(name: str, default: str) -> str:
    # `MODEL_NAME=` in .env sets the variable to "", which is never usable
    return os.environ.get(name, "").strip() or default


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
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {raw!r}")
    return value


def env_bool(name: str, default: str) -> bool:
    raw = env_str(name, default).lower()
    if raw in ("true", "1", "yes", "on"):
        return True
    if raw in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def local_path(path: str) -> str:
    # relative paths are taken from backend/, so uvicorn and pytest find the
    # same files wherever they're started from
    return os.path.normpath(os.path.join(BASE_DIR, os.path.expanduser(path)))


# HMAC only: the API signs and checks its own tokens, so there's no public key
# to hand out. The value is the shortest secret each one allows, its hash size
JWT_KEY_BYTES = {"HS256": 32, "HS384": 48, "HS512": 64}


def normalise_db_url(url: str) -> str:
    # hosting providers hand out postgres:// URLs, which SQLAlchemy rejects,
    # and plain postgresql:// leaves the driver choice to SQLAlchemy. Only
    # psycopg 3 is installed
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


class Settings:
    """All backend configuration. Import the `settings` instance, not this class."""

    def __init__(self):
        self.database_url = normalise_db_url(env_str("DATABASE_URL", f"sqlite:///{local_path('jobfit.db')}"))
        self.chroma_path = local_path(env_str("CHROMA_PATH", "./chroma"))
        self.ollama_url = env_str("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        # 3b over 1b: on the check-in questions (with the keyword pass) 1b got 3
        # of 9 fully right and made things up, 3b 6 of 9 with nothing wrong
        # (CHECK_IN.md, "After the check-in"). 0.8 to 3.2s an answer on CPU
        self.model_name = env_str("MODEL_NAME", "llama3.2:3b")
        # blank means a random secret per process (see auth.py), so tokens stop
        # working on restart. Set it in .env to keep logins across restarts
        self.jwt_secret_key = env_str("JWT_SECRET_KEY", "")
        algorithm = env_str("JWT_ALGORITHM", "HS256")
        self.jwt_algorithm = algorithm.upper()
        self.jwt_expire_minutes = env_int("JWT_EXPIRE_MINUTES", "30", 1, 1440)
        # chunks further than this from the question are dropped before the
        # prompt. Squared L2 on unit vectors, 0 (same) to 4 (opposite). On the
        # samples, off-topic questions were 1.56 and up, the hardest real ones 1.41
        self.max_distance = env_float("MAX_DISTANCE", "1.45", 0.0, 4.0)
        # Ollama on CPU can take a while, especially the first call after a load
        self.ollama_timeout = env_int("OLLAMA_TIMEOUT", "120", 5, 600)
        self.debug = env_bool("DEBUG", "false")

        if not self.ollama_url.startswith(("http://", "https://")):
            raise ValueError(f"OLLAMA_URL must start with http:// or https://, got {self.ollama_url!r}")
        if self.jwt_algorithm not in JWT_KEY_BYTES:
            raise ValueError(f"JWT_ALGORITHM must be one of {', '.join(JWT_KEY_BYTES)}, got {algorithm!r}")
        # PyJWT warns when the key is shorter than the hash, and a short secret can be brute-forced
        min_bytes = JWT_KEY_BYTES[self.jwt_algorithm]
        if self.jwt_secret_key and len(self.jwt_secret_key.encode()) < min_bytes:
            raise ValueError(f"JWT_SECRET_KEY must be at least {min_bytes} bytes for {self.jwt_algorithm}. "
                             f"Generate one with python -c 'import secrets; print(secrets.token_urlsafe({min_bytes}))'")
        if not self.database_url.startswith(("sqlite", "postgresql")):
            raise ValueError(f"DATABASE_URL must be a sqlite:// or postgresql+psycopg:// URL, got {self.database_url!r}")

    def __repr__(self) -> str:
        shown = dict(vars(self), jwt_secret_key="<set>" if self.jwt_secret_key else "<random per start>")
        # the Postgres password sits in the URL
        if "@" in self.database_url:
            scheme, rest = self.database_url.split("://", 1)
            shown["database_url"] = f"{scheme}://***@{rest.rsplit('@', 1)[1]}"
        return "Settings(" + ", ".join(f"{k}={v!r}" for k, v in shown.items()) + ")"


settings = Settings()


if __name__ == "__main__":
    print(settings)
