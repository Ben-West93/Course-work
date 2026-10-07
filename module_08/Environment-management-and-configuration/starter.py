"""
L6 — Environment Management & Configuration  (STARTER)
========================================================
Run with:
    python config.py        # prints current settings

Your goal: create a centralized Settings class that reads all
configuration from environment variables, with sensible defaults.

Required settings (at minimum):
    OLLAMA_URL             str   default: "http://localhost:11434"
    MODEL_NAME             str   default: "llama3.2:1b"
    CHROMA_PATH            str   default: "./rag_db"
    MAX_RESULTS            int   default: 3
    CONFIDENCE_THRESHOLD   float default: 1.0
    DEBUG                  bool  default: False

Key concepts:
    os.environ.get():
        import os
        value = os.environ.get("KEY", "default")

    Type coercion for non-string env vars:
        int(os.environ.get("MAX_RESULTS", "3"))
        float(os.environ.get("CONFIDENCE_THRESHOLD", "1.0"))
        os.environ.get("DEBUG", "false").lower() == "true"

    .env file pattern (plain text — no special library needed):
        Read with python-dotenv if installed, or load manually.
        For Docker Compose: add `env_file: .env` to each service.

    Best practices:
        - Never commit .env (add to .gitignore and .dockerignore)
        - Commit .env.example with placeholder / default values
        - Fail loudly for required secrets (raise ValueError if missing)

Usage pattern in other modules:
    from config import settings

    client = chromadb.PersistentClient(path=settings.chroma_path)
    MODEL = settings.model_name
"""

import os


class Settings:
    """
    Central configuration class. All values read from environment variables.
    Import as: from config import settings
    """

    # TODO: Add each setting as a property or set them in __init__
    # Example for ollama_url:
    #   self.ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    def __init__(self):
        # TODO: self.ollama_url   = ...
        # TODO: self.model_name   = ...
        # TODO: self.chroma_path  = ...
        # TODO: self.max_results  = int(...)
        # TODO: self.confidence_threshold = float(...)
        # TODO: self.debug        = os.environ.get("DEBUG", "false").lower() == "true"
        pass  # TODO

    def __repr__(self) -> str:
        """Print all settings for inspection."""
        # TODO: return a multi-line string showing each setting and its value
        pass  # TODO


# Module-level singleton — import this everywhere
settings = Settings()


if __name__ == "__main__":
    # TODO: print(settings) to verify the values are loaded correctly
    pass  # TODO
