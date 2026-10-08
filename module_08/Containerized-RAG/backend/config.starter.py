"""
Module 8 Project — Containerized RAG Assistant  (STARTER)
Backend: Settings
==================
Reads all configuration from environment variables.
No hardcoded URLs, paths, or model names — everything comes from the environment
so the same code runs locally and inside Docker without changes.

Your task: implement the Settings class below.

Required environment variables (see .env.example):
    OLLAMA_URL            URL of the Ollama service          default: http://localhost:11434
    MODEL_NAME            Ollama model to use                default: llama3.2:1b
    CHROMA_PATH           Path for ChromaDB persistent data  default: ./rag_db
    MAX_RESULTS           Number of chunks to retrieve       default: 3
    CONFIDENCE_THRESHOLD  Maximum distance for filtering     default: 1.0
    DEBUG                 Enable verbose logging             default: false
"""

import os


class Settings:
    def __init__(self):
        # TODO: Read each environment variable with os.environ.get()
        # Provide sensible defaults so the app works without a .env file.
        #
        # Example:
        #   self.ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        #
        # Remember: os.environ.get() always returns a string.
        # Cast numeric values:  int(os.environ.get("MAX_RESULTS", "3"))
        # Cast floats:          float(os.environ.get("CONFIDENCE_THRESHOLD", "1.0"))
        # Cast bools:           os.environ.get("DEBUG", "false").lower() == "true"

        # TODO: self.ollama_url = ...
        # TODO: self.model_name = ...
        # TODO: self.chroma_path = ...
        # TODO: self.max_results = ...
        # TODO: self.confidence_threshold = ...
        # TODO: self.debug = ...
        pass


# Module-level singleton — import and use this everywhere
settings = Settings()
