"""
L1 — Running Local LLMs with Ollama: Explorer  (STARTER)
==========================================================
Run with:
    python ollama_explorer.py

Prerequisites: Ollama running (`ollama serve`) with a model pulled.
               Adjust MODEL below to match what you have downloaded.

Your goal: run 4 experiments that show how system prompts, context,
           question length, and temperature affect LLM responses.

Experiments:
    1. Same question, 3 different system prompts
    2. RAG-style context grounding (answerable vs. unanswerable question)
    3. Response timing across short / medium / long questions
    4. (Bonus) Temperature comparison: 0.1 vs 1.0

Key concepts:
    Ollama /api/chat endpoint:
        POST http://localhost:11434/api/chat
        Body: {"model": MODEL, "messages": [...], "stream": False}
        Response: {"message": {"content": "..."}, ...}

    Message format:
        {"role": "system", "content": "..."}   — optional system prompt
        {"role": "user",   "content": "..."}   — user question

    Measuring time:
        import time
        start = time.time()
        # ... call generate() ...
        elapsed = time.time() - start
"""

import requests
import time

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"  # Change to match your downloaded model


# ── Core helper ────────────────────────────────────────────────────────────

def generate(messages: list[dict], temperature: float = 0.7) -> tuple[str, float]:
    """
    Call Ollama /api/chat and return (response_text, elapsed_seconds).

    TODO:
        1. POST to f"{OLLAMA_URL}/api/chat" with:
               {"model": MODEL, "messages": messages,
                "stream": False, "options": {"temperature": temperature}}
        2. Measure elapsed time with time.time()
        3. Extract and return the response text from:
               response.json()["message"]["content"]
        4. Handle requests.exceptions.ConnectionError — return an error
           string and 0.0 so experiments keep running
    """
    pass  # TODO


# ── Experiment 1: System prompt comparison ─────────────────────────────────

def experiment_1():
    """Same question, three different system prompts."""
    question = "What is an API?"

    # TODO: Define 3 system prompts:
    #   - None (empty string — send only the user message, no system role)
    #   - "Explain like I'm 5 years old."
    #   - "You are a senior software architect. Be technical and precise."
    prompts = []  # TODO: list of (label, system_prompt) tuples

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Same question, different system prompts")
    print("=" * 60)

    for label, system_prompt in prompts:
        # TODO: Build messages list (include system message only if system_prompt is non-empty)
        # TODO: Call generate(), print label + response + time
        pass  # TODO


# ── Experiment 2: RAG-style context grounding ──────────────────────────────

def experiment_2():
    """Provide context; test answerable vs. unanswerable question."""

    # TODO: Write a short paragraph (3–5 sentences) about any topic you choose
    context = ""  # TODO

    # TODO: Write one question that CAN be answered from the context
    answerable_question = ""  # TODO

    # TODO: Write one question that CANNOT be answered from the context
    unanswerable_question = ""  # TODO

    system_prompt = (
        "Answer ONLY from the provided context. "
        "If the context does not contain the answer, say exactly: "
        "'I don't have enough information to answer that.'"
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: RAG-style context grounding")
    print("=" * 60)

    for label, question in [
        ("Answerable", answerable_question),
        ("Unanswerable", unanswerable_question),
    ]:
        # TODO: Build messages with system_prompt + context embedded, then question
        # TODO: Call generate(), print label + question + response
        pass  # TODO


# ── Experiment 3: Response timing ──────────────────────────────────────────

def experiment_3():
    """Ask questions of varying length, compare response times."""

    # TODO: Define 3 questions — short (~5 words), medium (~20 words), long (~50 words)
    questions = []  # TODO: list of (label, question) tuples

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Response timing")
    print("=" * 60)

    for label, question in questions:
        # TODO: Call generate() (no system prompt needed), print label + time
        pass  # TODO


# ── Experiment 4 (bonus): Temperature comparison ───────────────────────────

def experiment_4():
    """Same prompt, temperature 0.1 vs 1.0."""
    question = "Tell me a one-sentence fact about the ocean."

    print("\n" + "=" * 60)
    print("EXPERIMENT 4 (Bonus): Temperature comparison")
    print("=" * 60)

    for temp in [0.1, 1.0]:
        # TODO: Call generate() with the given temperature, print temp + response
        pass  # TODO


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
