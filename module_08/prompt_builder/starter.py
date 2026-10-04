"""
L2 — The RAG Architecture: Prompt Builder  (STARTER)
======================================================
Run with:
    python prompt_builder.py

Your goal: build the prompt assembly step of a RAG pipeline.

Required features:
    1. A SYSTEM_PROMPT string that instructs the model to:
       - Answer ONLY from the provided context
       - Say "I don't have enough information" when the answer isn't in the context
       - Cite sources by name
    2. A format_chunks(chunks) function that formats a list of chunk dicts into
       a single context string — each chunk labelled with its source
    3. A build_prompt(question, chunks) function that returns the complete
       assembled prompt as a string (system prompt + context + question)
    4. A token_estimate(text) helper that returns characters // 4
    5. A run_test(question, chunks) function that assembles and prints the
       full prompt + token count
    6. At least 2 test cases with different questions and chunk sets

Key concepts:
    Chunk dict format:
        {"text": "...", "source": "filename.txt", "distance": 0.42}

    Prompt assembly pattern:
        [System prompt]
        CONTEXT:
        [Source: doc1.txt]
        <chunk text>

        ---

        [Source: doc2.txt]
        <chunk text>
        USER QUESTION: <question>

    Token estimate:
        A rough approximation: len(text) // 4
        (GPT-4 uses ~4 chars per token on average)
"""

# ── System prompt ──────────────────────────────────────────────────────────
# TODO: Write a SYSTEM_PROMPT string that instructs the model to:
#   - Answer ONLY from the context provided
#   - Say "I don't have enough information" if the answer isn't present
#   - Cite sources by mentioning the document name
SYSTEM_PROMPT = ""  # TODO


# ── Helpers ────────────────────────────────────────────────────────────────

def format_chunks(chunks: list[dict]) -> str:
    """Format a list of chunk dicts into a context string."""
    # TODO: For each chunk, produce:
    #   [Source: <chunk["source"]>]
    #   <chunk["text"]>
    # Separate chunks with "\n\n---\n\n"
    pass  # TODO


def token_estimate(text: str) -> int:
    """Rough token count estimate: characters // 4."""
    # TODO: return len(text) // 4
    pass  # TODO


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Assemble the full RAG prompt."""
    # TODO: Combine SYSTEM_PROMPT, "CONTEXT:\n" + format_chunks(chunks),
    #   and "USER QUESTION: " + question into one string
    pass  # TODO


# ── Test runner ────────────────────────────────────────────────────────────

def run_test(question: str, chunks: list[dict]) -> None:
    """Assemble and print a prompt with token estimate."""
    # TODO: Call build_prompt(), print the result, print the token estimate
    pass  # TODO


# ── Test cases ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test case 1
    # TODO: Define a question and at least 2 simulated chunks, call run_test()

    # Test case 2
    # TODO: Define a different question and different chunks, call run_test()
    pass
