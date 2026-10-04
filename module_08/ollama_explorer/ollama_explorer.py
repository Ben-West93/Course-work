"""
L1 — Running Local LLMs with Ollama: Explorer
==============================================
Run with:
    python ollama_explorer.py

Prerequisites: Ollama running (`ollama serve`) with a model pulled
               (`ollama pull llama3.2:1b`). Change MODEL to match.
"""

import requests
import time

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"  # Change to match your downloaded model
TIMEOUT = 120  # first call has to load the model into memory


# ── Core helper ────────────────────────────────────────────────────────────

def build_messages(question: str, system_prompt: str = "") -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages


def generate(messages: list[dict], temperature: float = 0.7) -> tuple[str, float]:
    """Call Ollama /api/chat and return (response_text, elapsed_seconds)."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    start = time.time()
    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=TIMEOUT)
        elapsed = time.time() - start
        data = response.json()
    except requests.exceptions.ConnectionError:
        return f"[error] could not connect to Ollama at {OLLAMA_URL} - is `ollama serve` running?", 0.0
    except requests.exceptions.Timeout:
        return f"[error] request timed out after {TIMEOUT}s", 0.0
    except ValueError:
        return f"[error] non-JSON response (HTTP {response.status_code})", 0.0

    # a missing model comes back as 404 with {"error": "..."} instead of a message
    if not response.ok or "message" not in data:
        return f"[error] HTTP {response.status_code}: {data.get('error', data)}", 0.0

    return data["message"].get("content", "").strip(), elapsed


def word_count(text: str) -> int:
    return 0 if text.startswith("[error]") else len(text.split())


def print_response(label: str, text: str, elapsed: float):
    print(f"\n--- {label} ({elapsed:.2f}s) ---")
    print(text)


# ── Experiment 1: System prompt comparison ─────────────────────────────────

def experiment_1():
    """Same question, three different system prompts."""
    question = "What is an API?"

    prompts = [
        ("No system prompt", ""),
        ("ELI5", "Explain like I'm 5 years old."),
        ("Senior architect", "You are a senior software architect. Be technical and precise."),
    ]

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Same question, different system prompts")
    print("=" * 60)

    results = []
    for label, system_prompt in prompts:
        text, elapsed = generate(build_messages(question, system_prompt))
        print_response(label, text, elapsed)
        results.append((label, word_count(text), elapsed))

    print("\nSummary:")
    for label, words, elapsed in results:
        print(f"  {label:<18} {words:>4} words  {elapsed:6.2f}s")

    # Observed: the system prompt changes tone and depth much more than length.
    # No prompt gave a generic bulleted overview (~380 words). ELI5 dropped the
    # jargon for a toy-box analogy and was the shortest/fastest. The architect
    # prompt brought in REST/SOAP/gRPC and HTTP verbs, but still opened almost
    # word for word like the default answer.


# ── Experiment 2: RAG-style context grounding ──────────────────────────────

def experiment_2():
    """Provide context; test answerable vs. unanswerable question."""

    context = (
        "The Golden Gate Bridge opened to traffic on May 27, 1937. "
        "It spans the Golden Gate strait, connecting San Francisco to Marin County. "
        "The main span is 4,200 feet long, and it was the longest suspension bridge "
        "main span in the world until 1964. "
        "Its distinctive color is called International Orange."
    )

    answerable_question = "What year did the Golden Gate Bridge open?"
    unanswerable_question = "How much did it cost to build the Golden Gate Bridge?"

    system_prompt = (
        "Answer ONLY from the provided context. "
        "If the context does not contain the answer, say exactly: "
        "'I don't have enough information to answer that.'"
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: RAG-style context grounding")
    print("=" * 60)
    print(f"Context: {context}")

    refusal = "I don't have enough information to answer that."
    for label, question in [
        ("Answerable", answerable_question),
        ("Unanswerable", unanswerable_question),
    ]:
        user_msg = f"Context:\n{context}\n\nQuestion: {question}"
        text, elapsed = generate(build_messages(user_msg, system_prompt))
        print_response(f"{label}: {question}", text, elapsed)
        if not text.startswith("[error]"):
            refused = refusal.lower().rstrip(".") in text.lower()
            print(f"  refused: {refused}")

    # Observed: llama3.2:1b stayed grounded here. It answered "May 27, 1937"
    # straight from the context and used the exact refusal sentence for the
    # cost question, then added a short explanation after it, so checking for
    # an exact string match would fail but a "contains" check works. Both calls
    # were way faster than experiment 1 because the answers are so short.


# ── Experiment 3: Response timing ──────────────────────────────────────────

def experiment_3():
    """Ask questions of varying length, compare response times."""

    questions = [
        ("Short", "What is a linked list?"),
        ("Medium", "Can you explain the difference between a linked list and an "
                   "array, and when I would pick one over the other?"),
        ("Long", "I'm building a music player app where users constantly add, remove, "
                 "and reorder songs in a playlist while it is playing. I've been storing "
                 "the playlist as a Python list but I'm worried about performance as "
                 "playlists grow to thousands of songs. Should I switch to a linked list, "
                 "and what tradeoffs should I think about?"),
    ]

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Response timing")
    print("=" * 60)

    results = []
    for label, question in questions:
        text, elapsed = generate(build_messages(question))
        in_words, out_words = len(question.split()), word_count(text)
        results.append((label, in_words, out_words, elapsed))
        print(f"\n{label} ({in_words} words in): {elapsed:.2f}s, {out_words} words out")

    print("\nSummary:")
    print(f"  {'Length':<8} {'In':>4} {'Out':>5} {'Time':>8} {'Words/s':>8}")
    for label, in_words, out_words, elapsed in results:
        rate = out_words / elapsed if elapsed else 0
        print(f"  {label:<8} {in_words:>4} {out_words:>5} {elapsed:>7.2f}s {rate:>8.1f}")

    # Observed: time tracks how much the model writes, not how much you send.
    # Medium and long both produced ~560 words, and long was only ~1s slower
    # despite 2.5x the input. Throughput stayed around 100-150 words/s, so
    # prompt processing is cheap next to generation on a 1B model.


# ── Experiment 4 (bonus): Temperature comparison ───────────────────────────

def experiment_4():
    """Same prompt, temperature 0.1 vs 1.0."""
    question = "Tell me a one-sentence fact about the ocean."

    print("\n" + "=" * 60)
    print("EXPERIMENT 4 (Bonus): Temperature comparison")
    print("=" * 60)

    # three runs each so the difference in variety is actually visible
    for temp in [0.1, 1.0]:
        print(f"\n--- temperature={temp} ---")
        for i in range(3):
            text, elapsed = generate(build_messages(question), temperature=temp)
            print(f"  run {i + 1} ({elapsed:.2f}s): {text}")

    # Observed: at 0.1 the runs were near-identical (two matched exactly, the
    # third only changed the ending). At 1.0 every run went somewhere different
    # (Great Barrier Reef, square km, species counts). Low temp for factual or
    # RAG answers, higher when you want variety.


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
