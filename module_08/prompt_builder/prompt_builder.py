"""
L2 — The RAG Architecture: Prompt Builder
==========================================
Run with:
    python prompt_builder.py

Builds the prompt assembly step of a RAG pipeline: system prompt + retrieved
chunks (labelled by source) + the user's question, then prints the prompt and
a rough token estimate (chars // 4).
"""

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the context "
    "provided below.\n"
    "Rules:\n"
    "1. Answer ONLY from the provided context. Do not use outside knowledge.\n"
    '2. If the answer is not in the context, say exactly: "I don\'t have enough '
    'information to answer that."\n'
    "3. Cite the source document filename for every fact you use, "
    "e.g. (Source: refund_policy.txt).\n"
)

SEPARATOR = "\n\n---\n\n"


def format_chunks(chunks: list[dict]) -> str:
    # .get() so one malformed chunk doesn't crash the whole prompt
    parts = [
        f"[Source: {c.get('source', 'unknown')}]\n{c.get('text', '').strip()}"
        for c in chunks
    ]
    return SEPARATOR.join(parts)


def token_estimate(text: str) -> int:
    return len(text) // 4


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = format_chunks(chunks)
    if not context:
        context = "(no documents retrieved)"
    return (
        f"{SYSTEM_PROMPT}\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION: {question.strip()}"
    )


def run_test(question: str, chunks: list[dict], label: str = "") -> None:
    prompt = build_prompt(question, chunks)
    context_tokens = token_estimate(format_chunks(chunks))

    print("=" * 70)
    print(f"TEST: {label or question}")
    print("=" * 70)
    print(prompt)
    print("-" * 70)
    print(f"Chunks:          {len(chunks)}")
    print(f"Characters:      {len(prompt)}")
    print(f"Context tokens:  ~{context_tokens}")
    print(f"Total tokens:    ~{token_estimate(prompt)}")
    print()


if __name__ == "__main__":
    q1 = "How long do customers have to request a refund?"
    chunks1 = [
        {
            "text": "Customers may request a full refund within 30 days of purchase. "
                    "Refunds are issued to the original payment method within 5-7 "
                    "business days.",
            "source": "refund_policy.txt",
            "distance": 0.21,
        },
        {
            "text": "Digital products are non-refundable once the download link has "
                    "been accessed, unless the file is corrupted.",
            "source": "refund_policy.txt",
            "distance": 0.38,
        },
        {
            "text": "Support tickets are answered within 24 hours on weekdays.",
            "source": "support_faq.md",
            "distance": 0.61,
        },
    ]
    run_test(q1, chunks1, "Case 1 - refund window (3 chunks, 2 sources)")

    q2 = "What embedding model does the search tool use and how big are the vectors?"
    chunks2 = [
        {
            "text": "The search tool embeds every chunk with the all-MiniLM-L6-v2 "
                    "sentence-transformers model.",
            "source": "architecture.md",
            "distance": 0.18,
        },
        {
            "text": "all-MiniLM-L6-v2 produces 384-dimensional vectors, which are "
                    "stored in a ChromaDB collection using cosine distance.",
            "source": "embeddings_notes.txt",
            "distance": 0.27,
        },
    ]
    run_test(q2, chunks2, "Case 2 - embedding model (2 chunks, 2 sources)")

    # edge cases
    run_test("What is the capital of France?", [], "Edge - empty chunk list")

    run_test(
        "Who wrote the onboarding guide?",
        [{"text": "The onboarding guide was written by the platform team.", "distance": 0.4}],
        "Edge - chunk missing 'source' key",
    )

    run_test(
        "   ",
        [{"text": "Café opening hours: 8am–6pm. 日本語のメニューあり ☕", "source": "café_info.txt", "distance": 0.3}],
        "Edge - blank question + multibyte unicode",
    )

    # scaling check: context size vs token estimate
    print("=" * 70)
    print("TOKEN SCALING (large context)")
    print("=" * 70)
    base = "Retrieval-augmented generation grounds answers in documents. "
    for n in (1, 10, 100, 1000):
        big = [{"text": base * n, "source": f"big_{n}.txt", "distance": 0.5}]
        prompt = build_prompt("Summarise the document.", big)
        print(f"repeat={n:>5}  chars={len(prompt):>7}  tokens~{token_estimate(prompt):>6}")
