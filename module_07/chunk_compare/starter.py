"""
L7 — Chunk and Compare
======================
Run with:
    python chunk_compare.py

Compares two chunking strategies (fixed-size sliding window vs. paragraph)
for semantic search over a ~600-word document covering four course topics.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Document (500-800 words, 4 topics) ───────────────────────────────────────
DOCUMENT = """Python is a general-purpose programming language known for its readable syntax and enormous ecosystem. Developers use it for scripting, data analysis, web services, and machine learning. A virtual environment isolates each project's dependencies so that one project's packages never conflict with another's. Tools such as pip install libraries into that environment, while pytest makes it easy to write automated tests that catch bugs early. Functions, classes, and modules let programmers organise code into small reusable pieces, and type hints document what each function expects and returns. Reading error messages carefully and using a debugger are everyday habits that turn confusing failures into quick fixes, and version control with Git records every change so that experiments can be undone safely.

Web APIs let programs talk to each other over HTTP. A client sends a request containing a method such as GET or POST, a URL, and optional headers or a JSON body. The server replies with a status code and a response body, usually JSON. Status codes like 200 mean success, 404 means the resource was not found, and 500 signals a server error. Frameworks such as FastAPI and Flask make it simple to define routes and validate incoming data. When calling an external API, good clients handle timeouts, check the status code, and never hard-code secret keys in source files.

Embeddings turn text into lists of numbers called vectors. A model such as all-MiniLM-L6-v2 reads a sentence and produces a vector with 384 dimensions, placing sentences with similar meanings close together in that space. This is what allows semantic search to find a sentence about cars when the query mentions automobiles, even though the words differ. Closeness is measured with cosine similarity, which compares the angle between two vectors and returns a score near one for very similar text and near zero for unrelated text. Because embeddings capture meaning rather than exact keywords, they power recommendations, clustering, and question answering.

Vector databases store embeddings and search them quickly. Instead of scanning every vector, they build indexes that find approximate nearest neighbours in milliseconds, even across millions of records. Each record usually keeps its vector alongside the original text and metadata, such as a title or a source file. Before storing a long document, developers split it into chunks, because a single embedding of a whole book blurs many ideas together. Chunk size matters: chunks that are too small lose surrounding context, while chunks that are too large dilute the specific fact a user wants. Overlap between neighbouring chunks helps preserve ideas that would otherwise be cut in half.

Retrieval brings these pieces together. To answer a question, the system embeds the query, asks the vector database for the closest chunks, and returns them ranked by similarity score. In a retrieval augmented generation pipeline, those chunks are then handed to a language model as context so that its answer is grounded in real documents. The quality of the final answer depends heavily on retrieval quality, which in turn depends on how the text was chunked, which embedding model was used, and how many results are returned. Testing several strategies on realistic queries is the best way to choose between them."""

# ── Chunking strategies ──────────────────────────────────────────────────────
def fixed_size_chunks(text, size=300, overlap=50):
    """Sliding-window character chunks of `size` chars, sharing `overlap` chars."""
    step = size - overlap
    if size <= 0 or overlap < 0 or step <= 0:
        raise ValueError("size must be > 0 and 0 <= overlap < size (step must be > 0)")
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):   # window already reached the end
            break
    return chunks


def paragraph_chunks(text):
    """Split on blank lines; normalise line endings, strip, drop empties."""
    text = text.replace("\r\n", "\n")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


# ── Model & search ───────────────────────────────────────────────────────────
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def search_chunks(query, chunks, embeddings, n=2):
    """Return the top-n (score, chunk) pairs, best first."""
    if not chunks:
        return []
    query_embedding = model.encode([query])                      # 2-D: (1, 384)
    scores = cosine_similarity(query_embedding, embeddings).flatten()  # 1-D
    top = np.argsort(scores)[::-1][:n]
    return [(float(scores[i]), chunks[i]) for i in top]


def show(label, results):
    print(f"\n  {label}")
    for score, chunk in results:
        text = chunk.replace("\n", " ")
        print(f"    [{score:.2f}] ({len(chunk)} chars) {text}")


# ── Run experiment ───────────────────────────────────────────────────────────
fixed = fixed_size_chunks(DOCUMENT)
paras = paragraph_chunks(DOCUMENT)
fixed_emb = model.encode(fixed)
para_emb = model.encode(paras)

print(f"Document: {len(DOCUMENT.split())} words | "
      f"{len(fixed)} fixed chunks | {len(paras)} paragraph chunks")

queries = [
    "How do I isolate Python project dependencies?",
    "What does an HTTP 404 status code mean?",
    "Why should long documents be split before storing them in a vector database?",
]

for query in queries:
    print("\n" + "=" * 60)
    print(f'Query: "{query}"')
    print("=" * 60)
    show("Fixed-size chunks (300 chars, 50 overlap):",
         search_chunks(query, fixed, fixed_emb, n=2))
    show("Paragraph chunks:", search_chunks(query, paras, para_emb, n=2))

# ── Analysis ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)
print("""
1. Which strategy returned more focused results?
   It depends on the query. Fixed-size chunks were more focused for the
   narrow fact questions (dependencies, HTTP 404): ~300 characters isolate the
   answer, while a paragraph carries it plus unrelated sentences, diluting its
   embedding. But for the broader "why split documents" question the
   paragraph scored higher (0.81 vs 0.60) because the full explanation
   spanned several sentences that one small chunk could not hold.

2. Which strategy missed context by cutting mid-sentence?
   Fixed-size chunks. The window cuts at a character count, so a chunk can
   begin or end in the middle of a word or sentence, and a needed fact can be
   split across two chunks. Overlap softens this but does not remove it.
   Paragraph chunks always keep complete sentences and complete ideas.

3. When would you choose each strategy?
   Paragraph chunks: well-structured text (articles, docs, notes) where each
   paragraph holds one idea, and where the reader needs full context.
   Fixed-size chunks: unstructured or uniformly long text (transcripts, logs,
   scraped pages) or when the embedding model has a token limit and you need
   predictable, even chunk sizes; use overlap to reduce mid-sentence cuts.
""")
