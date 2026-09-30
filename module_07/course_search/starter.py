"""
L7 — Course Content Search  (STARTER)
======================================
Run with:
    python starter.py

Your goal: build an interactive semantic search tool over course-summary sentences.

Required features:
    1. A list of at least 10 sentences summarising things learned in this course
    2. Embed all sentences once using SentenceTransformer('all-MiniLM-L6-v2')
    3. A search loop that:
       - Prompts the user for a query with input()
       - Embeds the query
       - Computes cosine similarity between the query and every sentence
       - Displays the top 3 matches with their scores
       - Exits when the user types "quit"

Key concepts:
    Encoding sentences:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)          # shape: (N, 384)
        query_emb  = model.encode([query])            # shape: (1, 384)

    Cosine similarity:
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_emb, embeddings)[0]  # shape: (N,)

    Top-N indices:
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:3]   # highest first
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Knowledge base ──────────────────────────────────────────────────────────
# TODO: Create a list called `sentences` with at least 10 short summaries of
#       things you've learned in this course. Examples:
#       "HTML uses tags to define the structure of a web page"
#       "FastAPI automatically validates request data using Pydantic models"
#       "Streamlit re-runs the entire script when any widget is interacted with"
#       Add at least 5 more from modules you've completed.

sentences = [
    # TODO: add your sentences here
]

# ── Model ───────────────────────────────────────────────────────────────────
print("Loading model...")
# TODO: Load SentenceTransformer('all-MiniLM-L6-v2') into a variable called `model`

# ── Encode knowledge base ───────────────────────────────────────────────────
print(f"Embedding {len(sentences)} sentences...")
# TODO: Encode all sentences and store in `doc_embeddings`

# ── Search loop ─────────────────────────────────────────────────────────────
print("\nSemantic Search — type 'quit' to exit\n")

while True:
    query = input("Search: ").strip()

    if query.lower() == "quit":
        break

    if not query:
        continue

    # TODO: Encode the query into `query_embedding`

    # TODO: Compute cosine similarity between query_embedding and doc_embeddings
    #       Store the 1-D scores array in `scores`

    # TODO: Get the indices of the top 3 results (highest score first)
    #       Store in `top_indices`

    print("\nTop 3 results:")
    # TODO: Loop over top_indices and print each sentence with its score
    #       Format: "  1. [0.6823] FastAPI automatically validates request data..."
    print()
