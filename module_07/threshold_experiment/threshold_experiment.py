"""
L7 — Similarity Threshold Experiment
====================================
Run with:
    python threshold_experiment.py

Shows how different similarity thresholds (0.3, 0.5, 0.7) change which
knowledge-base sentences are returned for the same query.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Knowledge base ───────────────────────────────────────────────────────────
knowledge_base = [
    # Python / development
    "Deploy FastAPI with uvicorn and Docker",
    "Python web frameworks like Django and Flask make building websites easier",
    "Set up a virtual environment to isolate your Python dependencies",
    "Write automated tests with pytest to catch bugs early",
    "Git branches let developers work on features without breaking the main code",

    # Cooking
    "Simmer the tomato sauce slowly with garlic and fresh basil",
    "Knead bread dough for ten minutes so the gluten develops properly",
    "Season a cast iron skillet with oil to keep it non-stick",
    "Roast vegetables at high heat to caramelize their natural sugars",
    "Let steak rest after grilling so the juices redistribute",

    # Space exploration
    "The James Webb Space Telescope observes distant galaxies in infrared light",
    "Astronauts on the International Space Station orbit Earth every 90 minutes",
    "Mars rovers search for signs of ancient water on the red planet",
    "Reusable rockets have dramatically lowered the cost of launching satellites",
    "A black hole's gravity is so strong that not even light can escape",

    # Music
    "Learning piano scales builds finger strength and musical fluency",
    "A guitar chord progression of G, C, and D is common in folk songs",
    "Jazz musicians improvise solos over complex harmonies",
    "Tuning your violin before every practice keeps the pitch accurate",
    "A metronome helps drummers keep a steady tempo",
]

# Topic of each knowledge-base sentence (5 per topic, in order above),
# used only to report on-topic sentences that a stricter threshold drops
# (i.e. they pass the loosest threshold, 0.3, but not the current one).
TOPICS = ["python"] * 5 + ["cooking"] * 5 + ["space"] * 5 + ["music"] * 5

# ── Test queries ─────────────────────────────────────────────────────────────
# (query, expected relevant topic or None for ambiguous)
test_queries = [
    ("How do I deploy my FastAPI Python app with Docker?", "python"),
    ("How long should I let a grilled steak rest?", "cooking"),
    ("What does the James Webb telescope observe in space?", "space"),
    ("How do I practice scales on the piano?", "music"),
    ("How do I keep things running smoothly?", None),   # ambiguous
    ("Practice makes perfect", None),                   # ambiguous
]

# ── Thresholds ───────────────────────────────────────────────────────────────
THRESHOLDS = [0.3, 0.5, 0.7]

# ── Model & embeddings ───────────────────────────────────────────────────────
print("Loading model and embedding knowledge base...")
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(knowledge_base)

# ── Run experiment ───────────────────────────────────────────────────────────
for query, expected_topic in test_queries:
    print(f'\nQuery: "{query}"')

    query_embedding = model.encode([query])
    # cosine_similarity returns shape (1, n_docs); flatten to 1-D
    scores = cosine_similarity(query_embedding, doc_embeddings).flatten()

    for threshold in THRESHOLDS:
        passing = [
            (float(score), sentence)
            for score, sentence in zip(scores, knowledge_base)
            if score >= threshold
        ]
        passing.sort(reverse=True)

        noun = "result" if len(passing) == 1 else "results"
        print(f"  Threshold {threshold}: {len(passing)} {noun}")
        for score, sentence in passing:
            print(f"    [{score:.2f}] {sentence}")

        if expected_topic is not None:
            passed = {s for _, s in passing}
            missed = [
                (float(score), sentence)
                for score, sentence, topic in zip(scores, knowledge_base, TOPICS)
                if topic == expected_topic and sentence not in passed
                and score >= THRESHOLDS[0]
            ]
            missed.sort(reverse=True)
            if missed:
                print(f"    Missed {len(missed)} relevant on-topic '{expected_topic}' sentence(s):")
                for score, sentence in missed:
                    print(f"      ({score:.2f}) {sentence}")
