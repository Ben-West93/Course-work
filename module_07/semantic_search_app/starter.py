"""
L7 — Expanded Semantic Search App  (STARTER)
=============================================
Run with:
    streamlit run starter.py

Your goal: extend a ChromaDB-backed Streamlit search tool with source
filtering, result counts, and expandable full text.

Required features:
    1. Load and chunk .txt / .md files from a docs/ folder
    2. Embed and store chunks in a ChromaDB collection (persistent)
    3. Sidebar multiselect to filter by source filename
    4. Search bar that queries ChromaDB (with optional where filter)
    5. "Showing X of Y total documents" count above results
    6. Each result: title (filename), preview (first 150 chars),
       st.expander showing full chunk text

Bonus (pick at least one):
    - A "Similar to this" button that re-queries using that chunk's text
    - Relevance badge (green/yellow/red) based on distance thresholds
    - Sidebar stat: total unique source files indexed

Key concepts:
    ChromaDB where filter for source files:
        where={"source": {"$in": selected_sources}}   # filter to selection
        # omit `where` entirely when no filter is active

    Reading all files from a folder:
        from pathlib import Path
        docs_dir = Path("docs")
        for file in docs_dir.glob("*.md"):
            text = file.read_text()
            # chunk and add to collection with metadata={"source": file.name}

    Counting total docs in collection:
        collection.count()
"""

import streamlit as st
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

DOCS_DIR   = Path("docs")
CHUNK_SIZE = 500
OVERLAP    = 100
N_RESULTS  = 5

# ── Page config ──────────────────────────────────────────────────────────────
# TODO: st.set_page_config(page_title, page_icon, layout="wide")

# ── Model & ChromaDB ─────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # TODO: Load SentenceTransformer('all-MiniLM-L6-v2')
    # TODO: Create a PersistentClient with path="./chroma_db"
    # TODO: Get or create collection "docs_search"
    # TODO: Return model and collection
    pass

model, collection = load_resources()

# ── Index documents ───────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Fixed-size chunking with overlap."""
    # TODO: implement sliding-window chunking
    return []

def index_docs():
    """Read docs/ folder and upsert all chunks into ChromaDB."""
    # TODO: Iterate over .txt and .md files in DOCS_DIR
    # TODO: Chunk each file's text
    # TODO: Embed chunks
    # TODO: Upsert with IDs like "filename_chunk_0", "filename_chunk_1", ...
    #       and metadata={"source": filename}
    pass

# TODO: Call index_docs() once (wrap in a check so it's not re-indexed every run)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

# TODO: Get the list of all unique source filenames from the collection
#       Hint: collection.get(include=["metadatas"])["metadatas"]
# TODO: st.sidebar.multiselect to let users pick which sources to search
#       Store selection in `selected_sources`

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("Semantic Search")

query = st.text_input("Search your documents", placeholder="Type a question or phrase...")

if query:
    # TODO: Build the `where` filter dict — only add it when selected_sources
    #       is non-empty and not equal to all sources

    # TODO: Query the collection
    #       results = collection.query(query_texts=[...], n_results=N_RESULTS,
    #                                  where=..., include=["documents","metadatas","distances"])

    total = collection.count()
    # TODO: Show "Showing X of Y total documents"

    # TODO: Iterate over results and for each chunk display:
    #   - st.subheader with the source filename and distance
    #   - A 150-character preview
    #   - st.expander("Show full text") containing the full chunk
