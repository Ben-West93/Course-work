"""
L7 — Expanded Semantic Search App
=================================
Run with (from the repo root):
    .venv/bin/streamlit run module_07/semantic_search_app/semantic_search_app.py

A ChromaDB-backed Streamlit search tool over the docs/ folder with source
filtering, result counts, expandable full text, relevance badges, a sidebar
source-count stat, and a "Similar to this" button on every result.
"""

import streamlit as st
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Paths are anchored to this file so the app works no matter which directory
# `streamlit run` is launched from.
APP_DIR    = Path(__file__).parent
DOCS_DIR   = APP_DIR / "docs"
DB_PATH    = APP_DIR / "chroma_db"
CHUNK_SIZE = 500
OVERLAP    = 100
N_RESULTS  = 5
PREVIEW_CHARS = 150

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")

# ── Model & ChromaDB ─────────────────────────────────────────────────────────
# @st.cache_resource runs this function once per server process and hands the
# same model/collection objects to every re-run and every browser session.
# Without it, the ~80 MB model would reload on every widget interaction.
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(name="docs_search")
    return model, collection

model, collection = load_resources()

# ── Index documents ───────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Fixed-size chunking with overlap."""
    text = text.strip()
    if not text:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start:start + size])
        if start + size >= len(text):  # last window already reached the end
            break
    return chunks

def index_docs() -> tuple[int, int]:
    """Read docs/ folder and upsert all chunks into ChromaDB.

    Returns (chunk_count, file_count).
    """
    ids, documents, metadatas = [], [], []
    files = sorted(p for p in DOCS_DIR.glob("*") if p.suffix in (".txt", ".md"))
    for file in files:
        for i, chunk in enumerate(chunk_text(file.read_text(encoding="utf-8"))):
            ids.append(f"{file.name}_chunk_{i}")
            documents.append(chunk)
            metadatas.append({"source": file.name})

    if ids:
        embeddings = model.encode(documents).tolist()
        # upsert() overwrites existing IDs instead of raising like add() would,
        # so re-indexing the same files never produces duplicate-ID errors.
        collection.upsert(ids=ids, documents=documents,
                          embeddings=embeddings, metadatas=metadatas)

    # Drop chunks that no longer exist (a file was shortened or deleted),
    # otherwise stale text would keep showing up in results.
    stale = set(collection.get(include=[])["ids"]) - set(ids)
    if stale:
        collection.delete(ids=list(stale))

    return len(ids), len(files)

# Index automatically on first launch (empty collection); afterwards the
# persistent DB is reused and the sidebar button re-indexes on demand.
if collection.count() == 0:
    with st.spinner("Indexing documents..."):
        index_docs()

# ── Session state ─────────────────────────────────────────────────────────────
# Streamlit re-runs the whole script on every interaction, so plain variables
# reset. The search box is bound to st.session_state["query"] via key="query",
# which lets the "Similar to this" button overwrite the query between runs.
if "query" not in st.session_state:
    st.session_state["query"] = ""

def search_similar(text: str):
    # Runs as an on_click callback, i.e. *before* the next re-run, which is the
    # only point where a widget's session_state value may be changed. Streamlit
    # then re-runs the script automatically with the new query in place.
    st.session_state["query"] = text

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

if st.sidebar.button("🔄 Re-index Documents"):
    with st.spinner("Re-indexing..."):
        n_chunks, n_files = index_docs()
    st.sidebar.success(f"Indexed {n_chunks} chunks from {n_files} files")

all_metadatas = collection.get(include=["metadatas"])["metadatas"] or []
all_sources = sorted({m["source"] for m in all_metadatas if m and "source" in m})

selected_sources = st.sidebar.multiselect(
    "Search only in these files",
    options=all_sources,
    placeholder="All files",
    help="Leave empty to search every file.",
)

st.sidebar.divider()
st.sidebar.metric("Source files indexed", len(all_sources))
st.sidebar.metric("Chunks in collection", collection.count())

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🔍 Semantic Search")
st.write("Search your course documents by meaning, not just keywords.")

query = st.text_input("Search your documents", key="query",
                      placeholder="Type a question or phrase...")

total = collection.count()

if total == 0:
    st.info(f"No documents indexed. Add .txt or .md files to `{DOCS_DIR}` "
            "and click 'Re-index Documents'.")
elif query.strip():
    query_kwargs = {
        "query_embeddings": model.encode([query]).tolist(),
        "n_results": min(N_RESULTS, total),
        "include": ["documents", "metadatas", "distances"],
    }
    # ChromaDB `$in` filter: keep only chunks whose "source" metadata is one of
    # the listed filenames. Only add `where` for a real subset — an empty
    # selection means "all files", and `{"$in": []}` would raise an error.
    if selected_sources and set(selected_sources) != set(all_sources):
        query_kwargs["where"] = {"source": {"$in": selected_sources}}

    results = collection.query(**query_kwargs)

    # Results are nested one level per query; guard against empty outer lists.
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    st.caption(f"Showing {len(documents)} of {total} total documents")

    if not documents:
        st.warning("No matching results. Try a different query or widen the source filter.")

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        if dist < 0.5:
            badge = "🟢 High"
        elif dist < 1.0:
            badge = "🟡 Medium"
        else:
            badge = "🔴 Low"

        source = (meta or {}).get("source", "unknown")
        with st.container(border=True):
            col_title, col_score = st.columns([3, 1])
            with col_title:
                st.subheader(f"{i + 1}. {source}")
            with col_score:
                st.markdown(f"**{badge}** · distance `{dist:.3f}`")

            preview = doc[:PREVIEW_CHARS] + ("..." if len(doc) > PREVIEW_CHARS else "")
            st.write(preview)

            with st.expander("Show full text"):
                st.write(doc)

            st.button("🔁 Similar to this", key=f"similar_{i}",
                      on_click=search_similar, args=(doc,))
