"""
Module 7 Project — Semantic Search Tool
========================================
app.py — Streamlit search interface

Run with:
    streamlit run app.py

Make sure you've indexed documents first:
    python ingest.py
(or use the "Re-index documents" button in the sidebar)
"""

import streamlit as st
from search import MEDIUM_RELEVANCE, get_collection_stats, is_searchable, search
from ingest import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, ingest

RELEVANCE_COLORS = {"High": "green", "Medium": "orange", "Low": "gray"}
MESSAGE_STYLES = {"success": st.success, "warning": st.warning, "error": st.error}
MIN_CHUNK_SIZE = 50
MAX_CHUNK_SIZE = 2000

st.set_page_config(page_title="Semantic Search Tool", page_icon="🔎", layout="wide")


def reindex(chunk_size: int, overlap: int) -> None:
    """Rebuild the index with the chosen settings and remember the outcome."""
    try:
        with st.spinner(f"Re-indexing with chunk size {chunk_size}, overlap {overlap}…"):
            summary = ingest(chunk_size=chunk_size, overlap=overlap, verbose=False)
        text = f"Indexed {summary['documents']} documents into {summary['chunks']} chunks."
        if summary["skipped"]:
            text += f" Skipped (not valid UTF-8): {', '.join(summary['skipped'])}."
        st.session_state.reindex_message = (
            "warning" if summary["skipped"] else "success", text
        )
    except (ValueError, FileNotFoundError) as error:
        st.session_state.reindex_message = ("error", f"Re-index failed: {error}")


def clamp(value: int, low: int, high: int) -> int:
    """Keep value inside [low, high] so widget defaults are always valid."""
    return max(low, min(high, value))


def render_sidebar(stats: dict) -> dict:
    """
    Draw the sidebar (collection stats, search controls, re-index) and
    return the current search settings.
    """
    with st.sidebar:
        st.header("Collection")
        col1, col2 = st.columns(2)
        col1.metric("Documents", stats["unique_sources"])
        col2.metric("Chunks", stats["total_chunks"])
        if stats["chunk_size"]:
            st.caption(f"Current index: chunk size {stats['chunk_size']}, overlap {stats['overlap']}")

        st.header("Search settings")
        n_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
        min_score = st.slider(
            "Minimum similarity score",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            help="Hide results scoring below this value. 0 shows everything.",
        )
        sources = st.multiselect(
            "Filter by source file",
            options=stats["source_names"],
            placeholder="All documents",
        )

        st.header("Index")
        # The index may have been built from the command line with settings
        # outside the sidebar's range, so clamp the defaults before using them.
        chunk_size = st.number_input(
            "Chunk size (characters)", min_value=MIN_CHUNK_SIZE, max_value=MAX_CHUNK_SIZE,
            value=clamp(stats["chunk_size"] or DEFAULT_CHUNK_SIZE, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE),
            step=50,
        )
        # The overlap's limits must not depend on the chunk size: Streamlit
        # treats a widget with new limits as a new widget and resets its value,
        # so changing the chunk size would throw away the overlap you typed.
        # The overlap < chunk size rule is checked below instead.
        current_overlap = stats["overlap"] if stats["overlap"] is not None else DEFAULT_OVERLAP
        overlap = st.number_input(
            "Overlap (characters)", min_value=0, max_value=MAX_CHUNK_SIZE - 1,
            value=clamp(current_overlap, 0, MAX_CHUNK_SIZE - 1),
            step=10,
        )
        settings_valid = overlap < chunk_size
        if not settings_valid:
            st.error("Overlap must be smaller than the chunk size.")
        if st.button("Re-index documents", use_container_width=True, disabled=not settings_valid):
            reindex(int(chunk_size), int(overlap))
            st.rerun()

        message = st.session_state.pop("reindex_message", None)
        if message:
            kind, text = message
            MESSAGE_STYLES[kind](text)

    return {
        "n_results": n_results,
        # Score = 1 - distance, so a minimum score is a maximum distance.
        "distance_threshold": 1 - min_score if min_score > 0 else None,
        "sources": sources or None,
    }


def render_result(rank: int, result: dict) -> None:
    """Draw a single search result card."""
    with st.container(border=True):
        color = RELEVANCE_COLORS[result["relevance"]]
        st.markdown(
            f"**#{rank} · {result['source']}** · chunk {result['chunk_index']} "
            f"&nbsp; :{color}-badge[{result['relevance']} relevance]"
        )
        st.progress(
            max(0.0, min(1.0, result["score"])),
            text=f"Similarity {result['score']:.3f} · distance {result['distance']:.3f}",
        )
        st.write(result["text"])


def main() -> None:
    """Render the full search page."""
    stats = get_collection_stats()
    settings = render_sidebar(stats)

    st.title("🔎 Semantic Search Tool")
    st.caption(
        "Ask a question in plain English. Results are ranked by meaning, "
        "not keyword overlap."
    )

    if stats["total_chunks"] == 0:
        st.warning(
            "The collection is empty. Click **Re-index documents** in the sidebar "
            "(or run `python ingest.py`) to build the index."
        )
        return

    query = st.text_input(
        "Search", placeholder="e.g. How do I keep state between reruns in Streamlit?"
    )

    if not query.strip():
        st.info("Type a question above to search the documents.")
        return
    if not is_searchable(query):
        st.warning("Please include some words. A query of only symbols can't be searched.")
        return

    with st.spinner("Searching…"):
        results = search(query, **settings)

    if not results:
        st.warning(
            "No results matched. Try lowering the minimum similarity score, "
            "clearing the source filter, or rephrasing the question."
        )
        return

    if results[0]["score"] < MEDIUM_RELEVANCE:
        scope = (
            "The selected source files may not cover this topic, so try clearing the filter."
            if settings["sources"]
            else "The documents may not cover this topic."
        )
        st.warning(
            f"No strong matches (best score {results[0]['score']:.3f}). "
            f"{scope} The closest results are shown below."
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Results", len(results))
    m2.metric("Best score", f"{results[0]['score']:.3f}")
    m3.metric("Average score", f"{sum(r['score'] for r in results) / len(results):.3f}")

    for rank, result in enumerate(results, 1):
        render_result(rank, result)


main()
