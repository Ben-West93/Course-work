"""
Module 7 Project — Semantic Search Tool
========================================
ingest.py — document loading, chunking, and ChromaDB storage

Run with:
    python ingest.py
    python ingest.py --chunk-size 200 --overlap 50

Every run rebuilds the collection from scratch, so the index only ever holds
chunks produced by a single chunk size / overlap setting. That keeps chunking
experiments clean: results are never a mix of two configurations.
"""

import argparse
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
# Paths are resolved relative to this file so the scripts work no matter which
# directory they are launched from.
BASE_DIR        = Path(__file__).resolve().parent
DOCS_DIR        = BASE_DIR / "docs"
CHROMA_PATH     = BASE_DIR / "chroma_data"
COLLECTION_NAME = "semantic_search"
MODEL_NAME      = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP    = 100
SUPPORTED_EXTENSIONS = {".txt", ".md"}

# Cosine distance means score = 1 - distance is the cosine similarity.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the sentence-transformer model once and reuse it."""
    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts with the shared model.

    Embeddings are L2-normalised so cosine distance behaves consistently.
    """
    vectors = get_model().encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.

    Strategy: a sliding character window of `chunk_size` that advances by
    `chunk_size - overlap`. To avoid cutting words in half, a chunk's end is
    pulled back to the last whitespace inside the window, and the next chunk's
    start is pushed forward to the next word boundary. A single word longer
    than the window is split at the hard character limit.

    Args:
        text:       Full document text.
        chunk_size: Maximum characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.

    Raises:
        ValueError: if chunk_size < 1, overlap < 0, or overlap >= chunk_size.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = " ".join(text.split())  # collapse newlines / repeated whitespace
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Back up to the last space so the chunk ends on a whole word.
            space = text.rfind(" ", start, end + 1)
            if space > start:
                end = space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break

        # Step forward, keeping `overlap` characters of shared context.
        next_start = max(end - overlap, start + 1)
        if text[next_start - 1] != " " and text[next_start] != " ":
            # Landed mid-word: skip to the space after it (at most the space
            # at `end`) so the next chunk starts cleanly. If there is none,
            # this is a hard split inside one long word, so keep the position.
            space = text.find(" ", next_start, end + 1)
            if space != -1:
                next_start = space + 1
        while next_start < length and text[next_start] == " ":
            next_start += 1
        start = next_start

    return chunks


def load_documents(docs_dir: Path, skipped: list[str] = None) -> list[dict]:
    """
    Read all .txt and .md files from docs_dir.

    Files are returned in alphabetical order; empty files are skipped.
    Files that are not valid UTF-8 are skipped too, so one bad file cannot
    block the whole index.

    Args:
        docs_dir: Folder to read.
        skipped:  Optional list; names of unreadable files are appended to it.

    Returns:
        List of dicts: {"filename": str, "text": str}

    Raises:
        FileNotFoundError: if docs_dir does not exist.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

    documents = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            if skipped is not None:
                skipped.append(path.name)
            continue
        if text:
            documents.append({"filename": path.name, "text": text})
    return documents


def get_collection(chroma_path: Path, collection_name: str):
    """Create (or retrieve) a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        name=collection_name, metadata=COLLECTION_METADATA
    )


def reset_collection(chroma_path: Path, collection_name: str):
    """Delete the collection if it exists and return a fresh, empty one."""
    client = chromadb.PersistentClient(path=str(chroma_path))
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
    return client.create_collection(
        name=collection_name, metadata=COLLECTION_METADATA
    )


def ingest(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    docs_dir: Path = DOCS_DIR,
    chroma_path: Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    verbose: bool = True,
) -> dict:
    """
    Full ingestion pipeline: load → chunk → embed → upsert.

    Each chunk is stored with metadata: source filename, chunk index,
    and the chunk size used — so experiments with different sizes can
    be compared without ambiguity.

    Everything that can fail (loading, chunking, embedding) happens before the
    old collection is deleted, so a failed run leaves the previous index intact.

    Returns:
        {"documents": int, "chunks": int, "chunk_size": int, "overlap": int,
         "skipped": list[str]}  # unreadable files that were left out

    Raises:
        FileNotFoundError: if docs_dir does not exist.
        ValueError:        if no readable documents are found, or the chunk
                           settings are invalid.
    """
    skipped = []
    documents = load_documents(docs_dir, skipped=skipped)
    if not documents:
        raise ValueError(
            f"No readable .txt or .md documents found in {docs_dir}. "
            "The existing index was left unchanged."
        )

    ids, texts, metadatas = [], [], []
    for doc in documents:
        for index, chunk in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            ids.append(f"{doc['filename']}::{index}")
            texts.append(chunk)
            metadatas.append({
                "source": doc["filename"],
                "chunk_index": index,
                "chunk_size": chunk_size,
                "overlap": overlap,
            })

    embeddings = embed(texts)
    collection = reset_collection(chroma_path, collection_name)
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    summary = {
        "documents": len(documents),
        "chunks": len(texts),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "skipped": skipped,
    }
    if verbose:
        print(
            f"Indexed {summary['documents']} documents into {summary['chunks']} chunks "
            f"(chunk_size={chunk_size}, overlap={overlap}) → {chroma_path}"
        )
        if skipped:
            print(f"Skipped (not valid UTF-8): {', '.join(skipped)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index docs/ into ChromaDB")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap",    type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()
    try:
        ingest(chunk_size=args.chunk_size, overlap=args.overlap)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))  # prints usage + message, exits with code 2
