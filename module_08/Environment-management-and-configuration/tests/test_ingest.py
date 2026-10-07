import my_rag_api as api
from conftest import BrokenCollection


def ingest(client):
    r = client.post("/ingest")
    assert r.status_code == 200
    return r.json()


# ── the real docs/ folder ──────────────────────────────────────────────────

def test_ingest_real_docs(client, empty_collection):
    body = ingest(client)
    assert body == {
        "chunks_ingested": 32,
        "files_ingested": 7,
        "chunks_removed": 0,
        "message": "Ingested 32 chunks from 7 files",
    }
    assert empty_collection.count() == 32


def test_ingest_twice_is_idempotent(client, empty_collection):
    first = ingest(client)
    second = ingest(client)
    assert second == first
    assert empty_collection.count() == 32


def test_ingested_chunks_have_stable_ids_and_metadata(client, empty_collection):
    ingest(client)
    stored = empty_collection.get(include=["documents", "metadatas"])
    for chunk_id, doc, meta in zip(stored["ids"], stored["documents"], stored["metadatas"], strict=True):
        assert chunk_id == f"{meta['source']}_{meta['chunk_index']}"
        assert doc.strip()


def test_ingest_cleans_up_an_old_37_chunk_db(client, empty_collection):
    # before headings were merged, docs/ came out as 37 chunks
    empty_collection.add(ids=[f"old_{i}" for i in range(5)] + ["chromadb.md_0"],
                         documents=["stale"] * 5 + ["# ChromaDB"])
    body = ingest(client)
    assert body["chunks_removed"] == 5
    assert body["message"] == "Ingested 32 chunks from 7 files, removed 5 old chunks"
    assert empty_collection.count() == 32
    assert empty_collection.get(ids=["chromadb.md_0"])["documents"][0] != "# ChromaDB"


# ── files changing between ingests ─────────────────────────────────────────

def test_shortened_file_drops_its_extra_chunks(client, empty_collection, docs_dir):
    notes = docs_dir / "notes.md"
    notes.write_text("one\n\ntwo\n\nthree")
    assert ingest(client)["chunks_ingested"] == 3

    notes.write_text("one")
    body = ingest(client)
    assert (body["chunks_ingested"], body["chunks_removed"]) == (1, 2)
    assert body["message"] == "Ingested 1 chunk from 1 file, removed 2 old chunks"
    assert empty_collection.get()["ids"] == ["notes.md_0"]


def test_deleted_file_drops_its_chunks(client, empty_collection, docs_dir):
    (docs_dir / "a.md").write_text("alpha")
    (docs_dir / "b.md").write_text("bravo\n\ncharlie")
    ingest(client)

    (docs_dir / "b.md").unlink()
    body = ingest(client)
    assert body["chunks_removed"] == 2
    assert body["message"] == "Ingested 1 chunk from 1 file, removed 2 old chunks"
    assert sorted(empty_collection.get()["ids"]) == ["a.md_0"]


# ── empty or missing docs ──────────────────────────────────────────────────

def test_empty_folder_changes_nothing(client, empty_collection, docs_dir):
    empty_collection.add(ids=["keep"], documents=["already here"])
    body = ingest(client)
    assert body["chunks_ingested"] == body["files_ingested"] == body["chunks_removed"] == 0
    assert body["message"].endswith("nothing changed")
    assert empty_collection.count() == 1


def test_missing_folder_changes_nothing(client, empty_collection, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DOCS_DIR", str(tmp_path / "does_not_exist"))
    empty_collection.add(ids=["keep"], documents=["already here"])
    assert ingest(client)["chunks_ingested"] == 0
    assert empty_collection.count() == 1


def test_only_empty_files_changes_nothing(client, empty_collection, docs_dir):
    (docs_dir / "empty.md").write_text("")
    (docs_dir / "blank.txt").write_text("  \n\n \t\n")
    (docs_dir / "heading_only.md").write_text("# Title\n\n## Subtitle\n")
    assert ingest(client)["chunks_ingested"] == 0


# ── which files and how they're split ──────────────────────────────────────

def test_only_txt_and_md_files_are_read(client, empty_collection, docs_dir):
    (docs_dir / "a.txt").write_text("text file")
    (docs_dir / "b.md").write_text("markdown file")
    (docs_dir / "C.MD").write_text("upper case extension")
    for name in ["notes.pdf", "script.py", "data.json", "README", "a.txt.bak"]:
        (docs_dir / name).write_text("should be skipped")
    (docs_dir / "folder.md").mkdir()

    body = ingest(client)
    assert body["files_ingested"] == 3
    assert sorted(m["source"] for m in empty_collection.get()["metadatas"]) == ["C.MD", "a.txt", "b.md"]


def test_windows_line_endings_split_into_paragraphs(docs_dir):
    (docs_dir / "crlf.txt").write_bytes(b"first para\r\nsame para\r\n\r\nsecond para")
    texts = [c["text"] for c in api.load_documents(str(docs_dir))]
    assert texts == ["first para\nsame para", "second para"]


def test_invalid_utf8_doesnt_crash(client, empty_collection, docs_dir):
    (docs_dir / "latin1.txt").write_bytes("caf\xe9 ol\xe9".encode("latin-1"))
    assert ingest(client)["chunks_ingested"] == 1
    assert "�" in empty_collection.get()["documents"][0]


def test_headings_are_merged_into_the_next_paragraph(docs_dir):
    (docs_dir / "doc.md").write_text(
        "# Title\n\n## Section\n\nFirst paragraph.\n\nSecond paragraph.\n\n## Trailing heading\n"
    )
    texts = [c["text"] for c in api.load_documents(str(docs_dir))]
    assert texts == ["# Title\n\n## Section\n\nFirst paragraph.", "Second paragraph."]


def test_heading_and_text_on_adjacent_lines_stay_together(docs_dir):
    (docs_dir / "doc.md").write_text("# Title\nBody straight under it.")
    assert [c["text"] for c in api.load_documents(str(docs_dir))] == ["# Title\nBody straight under it."]


def test_ingest_ignores_a_request_body(client, empty_collection, docs_dir):
    (docs_dir / "a.md").write_text("alpha")
    r = client.post("/ingest", json={"anything": True})
    assert r.status_code == 200
    assert r.json()["chunks_ingested"] == 1


def test_message_uses_singular_for_one(client, empty_collection, docs_dir):
    (docs_dir / "a.md").write_text("alpha\n\nbravo")
    ingest(client)
    (docs_dir / "a.md").write_text("alpha")
    assert ingest(client)["message"] == "Ingested 1 chunk from 1 file, removed 1 old chunk"


# ── ChromaDB failures ──────────────────────────────────────────────────────

def test_chroma_write_failure_is_503_and_logged(client, monkeypatch, docs_dir, caplog):
    (docs_dir / "a.md").write_text("alpha")
    monkeypatch.setattr(api, "collection", BrokenCollection("disk is full"))
    r = client.post("/ingest")
    assert r.status_code == 503
    assert r.json() == {"detail": "Could not write to ChromaDB"}
    assert "disk is full" in caplog.text


class FailingDelete:
    def __init__(self, collection):
        self.collection = collection

    def __getattr__(self, name):
        return getattr(self.collection, name)

    def delete(self, **kwargs):
        raise RuntimeError("database is locked")


def test_retry_after_a_failed_cleanup_finishes_the_job(client, empty_collection, docs_dir, monkeypatch):
    notes = docs_dir / "notes.md"
    notes.write_text("one\n\ntwo\n\nthree")
    ingest(client)
    notes.write_text("one")

    # the upsert goes through, deleting the 2 stale chunks doesn't
    monkeypatch.setattr(api, "collection", FailingDelete(empty_collection))
    assert client.post("/ingest").status_code == 503
    assert empty_collection.count() == 3

    monkeypatch.setattr(api, "collection", empty_collection)
    body = ingest(client)
    assert (body["chunks_ingested"], body["chunks_removed"]) == (1, 2)
    assert empty_collection.get()["ids"] == ["notes.md_0"]
