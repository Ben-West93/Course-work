"""
Module 7 Project — Semantic Search Tool
========================================
test_search_tool.py — unit tests for chunking, loading, search and evaluation

Run with:
    python -m unittest test_search_tool.py -v

The search tests build a throwaway index in a temporary folder, so they never
touch the real chroma_data/ index.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

import ingest
import search
from evaluate import precision_recall


class ChunkTextTests(unittest.TestCase):
    """chunk_text: sizes, overlap, word boundaries and bad input."""

    TEXT = " ".join(f"word{i}" for i in range(200))

    def test_chunks_respect_max_size(self):
        for chunk in ingest.chunk_text(self.TEXT, chunk_size=50, overlap=10):
            self.assertLessEqual(len(chunk), 50)

    def test_chunks_do_not_split_words(self):
        words = set(self.TEXT.split())
        for chunk in ingest.chunk_text(self.TEXT, chunk_size=50, overlap=10):
            self.assertTrue(set(chunk.split()) <= words, chunk)

    def test_mixed_word_lengths_do_not_split_words(self):
        # Regression: a chunk ending exactly on a space used to make the
        # next chunk start mid-word ("a the" → ["a", "th", "e"]).
        self.assertEqual(ingest.chunk_text("a the", 3, 0), ["a", "the"])
        text = "a the database xxxx bb of SQL joins é 日本語"
        words = set(text.split())
        for size in range(max(len(w) for w in words), 40):
            for overlap in range(size):
                for chunk in ingest.chunk_text(text, size, overlap):
                    self.assertTrue(set(chunk.split()) <= words, (size, overlap, chunk))

    def test_all_words_are_covered(self):
        chunks = ingest.chunk_text(self.TEXT, chunk_size=50, overlap=10)
        covered = set(" ".join(chunks).split())
        self.assertEqual(covered, set(self.TEXT.split()))

    def test_overlap_repeats_context(self):
        chunks = ingest.chunk_text(self.TEXT, chunk_size=60, overlap=20)
        for first, second in zip(chunks, chunks[1:]):
            # Consecutive chunks share at least one whole word.
            self.assertTrue(set(first.split()) & set(second.split()))
        no_overlap = ingest.chunk_text(self.TEXT, chunk_size=60, overlap=0)
        for first, second in zip(no_overlap, no_overlap[1:]):
            self.assertFalse(set(first.split()) & set(second.split()))

    def test_smaller_chunks_make_more_chunks(self):
        small = ingest.chunk_text(self.TEXT, chunk_size=100, overlap=20)
        large = ingest.chunk_text(self.TEXT, chunk_size=400, overlap=80)
        self.assertGreater(len(small), len(large))

    def test_short_text_is_single_chunk(self):
        self.assertEqual(ingest.chunk_text("  hello   world \n", 100, 10), ["hello world"])

    def test_empty_text_gives_no_chunks(self):
        self.assertEqual(ingest.chunk_text("   \n ", 100, 10), [])

    def test_long_word_is_hard_split(self):
        chunks = ingest.chunk_text("x" * 25, chunk_size=10, overlap=0)
        self.assertEqual(chunks, ["x" * 10, "x" * 10, "x" * 5])

    def test_invalid_settings_raise(self):
        with self.assertRaises(ValueError):
            ingest.chunk_text(self.TEXT, chunk_size=0, overlap=0)
        with self.assertRaises(ValueError):
            ingest.chunk_text(self.TEXT, chunk_size=50, overlap=50)
        with self.assertRaises(ValueError):
            ingest.chunk_text(self.TEXT, chunk_size=50, overlap=-1)


class LoadDocumentsTests(unittest.TestCase):
    """load_documents: file types, empty files, missing folder, real corpus."""

    def test_reads_txt_and_md_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "b.md").write_text("markdown doc")
            (tmp / "a.txt").write_text("text doc")
            (tmp / "c.pdf").write_text("ignored")
            (tmp / "empty.txt").write_text("   ")
            docs = ingest.load_documents(tmp)
        self.assertEqual([d["filename"] for d in docs], ["a.txt", "b.md"])

    def test_non_utf8_file_is_skipped_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "good.txt").write_text("readable doc")
            (tmp / "bad.txt").write_bytes(b"caf\xe9 \xff\xfe broken")
            skipped = []
            docs = ingest.load_documents(tmp, skipped=skipped)
        self.assertEqual([d["filename"] for d in docs], ["good.txt"])
        self.assertEqual(skipped, ["bad.txt"])

    def test_missing_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            ingest.load_documents(Path("does-not-exist"))

    def test_corpus_has_at_least_eight_documents(self):
        self.assertGreaterEqual(len(ingest.load_documents(ingest.DOCS_DIR)), 8)


class PrecisionRecallTests(unittest.TestCase):
    """precision_recall: source-level metrics."""

    def test_perfect_match(self):
        self.assertEqual(precision_recall(["a.txt"], ["a.txt"]), (1.0, 1.0))

    def test_duplicates_count_once(self):
        self.assertEqual(precision_recall(["a.txt", "a.txt", "b.txt"], ["a.txt"]), (0.5, 1.0))

    def test_partial_recall(self):
        self.assertEqual(precision_recall(["a.txt"], ["a.txt", "b.txt"]), (1.0, 0.5))

    def test_empty_inputs(self):
        self.assertEqual(precision_recall([], ["a.txt"]), (0.0, 0.0))
        self.assertEqual(precision_recall(["a.txt"], []), (0.0, 0.0))


class SearchTests(unittest.TestCase):
    """search / get_collection_stats against a temporary index."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.chroma_path = Path(cls.tmp.name) / "chroma"
        cls.patch = mock.patch.object(search, "CHROMA_PATH", cls.chroma_path)
        cls.patch.start()
        ingest.ingest(chunk_size=300, overlap=60, chroma_path=cls.chroma_path, verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls.tmp.cleanup()

    def test_returns_ranked_results_with_scores(self):
        results = search.search("What is a primary key and a foreign key?", n_results=4)
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0]["source"], "sql-databases.txt")
        distances = [r["distance"] for r in results]
        self.assertEqual(distances, sorted(distances))
        for r in results:
            self.assertEqual(set(r), {"text", "source", "chunk_index", "distance", "score", "relevance"})
            self.assertAlmostEqual(r["score"], 1 - r["distance"], places=3)
            self.assertIn(r["relevance"], {"High", "Medium", "Low"})

    def test_empty_query_returns_nothing(self):
        self.assertEqual(search.search(""), [])
        self.assertEqual(search.search("   "), [])

    def test_punctuation_only_query_returns_nothing(self):
        for query in ["???", " !!! ...", "🔎🔎", "-- // **"]:
            self.assertEqual(search.search(query), [], query)

    def test_is_searchable(self):
        self.assertTrue(search.is_searchable("what is SQL?"))
        self.assertTrue(search.is_searchable("404"))
        self.assertFalse(search.is_searchable(""))
        self.assertFalse(search.is_searchable("?!  ..."))

    def test_source_filter_with_more_results_than_chunks(self):
        fastapi_chunks = len(search.get_collection().get(where={"source": "fastapi.txt"})["ids"])
        results = search.search("api", n_results=50, sources=["fastapi.txt"])
        self.assertEqual(len(results), fastapi_chunks)
        self.assertEqual({r["source"] for r in results}, {"fastapi.txt"})

    def test_unknown_source_filter_returns_nothing(self):
        self.assertEqual(search.search("api", sources=["does-not-exist.txt"]), [])

    def test_zero_results_requested(self):
        self.assertEqual(search.search("python", n_results=0), [])

    def test_threshold_filters_results(self):
        self.assertEqual(search.search("python decorators", distance_threshold=0.0), [])
        for r in search.search("python decorators", n_results=10, distance_threshold=0.6):
            self.assertLessEqual(r["distance"], 0.6)

    def test_source_filter(self):
        results = search.search("databases", n_results=5, sources=["fastapi.txt"])
        self.assertTrue(results)
        self.assertEqual({r["source"] for r in results}, {"fastapi.txt"})

    def test_n_results_larger_than_collection(self):
        total = search.get_collection_stats()["total_chunks"]
        self.assertEqual(len(search.search("python", n_results=total + 50)), total)

    def test_stats(self):
        stats = search.get_collection_stats()
        self.assertEqual(stats["unique_sources"], 8)
        self.assertEqual(stats["chunk_size"], 300)
        self.assertEqual(stats["overlap"], 60)
        self.assertGreater(stats["total_chunks"], 8)

    def test_metadata_stored_per_chunk(self):
        metadatas = search.get_collection().get(include=["metadatas"])["metadatas"]
        for m in metadatas:
            self.assertEqual(set(m), {"source", "chunk_index", "chunk_size", "overlap"})


class EmptyCollectionTests(unittest.TestCase):
    """search on a collection with no documents."""

    def test_empty_collection(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(search, "CHROMA_PATH", Path(tmp) / "chroma"):
            self.assertEqual(search.search("anything"), [])
            self.assertEqual(search.get_collection_stats()["total_chunks"], 0)


class IngestSafetyTests(unittest.TestCase):
    """A failed re-index must leave the existing index untouched."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.chroma_path = self.root / "chroma"
        ingest.ingest(chunk_size=500, overlap=100, chroma_path=self.chroma_path, verbose=False)
        self.before = self.count()

    def tearDown(self):
        self.tmp.cleanup()

    def count(self) -> int:
        """Number of chunks currently in the temporary index."""
        return ingest.get_collection(self.chroma_path, ingest.COLLECTION_NAME).count()

    def test_empty_docs_folder_keeps_index(self):
        empty = self.root / "empty_docs"
        empty.mkdir()
        with self.assertRaises(ValueError):
            ingest.ingest(docs_dir=empty, chroma_path=self.chroma_path, verbose=False)
        self.assertEqual(self.count(), self.before)

    def test_only_unreadable_files_keeps_index(self):
        bad = self.root / "bad_docs"
        bad.mkdir()
        (bad / "bad.txt").write_bytes(b"\xff\xfe\xfa")
        with self.assertRaises(ValueError):
            ingest.ingest(docs_dir=bad, chroma_path=self.chroma_path, verbose=False)
        self.assertEqual(self.count(), self.before)

    def test_invalid_chunk_settings_keep_index(self):
        with self.assertRaises(ValueError):
            ingest.ingest(chunk_size=100, overlap=100, chroma_path=self.chroma_path, verbose=False)
        self.assertEqual(self.count(), self.before)

    def test_unreadable_file_is_skipped_during_ingest(self):
        docs = self.root / "mixed_docs"
        docs.mkdir()
        (docs / "good.txt").write_text("SQL joins combine rows from two tables.")
        (docs / "bad.txt").write_bytes(b"\xff\xfe\xfa")
        summary = ingest.ingest(docs_dir=docs, chroma_path=self.chroma_path, verbose=False)
        self.assertEqual(summary["documents"], 1)
        self.assertEqual(summary["skipped"], ["bad.txt"])


class AppTests(unittest.TestCase):
    """Run the real Streamlit script against temporary indexes."""

    APP = str(Path(__file__).resolve().parent / "app.py")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.chroma_path = Path(self.tmp.name) / "chroma"
        self.patch = mock.patch.object(search, "CHROMA_PATH", self.chroma_path)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def run_app(self, query: str = None) -> AppTest:
        """Run app.py (optionally submitting a query) and assert it raised nothing."""
        app = AppTest.from_file(self.APP, default_timeout=60).run()
        if query is not None:
            app.text_input[0].input(query).run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        return app

    def warnings(self, app: AppTest) -> str:
        """All warning messages on the page, joined into one string."""
        return " ".join(w.value for w in app.warning)

    def test_empty_collection_shows_warning(self):
        app = self.run_app()
        self.assertIn("collection is empty", self.warnings(app))

    def test_chunk_size_above_sidebar_range_does_not_crash(self):
        ingest.ingest(chunk_size=3000, overlap=200, chroma_path=self.chroma_path, verbose=False)
        app = self.run_app()
        self.assertEqual(app.number_input[0].value, 2000)
        self.assertEqual(app.number_input[1].value, 200)

    def test_chunk_size_below_sidebar_range_does_not_crash(self):
        ingest.ingest(chunk_size=30, overlap=20, chroma_path=self.chroma_path, verbose=False)
        app = self.run_app()
        self.assertEqual(app.number_input[0].value, 50)
        self.assertEqual(app.number_input[1].value, 20)

    def test_off_topic_query_warns_no_strong_matches(self):
        ingest.ingest(chroma_path=self.chroma_path, verbose=False)
        app = self.run_app("best pizza dough recipe")
        self.assertIn("No strong matches", self.warnings(app))

    def test_on_topic_query_has_no_weak_match_warning(self):
        ingest.ingest(chroma_path=self.chroma_path, verbose=False)
        app = self.run_app("Why are dictionary lookups faster than searching a list?")
        self.assertNotIn("No strong matches", self.warnings(app))
        self.assertIn("Results", [m.label for m in app.metric])

    def test_overlap_survives_chunk_size_change(self):
        ingest.ingest(chroma_path=self.chroma_path, verbose=False)
        app = self.run_app()
        app.number_input[1].set_value(60).run()
        app.number_input[0].set_value(300).run()
        self.assertEqual(app.number_input[0].value, 300)
        self.assertEqual(app.number_input[1].value, 60)

    def test_overlap_not_smaller_than_chunk_size_blocks_reindex(self):
        ingest.ingest(chroma_path=self.chroma_path, verbose=False)
        app = self.run_app()
        app.number_input[0].set_value(100).run()
        app.number_input[1].set_value(150).run()
        self.assertIn("Overlap must be smaller", " ".join(e.value for e in app.error))
        self.assertTrue(app.button[0].disabled)
        app.number_input[1].set_value(20).run()
        self.assertFalse(app.error)
        self.assertFalse(app.button[0].disabled)

    def test_weak_match_warning_mentions_filter(self):
        ingest.ingest(chroma_path=self.chroma_path, verbose=False)
        app = self.run_app()
        app.multiselect[0].set_value(["fastapi.txt"]).run()
        app.text_input[0].input("best pizza dough recipe").run()
        self.assertIn("selected source files", self.warnings(app))
        app.multiselect[0].set_value([]).run()
        self.assertIn("documents may not cover", self.warnings(app))

    def test_punctuation_only_query_warns(self):
        ingest.ingest(chroma_path=self.chroma_path, verbose=False)
        app = self.run_app("???")
        self.assertIn("only symbols", self.warnings(app))


class CommandLineTests(unittest.TestCase):
    """Bad command-line arguments give a usage message, not a traceback."""

    HERE = Path(__file__).resolve().parent

    def run_script(self, *args: str) -> subprocess.CompletedProcess:
        """Run one of the project scripts with the current interpreter."""
        return subprocess.run(
            [sys.executable, *args], cwd=self.HERE, capture_output=True, text=True, timeout=300
        )

    def assert_usage_error(self, result: subprocess.CompletedProcess, message: str):
        """Exit code 2, a usage line, the message, and no traceback."""
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("usage:", result.stderr)
        self.assertIn(message, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_ingest_overlap_not_smaller_than_chunk_size(self):
        self.assert_usage_error(
            self.run_script("ingest.py", "--chunk-size", "100", "--overlap", "100"),
            "overlap must be >= 0 and smaller than chunk_size",
        )

    def test_ingest_zero_chunk_size(self):
        self.assert_usage_error(
            self.run_script("ingest.py", "--chunk-size", "0"), "chunk_size must be at least 1"
        )

    def test_evaluate_zero_results(self):
        self.assert_usage_error(
            self.run_script("evaluate.py", "--n-results", "0"), "--n-results must be at least 1"
        )

    def test_experiment_bad_overlap_ratio(self):
        self.assert_usage_error(
            self.run_script("experiment.py", "--overlap-ratio", "1.5"), "--overlap-ratio must be"
        )

    def test_experiment_bad_chunk_size(self):
        self.assert_usage_error(
            self.run_script("experiment.py", "--chunk-sizes", "300", "0"), "--chunk-sizes must"
        )


if __name__ == "__main__":
    unittest.main()
