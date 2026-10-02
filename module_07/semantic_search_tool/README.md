# Module 7 Project: Semantic Search Tool

## Overview

This is a semantic search tool for a folder of documents. The documents in `docs/`
are split into overlapping chunks, embedded with the `all-MiniLM-L6-v2` model, and
stored in a ChromaDB collection. The Streamlit app lets you type a question and see
the closest matching chunks, ranked by similarity score. I also ran a chunking
experiment to compare three chunk sizes, and the results are at the bottom of this
file.

## Setup

You need Python 3.10 or newer. I used Python 3.13.

1. Create a virtual environment and install the requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Build the index:
   ```bash
   python ingest.py
   ```
   You can also pick the chunk size and overlap, for example
   `python ingest.py --chunk-size 300 --overlap 60`.

   The first run downloads the embedding model (about 90 MB), so you need to be
   online for that. After that it is cached and works offline.

3. Start the app:
   ```bash
   streamlit run app.py
   ```
   The index can also be rebuilt from the sidebar with the Re-index button.

   Note: I turned off Streamlit's file watcher in `.streamlit/config.toml`. With it
   on, the terminal fills up with `ModuleNotFoundError` messages from the
   `transformers` library (they don't break anything, but they're noisy). Because
   of this the app won't reload by itself after a code change, so stop it with
   Ctrl+C and start it again.

4. Other scripts:
   ```bash
   python evaluate.py      # precision and recall on the current index
   python experiment.py    # runs the chunking experiment
   python -m unittest test_search_tool.py -v
   ```

## Files

| File | What it does |
|------|--------------|
| `app.py` | Streamlit interface |
| `ingest.py` | Loads, chunks, embeds and stores the documents |
| `search.py` | Runs a query against ChromaDB and returns ranked results |
| `evaluate.py` | Test queries and precision/recall |
| `experiment.py` | Re-indexes at different chunk sizes and compares them |
| `experiment_results.md` | Full output from the experiment |
| `test_search_tool.py` | Tests |
| `docs/` | The 8 documents from the starter corpus |
| `chroma_data/` | The saved index (created when you run `ingest.py`, not committed) |
| `requirements.txt` | Package versions |

## How It Works

**Ingestion (`ingest.py`).** `load_documents` reads every `.txt` and `.md` file in
`docs/`. `chunk_text` splits each document into fixed-size chunks by character
count, with some overlap between neighbouring chunks. I made the chunk edges move to
the nearest space so words don't get cut in half. Each chunk is embedded and saved
with its source file, chunk index, chunk size and overlap. Every run clears the old
collection first so two chunk sizes never get mixed together. The collection uses
cosine distance, so the score is `1 - distance`.

**Search (`search.py`).** `search()` embeds the question with the same model and
asks ChromaDB for the closest chunks. It can filter to certain source files and drop
anything past a distance threshold. Each result has the text, source, chunk index,
distance, score and a relevance label (High if the score is 0.50 or more, Medium if
0.30 or more, otherwise Low).

**App (`app.py`).** The sidebar shows the document and chunk counts, and has a
slider for the number of results, a minimum score slider, a source file filter, and
chunk size/overlap boxes with a Re-index button. Results show up as cards with the
source, score, relevance label and the chunk text.

### Edge cases

- Empty searches, or searches with only symbols like `???`, don't run. The app asks
  for a question instead.
- If nothing passes the score threshold, the app says so and suggests lowering it
  or clearing the filter.
- If the best match is weak (under 0.30), the results still show but with a warning
  that the documents might not cover that topic.
- If the index is empty, the app tells you to re-index.
- Bad chunk settings (like overlap bigger than the chunk size) are rejected.
- If `docs/` is empty or missing, or a file can't be read, the old index is kept
  instead of being wiped. Unreadable files are skipped and listed.

### Limitations

- Only files directly in `docs/` are read, not subfolders, and only `.txt` and
  `.md`.
- Re-indexing rebuilds everything, which is fine for 8 documents but would be slow
  for a lot of them.
- Chunks can start or end in the middle of a sentence.
- The model only reads the first 256 tokens of a chunk (roughly 1,300 characters
  here), so very large chunk sizes get cut off.
- The model is English only. Questions in other languages get low scores.

## Evaluation

`evaluate.py` has five test questions, each with the file that should answer it.
Precision and recall are counted by source file. On the default index (chunk size
500, overlap 100, top 5 results) I got:

- Average precision: 0.87
- Average recall: 1.00
- Top result was correct for 5 out of 5 questions

With `--threshold 0.6` (minimum score 0.4), recall dropped to 0.70. One question
had no chunk scoring above 0.4, and another only found one of its two files.

## Chunking Experiment Results

**Chunk sizes tested:** 150, 300 and 600 characters. The overlap was 20% of the
chunk size each time (30, 60 and 120), which gave 170, 87 and 44 chunks. The full
top-3 results for every question are in `experiment_results.md`.

**Test queries:**
1. How do I keep a variable's value between reruns in Streamlit? (`streamlit.txt`)
2. Which HTTP status code means the user is logged in but not allowed to access something? (`rest-apis.txt`)
3. How can my API automatically reject requests with bad data? (`fastapi.txt`, `rest-apis.txt`)
4. Why are dictionary lookups faster than searching a list? (`python-advanced.txt`)
5. How do I make several database writes all succeed or all fail together? (`sql-databases.txt`)

I tried to word these differently from the documents (for example "logged in but
not allowed" instead of "403 Forbidden") so the search had to match on meaning and
not just the same words.

**Overall results:**

| Chunk size | Chunks | Avg precision (top 5) | Avg recall | Top-1 correct | Relevant in top 3 | Avg top-3 score |
|---|---|---|---|---|---|---|
| 150 | 170 | 0.80 | 1.00 | 5/5 | 14/15 | 0.473 |
| 300 | 87 | 0.63 | 1.00 | 5/5 | 15/15 | 0.408 |
| 600 | 44 | 0.77 | 1.00 | 5/5 | 14/15 | 0.397 |

**By question (relevant in top 3 / average top-3 score):**

| Query | 150 | 300 | 600 | Best |
|---|---|---|---|---|
| 1. Streamlit reruns | 3/3, 0.432 | 3/3, 0.429 | 3/3, 0.414 | about the same |
| 2. HTTP "not allowed" | 3/3, 0.505 | 3/3, 0.395 | 3/3, 0.379 | 150 |
| 3. API rejects bad data | 3/3, 0.464 | 3/3, 0.422 | 3/3, 0.397 | 300 / 600 (both files in top 3) |
| 4. Dictionary lookups | 2/3, 0.540 | 3/3, 0.463 | 2/3, 0.488 | 300 |
| 5. All-or-nothing writes | 3/3, 0.424 | 3/3, 0.332 | 3/3, 0.305 | 150 |

**Findings:**

All three sizes found the right document for every question. The 8 documents are
about very different topics, so even big chunks end up in the right file. The
difference was in how high the scores were and how clean the top results were.

The 150-character chunks got the highest scores, especially on questions where the
answer is one sentence. For question 2 the top score was 0.599 at 150 vs 0.454 at
600, and for question 5 it was 0.501 vs 0.382. I think this is because a
150-character chunk is about the size of that one sentence, so its embedding is
mostly about the answer. A 600-character chunk also includes other things (like the
400 and 401 codes, or JOINs and indexes), which waters it down.

The 600-character chunks had the lowest scores on 4 of the 5 questions, including
the lowest average of the whole experiment (0.305 on question 5). On question 4 an
unrelated chunk from `embeddings-and-vectors.txt` also made it into the top 3.

Small chunks have a downside too. At 150 characters a result is often half a
sentence, so it points to the right place but doesn't always make sense on its own.
On question 4 a short chunk about vector search from `embeddings-and-vectors.txt`
got into the top 3. Question 3 showed another problem: at 150 the whole top 3 came
from `rest-apis.txt` (the bit about 400 Bad Request), and the FastAPI validation
chunk didn't show up until lower down. At 300 and 600 both files were in the top 3,
because a bigger chunk holds enough of the FastAPI explanation to match the question.

Overall I think 300 worked best. It was the only size where all 15 of the top-3
results were relevant, it found both relevant files for question 3, and the results
are long enough to actually read. Its precision at top 5 was the lowest (0.63), but
that's because once the best chunks from the right file are used up, results 4 and 5
come from other files. Each question only has one right file, so asking for 5
results makes that happen.

Question 1 barely changed between sizes (0.432, 0.429, 0.414). The answer is spread
across two paragraphs, so no chunk size picks it out much better than the others.

**Conclusion:** smaller chunks are better for short factual questions, but 300
characters with 20% overlap gave the best balance between accurate results and
results that are readable. I left the default at 500/100 from the starter, which is
in between, and the chunk size can be changed in the sidebar to compare.
