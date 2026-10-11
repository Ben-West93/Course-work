# JobFit: Capstone Proposal

**JobFit** keeps the job postings I'm applying to next to my skills profile, then shows which skills each role wants that I don't have yet, quoting the lines of the posting it's based on.

**User story:** As a junior developer applying for AI engineering roles, I want to compare my skills against the job postings I've saved so that I know which gaps to close before I apply.

## Why this project

| Option | Verdict |
|---|---|
| Study Buddy | Solid, but it's my Module 08 RAG app with uploads added. Not much new to show. |
| Recipe Assistant | The RAG part mostly turns into filtered search, and I'd need a recipe dataset I'm allowed to redistribute. |
| **Job Posting Analyzer** | **Chosen.** |
| Documentation Helper | Close second. Department-scoped retrieval is a good security story, but the whole corpus would be an invented company. |
| Personal Knowledge Base | Clean CRUD, but summarizing my own notes adds little over search, and notes apps are in every portfolio. |

The job analyzer is the one option where SQL and RAG each do work the other can't. Which skills a posting asks for and which ones I have are structured facts, so they live in tables and the match score is a query, not a guess from a 1B model. The posting text is unstructured, so it goes into ChromaDB, and the LLM uses it to explain the gaps and cite what it read. It also makes the best interview demo: paste in the interviewer's own posting and run the analysis live. And I'll actually use it.

## Required components

| Component | In JobFit |
|---|---|
| FastAPI, 3+ endpoints | `POST /auth/register`, `POST /auth/token`, `GET`/`PUT /me/skills`, `GET`/`POST`/`PUT`/`DELETE /jobs`, `POST /jobs/{id}/analyze`, `POST /ask`, `GET /recommendations`, `GET /health` |
| SQLAlchemy, 2+ related tables | `users` 1→N `job_postings` 1→N `analyses`. `skills` links to users through `user_skills` (level 1–5) and to postings through `job_skills` (required or nice-to-have) |
| Pydantic on every endpoint | request models and a `response_model` on every route (`JobCreate`, `JobOut`, `SkillLevel`, `AnalysisOut`, `AskRequest`, `AskResponse` with `Citation`), length limits on posting text, normalized skill names |
| JWT auth | OAuth2 password flow on `/auth/token`, PyJWT tokens that expire after 30 minutes, Argon2 hashes via pwdlib. Every query is scoped to the token's user. Streamlit keeps the token in session state and goes back to login on a 401 |
| RAG (ChromaDB + Ollama) | postings chunked with my Module 08 chunker and stored with `user_id` and `job_id` metadata. Analyze retrieves one posting's chunks, ask searches all of the user's postings, and `llama3.2:1b` writes the answer with chunk citations |
| Streamlit frontend | login, My Skills, Jobs (add, edit, delete, upload), Analyze (match %, missing skills, cited excerpts), Recommendations (postings ranked by match, plus the missing skill that would raise the most of them), Ask |
| Docker Compose, one command | `docker-compose up --build` starts `db` (Postgres), `ollama`, a one-off `ollama-pull` that downloads the model, `backend` and `frontend`, ordered by healthchecks. No manual `ollama pull` step like in Module 08 |
| 5+ passing tests | about a dozen pytest tests with Ollama mocked: register and login, wrong password and expired token give 401, user B gets 404 on user A's posting, deleting a posting removes its chunks, bad input gives 422, match score maths, `/ask` returns citations |
| README | setup, Mermaid architecture diagram, usage walkthrough with screenshots, endpoint table, env vars, troubleshooting |
| requirements files | pinned `backend/requirements.txt`, `backend/requirements-dev.txt` (test tools only) and `frontend/requirements.txt` |

## Data source

- **Sample postings:** about 25 junior AI, ML and backend postings for fictional companies, written for this project as Markdown in `backend/seed/jobs/` and loaded for a demo account on first start.
- **My own postings:** pasted into the Jobs page or uploaded as `.txt`/`.md`. Stored per user in the database, never committed.
- **Skills list:** a CSV of about 150 skills with aliases (`k8s` → Kubernetes, `Postgres` → PostgreSQL) seeded into `skills`. Pulling skills out of a posting is a match against this list, so the score comes out the same every run.

PDF parsing and scraping job boards are out of scope.

## Tech stack

Confirmed: FastAPI + SQLAlchemy 2.0 (Postgres in Compose, in-memory SQLite in tests) + ChromaDB (all-MiniLM-L6-v2 baked into the image) + Ollama (`llama3.2:1b`) + Streamlit + Docker Compose.

## Risks

**1. Ollama isn't ready for the first request.** On a fresh `up` the model might still be downloading, and the first call has to load it into memory, which can take long enough to time out. The 1B model may also write a thin analysis.
Backup: the `ollama-pull` service handles the download, the backend sends a warm-up prompt once the model is there, and `/health` reports `degraded` so the UI shows "model loading" instead of an error. The match score and missing-skills list come from SQL, so Analyze still works with Ollama down. If the 1B output is too weak, `MODEL_NAME=llama3.2:3b` in `.env` swaps the model without code changes.

**2. Postgres and ChromaDB drift apart.** Editing or deleting a posting leaves stale chunks in Chroma, a failed embed leaves a posting with no vectors, and switching embedding models later breaks the collection with a dimension mismatch.
Backup: Postgres is the source of truth. Every posting write goes through one function that commits SQL, replaces that posting's chunks by `job_id`, and only then sets `indexed_at`. Postings without `indexed_at` are retried on startup, and `POST /jobs/{id}/reindex` does it by hand. The collection name includes the embedding model (`jobs_minilm_l6`), so a new model gets a new collection rebuilt from SQL instead of a crash.
