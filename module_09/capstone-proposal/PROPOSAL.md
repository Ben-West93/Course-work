# JobFit: Capstone Proposal

**JobFit** keeps the job postings I'm applying to next to my skills profile and shows which skills each role wants that I don't have yet, quoting the posting lines it's based on.

**User story:** As a junior developer applying for AI engineering roles, I want to compare my skills against the job postings I've saved so that I know which gaps to close before I apply.

**Why this one:** SQL and RAG both do real work. The match score is a SQL query over structured skills, and the LLM explains the gaps from the posting text with citations. Study Buddy repeats Module 08, Recipes and Knowledge Base come down to search, and Documentation Helper needs made-up docs.

## Required components

| Component | In JobFit |
|---|---|
| FastAPI | `/auth/register`, `/auth/token`, `/me/skills`, `/jobs` (CRUD), `/jobs/{id}/analyze`, `/ask`, `/recommendations` |
| SQLAlchemy | `users` 1→N `job_postings` 1→N `analyses`, plus `skills` via two link tables |
| Pydantic | request and response models on every route, length limits on posting text |
| JWT auth | OAuth2 password flow, 30-minute PyJWT tokens, Argon2 hashes, per-user queries |
| RAG | postings in ChromaDB tagged by user and job, `llama3.2:1b` answers with citations |
| Streamlit | login, My Skills, Jobs, Analyze, Recommendations and Ask pages |
| Compose | `docker-compose up --build` runs Postgres, Ollama, model pull, API, UI |
| 5+ tests | about 12 pytest tests (Ollama mocked): auth, user isolation, cascades, validation |
| README | setup, Mermaid architecture diagram, usage guide |
| Requirements | pinned backend and frontend `requirements.txt`, plus `requirements-dev.txt` |

**Data source:** about 25 sample junior AI and backend postings for fictional companies (`backend/seed/jobs/`), plus my own postings pasted or uploaded as `.txt`/`.md`, never committed. A CSV of about 150 skills with aliases drives skill extraction, so scores are repeatable.

**Tech stack:** FastAPI + SQLAlchemy + ChromaDB + Ollama + Streamlit + Docker Compose.

## Risks

1. **Ollama isn't ready for the first request** (model still downloading or loading). Backup: a model-pull service, a warm-up call and a `degraded` health status. The match score comes from SQL, so Analyze works without Ollama. `MODEL_NAME=llama3.2:3b` if 1B is too weak.
2. **Postgres and ChromaDB drift apart** (stale chunks, or a new embedding model breaking the collection). Backup: Postgres is the source of truth. One write path replaces a posting's chunks and sets `indexed_at` on success, with a reindex endpoint and the embedding model in the collection name.
