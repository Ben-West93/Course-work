# Course-work

My coursework, one folder per project inside each module. Each project runs on
its own: its README (or the docstring at the top of the script) says how.

## Module 07: embeddings, semantic search and prompts

| Project | What it is |
|---|---|
| [chunk_compare](module_07/chunk_compare/) | Compares fixed-size sliding-window chunking with paragraph chunking for semantic search |
| [course_search](module_07/course_search/) | Interactive semantic search over course-summary sentences |
| [eval_search_system](module_07/eval_search_system/) | Measures a ChromaDB search with precision and recall across several retrieval settings |
| [injection_defense](module_07/injection_defense/) | Input and output validators plus a hardened system prompt that protect a RAG pipeline from prompt injection |
| [knowledge_base](module_07/knowledge_base/) | A persistent ChromaDB collection of course notes with metadata-filtered semantic search |
| [prompt_workshop](module_07/prompt_workshop/) | Bad and good prompt pairs run side by side, plus a system prompt for a course study assistant |
| [semantic_search_app](module_07/semantic_search_app/) | Streamlit and ChromaDB app that searches the course notes by meaning |
| [semantic_search_tool](module_07/semantic_search_tool/) | Semantic search over a folder of documents, chunked and embedded with all-MiniLM-L6-v2 |
| [threshold_experiment](module_07/threshold_experiment/) | How similarity thresholds of 0.3, 0.5 and 0.7 change which sentences come back for the same query |

## Module 08: RAG, FastAPI and Docker

| Project | What it is |
|---|---|
| [ollama_explorer](module_08/ollama_explorer/) | Four experiments against a local model through Ollama's `/api/chat` |
| [prompt_builder](module_08/prompt_builder/) | The prompt assembly step of a RAG pipeline |
| [my_rag](module_08/my_rag/) | A RAG pipeline over my course notes: ChromaDB retrieval and streamed answers |
| [rag_guardrails](module_08/rag_guardrails/) | `my_rag` with four reliability guardrails added |
| [my_rag_api](module_08/my_rag_api/) | The RAG pipeline wrapped in a FastAPI app |
| [docker-first-steps](module_08/docker-first-steps/) | A minimal FastAPI app and its Dockerfile |
| [dockerfile-creation](module_08/dockerfile-creation/) | `my_rag_api` packaged as a Docker image |
| [Dockerfile-RAG-API](module_08/Dockerfile-RAG-API/) | The RAG FastAPI backend built into a Docker image |
| [docker-compose.yml](module_08/docker-compose.yml/) | The RAG backend and Ollama as two containers on one Compose network (the folder is named after the file) |
| [Containerizing-Your-RAG-Application](module_08/Containerizing-Your-RAG-Application/) | The whole RAG app as one Compose stack: FastAPI backend, Streamlit frontend and Ollama |
| [Environment-management-and-configuration](module_08/Environment-management-and-configuration/) | The Compose stack with every setting moved into environment variables |
| [RAG-Application-CI](module_08/RAG-Application-CI/) | GitHub Actions CI for the RAG app: tests, Docker builds and lint |
| [Containerized-RAG](module_08/Containerized-RAG/) | **Module 08 capstone.** A question-answering assistant over course notes: FastAPI, ChromaDB, Ollama and Streamlit in Docker Compose |

## Module 09: capstone

| Project | What it is |
|---|---|
| [capstone-proposal](module_09/capstone-proposal/) | The one-page proposal for JobFit |
| [capstone](module_09/capstone/) | **JobFit.** Saved job postings, and questions about them answered with citations. FastAPI, SQLAlchemy, Postgres, ChromaDB, Ollama and Streamlit in Docker Compose. The Day 8 status is in [`CHECK_IN.md`](module_09/capstone/CHECK_IN.md) |
