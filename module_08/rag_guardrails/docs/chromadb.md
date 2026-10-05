# ChromaDB

ChromaDB is an open-source vector database. It stores documents, their embeddings, metadata and ids in collections, and handles the embedding step automatically if you pass raw text.

chromadb.Client() keeps everything in memory and loses it when the script exits. chromadb.PersistentClient(path="./rag_db") writes the data to disk in that folder so it survives restarts.

collection.add() fails if an id already exists. collection.upsert() inserts new ids and overwrites existing ones, which makes re-running an ingestion script safe.

collection.query(query_texts=[...], n_results=k) returns a dict of lists: documents, metadatas, distances and ids. Each is nested one level deep because you can send several queries at once, so results["documents"][0] holds the matches for the first query.

Metadata values must be strings, ints, floats or bools. A where filter such as where={"source": "notes.md"} restricts the search to matching chunks.
