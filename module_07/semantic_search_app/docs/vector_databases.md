# Vector Databases and ChromaDB

A vector database stores embeddings, which are lists of numbers that represent the meaning of text, images, or other data, and retrieves the stored items that are nearest to a query vector. Instead of matching exact keywords like a SQL LIKE clause, a vector database answers questions such as "which paragraphs mean something similar to this question?" This is the retrieval step behind semantic search and retrieval-augmented generation (RAG).

ChromaDB is an open-source vector database that runs inside your Python process. chromadb.Client() creates an in-memory store that disappears when the program exits, while chromadb.PersistentClient(path="./chroma_db") writes everything to disk so the data survives restarts. Data lives in collections, created with client.get_or_create_collection(name="docs"), which is safe to call on every run.

Each record in a collection has a unique string ID, the original document text, an optional embedding, and an optional metadata dictionary. collection.add() inserts new records and raises an error if an ID already exists, while collection.upsert() inserts new IDs and overwrites existing ones. Upsert is the right choice for re-indexing because running it twice with the same IDs leaves the count unchanged.

Queries are made with collection.query(query_texts=[...]) or query_embeddings=[...], plus n_results to limit how many neighbours come back. The response contains parallel lists of ids, documents, metadatas, and distances, nested one level deep because you can send several queries at once. Lower distance means more similar; with the default L2 metric and normalized embeddings, distances range from 0 to 4.

Metadata filters narrow the search before similarity ranking. The where parameter accepts operators such as $eq, $ne, $gt, $lt, $in, and $nin. For example, where={"source": {"$in": ["a.md", "b.md"]}} restricts results to chunks whose source metadata is one of the listed files. Filters can be combined with $and and $or. collection.count() returns the total number of records, and collection.get() fetches records by ID or filter without ranking.
