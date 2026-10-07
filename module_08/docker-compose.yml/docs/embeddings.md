# Embeddings

An embedding is a list of numbers (a vector) that represents the meaning of a piece of text. Texts with similar meaning end up close together in vector space, even if they share no words.

ChromaDB uses the all-MiniLM-L6-v2 sentence-transformer model by default. It produces 384-dimensional vectors and runs locally on CPU, so no API key is needed.

Similarity between two embeddings is usually measured with cosine similarity or L2 (Euclidean) distance. With distance, lower means more similar. With cosine similarity, higher means more similar.

The same embedding model must be used for indexing and querying. Mixing models gives vectors from different spaces and the search results become meaningless.
