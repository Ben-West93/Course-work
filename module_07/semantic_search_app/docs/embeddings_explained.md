# Embeddings Explained

An embedding is a fixed-length vector of floating point numbers that captures the meaning of a piece of text. Sentences with similar meaning end up close together in this vector space even when they share no words, so "How do I fix a flat tire?" lands near "Repairing a punctured bicycle wheel." This property is what allows computers to search by meaning rather than by exact keywords.

The sentence-transformers library makes embeddings easy to produce locally. SentenceTransformer("all-MiniLM-L6-v2") loads a small, fast model that outputs 384-dimensional vectors, and model.encode(["first sentence", "second sentence"]) returns a NumPy array with one row per input. The model is downloaded once (about 80 MB) and cached on disk for later runs.

Similarity between two embeddings is measured with a distance or similarity function. Cosine similarity compares the angle between vectors and ranges from -1 to 1, where 1 means identical direction. Euclidean (L2) distance measures straight-line distance, where 0 means identical. For normalized vectors the two are directly related, so ranking by either produces the same order.

Chunking matters because embedding models have a maximum input length and because one vector cannot represent many unrelated ideas well. A common approach is fixed-size chunking with overlap: split the text into windows of, say, 500 characters and start each window 400 characters after the previous one, so 100 characters are shared. The overlap prevents a sentence that falls on a boundary from being cut in half and lost.

Choosing chunk size is a trade-off. Small chunks give precise matches but can lose surrounding context, while large chunks keep context but dilute the signal so that the vector represents an average of several topics. Paragraph-based chunking respects natural boundaries but produces uneven sizes. Experimenting with a few sizes and checking results on realistic queries is the most reliable way to choose.
