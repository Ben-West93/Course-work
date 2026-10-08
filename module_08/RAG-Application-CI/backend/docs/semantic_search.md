# Semantic Search and Thresholds

Semantic search ranks results by meaning rather than keyword overlap. A query like "how do I save vectors to disk" can match a note about PersistentClient even though the words differ.

Every query returns the k nearest chunks, even when nothing is actually relevant. A distance threshold filters out weak matches so the system can say "no results" instead of returning noise.

In the threshold experiment, L2 distances below about 1.0 were usually on-topic for all-MiniLM-L6-v2, while unrelated queries came back around 1.4 to 1.8. The right cutoff depends on the model and the corpus, so it has to be tuned with real queries.

Evaluating a search system uses metrics like precision@k (how many of the top k results are relevant) and recall (how many of the relevant documents were found at all).
