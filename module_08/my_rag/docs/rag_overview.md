# Retrieval-Augmented Generation (RAG)

RAG combines a search step with a language model. Instead of relying only on what the model memorised during training, the system retrieves relevant text from your own documents and passes it to the model as context.

The pipeline has three stages: ingest (load, chunk, embed and store documents), retrieve (embed the question and find the closest chunks), and generate (build a prompt with the chunks and ask the LLM to answer).

RAG reduces hallucination because the model is told to answer only from the supplied context. It also lets you update knowledge by changing the documents rather than retraining the model.

A good RAG system prompt tells the model to cite its sources and to admit when the context does not contain the answer. Without that instruction small models tend to fill gaps with made-up facts.

Top-k is the number of chunks retrieved. Too few and the answer may be missing; too many and the prompt fills with noise and gets slower. Three to five chunks is a common starting point.
