# Prompt Injection

Prompt injection is when text supplied by a user or a retrieved document contains instructions that try to override the system prompt, for example "ignore all previous instructions".

In a RAG system the documents themselves are an attack surface. If someone can add a file to the corpus, they can plant instructions that the model may follow when that chunk is retrieved.

Defences from the injection_defense exercise: wrap untrusted text in clear delimiters, tell the model that context is data and not instructions, filter obvious attack phrases before they reach the model, and check the output for leaked system prompt text.

No single defence is complete. Layering several checks and limiting what the model is allowed to do with its output is the practical approach.
