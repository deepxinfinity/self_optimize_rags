# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a parametric language model with a non-parametric
memory in the form of a document index. Instead of relying solely on knowledge stored in the model
weights, a RAG system retrieves relevant passages from an external corpus at inference time and
conditions the generator on those passages.

## Dense vs Sparse Retrieval

Sparse retrieval methods such as BM25 score documents using term-frequency statistics over an
inverted index; they are fast and strong on exact keyword matches. Dense retrieval instead embeds
queries and documents into a shared vector space and ranks by cosine similarity, capturing semantic
matches that share no surface terms. Hybrid retrieval linearly combines the two scores with a
tunable weight to get the benefits of both.

## Reranking

A first-stage retriever favors recall, returning a broad candidate set. A cross-encoder reranker
then jointly encodes the query and each candidate passage to produce a precise relevance score,
reordering the shortlist. Reranking improves precision at the top ranks at the cost of extra latency,
so systems typically rerank only a small number of candidates.

## Chunking

Long documents are split into chunks before indexing. Chunk size and overlap trade off retrieval
granularity against context completeness: small chunks localize the answer but may split evidence,
while large chunks preserve context but dilute the relevance signal.
