# Evaluating Retrieval and Generation

Evaluating a RAG system separates cleanly into retrieval metrics and answer-quality metrics.

## Retrieval Metrics

Hit rate measures the fraction of queries for which at least one gold-relevant document appears in
the retrieved set. Mean Reciprocal Rank (MRR) rewards ranking the first relevant document highly: it
is the average over queries of one divided by the rank of the first relevant result. Both metrics
require gold relevance labels and need no answer generation, which makes them cheap to compute.

## Judged Metrics

Faithfulness measures whether the generated answer is grounded in the retrieved context, penalizing
claims that cannot be supported by the passages. Answer relevancy measures whether the answer
actually addresses the question. These metrics are typically computed with an LLM-as-judge, which is
more expensive and should use a model from a different family than the answerer to avoid
self-evaluation bias.

## Synthetic Evaluation Reliability

A central question is whether optimizing against synthetically generated evaluation questions yields
gains that transfer to real questions, and whether the model that generated the synthetic set
matters. Measuring transfer requires a held-out set of real questions that the optimizer never sees.
