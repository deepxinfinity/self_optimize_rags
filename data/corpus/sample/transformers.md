# The Transformer Architecture

The Transformer is a neural network architecture introduced in the paper "Attention Is All You
Need" (Vaswani et al., 2017). Unlike recurrent neural networks, the Transformer relies entirely
on a mechanism called self-attention to draw global dependencies between input and output, without
using sequence-aligned recurrence or convolution.

## Self-Attention

Self-attention, also called scaled dot-product attention, computes a weighted sum of value vectors,
where the weight assigned to each value is determined by a compatibility function of the query with
the corresponding key. The scaling factor is the inverse square root of the key dimension, which
prevents the dot products from growing too large in magnitude and pushing the softmax into regions
with vanishingly small gradients.

## Multi-Head Attention

Rather than performing a single attention function, the Transformer uses multiple attention heads
in parallel. Each head projects the queries, keys, and values into a lower-dimensional space,
performs attention, and the results are concatenated and projected again. Multi-head attention lets
the model jointly attend to information from different representation subspaces at different
positions.

## Positional Encoding

Because the Transformer contains no recurrence and no convolution, it injects information about the
relative or absolute position of tokens using positional encodings. The original paper uses sine
and cosine functions of different frequencies added to the input embeddings.
