# FastFormer-pytorch
Implementation of FastFormer in Pytorch

## Fastformer for Sentiment Analysis

A fast and efficient transformer-inspired architecture that leverages global context without the quadratic overhead of traditional self-attention. This repository implements a lightweight version of the Fastformer designed for sentiment analysis, combining efficiency with expressiveness.

The paper: https://arxiv.org/pdf/2108.09084v6

---

## Table of Contents

- [Overview](#overview)
- [Architecture Details](#architecture-details)
  - [Input Projection](#input-projection)
  - [Additive Attention](#additive-attention)
  - [Global Context for Keys](#global-context-for-keys)
  - [Global Context for Values](#global-context-for-values)
  - [Output Transformation](#output-transformation)
  - [Fastformer Layer](#fastformer-layer)
  - [Fastformer for Sentiment Analysis](#fastformer-for-sentiment-analysis)

---

## Overview

The Fastformer architecture presented here is a streamlined alternative to the classic Transformer model. It reduces computational overhead by avoiding full pairwise attention. Instead, it computes a global context vector through averaging (additive attention), which is then fused with key and value matrices to produce context-aware representations.

This repository implements the following key modules:

- **InputProjection:** Projects input embeddings into queries, keys, and values.
- **AdditiveAttention:** Averages query representations to produce a global query.
- **GlobalContextKey:** Fuses key representations with the global query to yield a context-aware global key.
- **GlobalContextValue:** Incorporates global key information into value representations.
- **OutputTransformation:** Applies a final linear transformation with residual connections.
- **FastformerLayer:** Combines the above modules into a complete attention layer.
- **FastformerForSentimentAnalysis:** Stacks Fastformer layers with an embedding, pooling, and classification head for sentiment analysis tasks.

---

## Architecture Details

### Input Projection

- **Module:** `InputProjection`
- **Functionality:**  
  Maps input embeddings into three separate spaces (Q, K, V) using three linear layers.  
- **Input/Output:**  
  - Input: `[batch_size, seq_len, input_dim]`  
  - Output: Three tensors with shape `[batch_size, seq_len, d_k]` for queries, keys, and values.

---

### Additive Attention

- **Module:** `AdditiveAttention`
- **Functionality:**  
  Computes a global query vector by averaging the query matrix along the sequence dimension.  
- **Input/Output:**  
  - Input: `[batch_size, seq_len, d_k]`  
  - Output: Global query vector `[batch_size, d_k]`.

---

### Global Context for Keys

- **Module:** `GlobalContextKey`
- **Functionality:**  
  Multiplies the key matrix with the global query vector (broadcasting over the sequence) and then averages to produce a global key vector.  
- **Input/Output:**  
  - Inputs: Key matrix `[batch_size, seq_len, d_k]`, global query `[batch_size, d_k]`  
  - Outputs: Global key vector `[batch_size, d_k]` and context-aware key matrix.

---

### Global Context for Values

- **Module:** `GlobalContextValue`
- **Functionality:**  
  Fuses the value matrix with the global key vector to produce context-aware values.  
- **Input/Output:**  
  - Inputs: Value matrix `[batch_size, seq_len, d_k]`, global key `[batch_size, d_k]`  
  - Output: Global context-aware value matrix `[batch_size, seq_len, d_k]`.

---

### Output Transformation

- **Module:** `OutputTransformation`
- **Functionality:**  
  Applies a linear transformation to the context-aware values and then adds the original query for a residual connection.  
- **Input/Output:**  
  - Inputs: Global context-aware value, query matrix  
  - Output: Final output `[batch_size, seq_len, d_k]`.

---

### Fastformer Layer

- **Module:** `FastformerLayer`
- **Functionality:**  
  Combines the above modules to form a single attention layer:
  1. Projects inputs into Q, K, V.
  2. Computes a global query.
  3. Generates a context-aware global key.
  4. Computes context-aware values.
  5. Merges values with queries via output transformation.
- **Output:**  
  Tensor of shape `[batch_size, seq_len, d_k]`.

---

### Fastformer for Sentiment Analysis

- **Module:** `FastformerForSentimentAnalysis`
- **Functionality:**  
  An end-to-end model that:
  - Embeds tokenized input using an embedding layer.
  - Processes embeddings through sequential Fastformer layers.
  - Applies adaptive average pooling over the sequence dimension.
  - Uses dropout and a final linear layer to classify sentiment.
- **Workflow:**
  1. **Embedding:** Convert input token IDs to embeddings.
  2. **Encoding:** Process embeddings with Fastformer layers.
  3. **Pooling:** Pool features to get a fixed-length representation.
  4. **Classification:** Classify using a linear head to output sentiment logits.

---

