# Architecture Taxonomy

This document defines the initial taxonomy used by the repository.

## 1. Dense decoder-only Transformer

Core traits:

- autoregressive causal language modeling;
- token embedding plus stacked Transformer decoder blocks;
- self-attention with causal masking;
- MLP/FFN sublayers;
- normalization and residual topology;
- positional encoding, usually RoPE or a RoPE variant in recent models.

Important comparison axes:

- pre-norm vs post-norm;
- RMSNorm vs LayerNorm;
- SwiGLU/GeGLU/ReLU-style FFN;
- MHA vs MQA vs GQA;
- RoPE base and scaling strategy;
- sliding-window or global attention;
- tokenizer vocabulary and special-token policy.

## 2. Mixture-of-Experts Transformer

Core traits:

- sparse expert activation;
- router/gating network;
- top-k expert selection;
- load balancing strategy;
- expert parallelism during training/inference.

Important comparison axes:

- number of total experts;
- active experts per token;
- shared experts vs routed experts;
- auxiliary load-balancing loss vs auxiliary-loss-free balancing;
- token dropping or capacity factor;
- communication pattern under distributed inference.

## 3. Attention variants

Important families:

- Multi-Head Attention;
- Multi-Query Attention;
- Grouped-Query Attention;
- sliding-window attention;
- sparse attention;
- latent/compressed key-value attention;
- retrieval-augmented attention;
- memory-augmented attention.

The repository treats attention as both a model architecture question and a systems question because KV-cache design dominates long-context inference cost.

## 4. State-space and recurrent hybrids

Core traits:

- recurrence or state update replaces some or all attention operations;
- linear or near-linear sequence scaling;
- selective state update or data-dependent recurrence;
- hybridization with Transformer attention blocks.

Important comparison axes:

- expressivity under long-range dependency;
- training parallelism;
- inference memory footprint;
- stability of state updates;
- compatibility with instruction tuning and tool-use behavior.

## 5. Long-context architecture

Long-context ability can come from several layers:

- positional scaling;
- attention sparsity;
- recurrent memory;
- retrieval augmentation;
- context compression;
- KV-cache paging/offloading;
- training data and curriculum.

This repository separates architectural context length from advertised context length.

## 6. Inference architecture

Modern large-model architecture is inseparable from serving design.

Relevant mechanisms:

- KV-cache layout;
- paged attention;
- continuous batching;
- speculative decoding;
- tensor/expert/pipeline parallelism;
- quantization-aware kernels;
- prefix caching;
- constrained decoding.
