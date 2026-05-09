# Architecture Source Map

Date: 2026-05-09

This file maps each imported model family to the strongest local source available in the repository.

## Imported repositories confirmed

| Import | Local root | Upstream role | Strongest evidence |
|---|---|---|---|
| Qwen3 | `vendor/dense-transformer/qwen3/upstream` | Official release / documentation repository | README + vLLM implementation |
| DeepSeek-R1 | `vendor/reasoning/deepseek-r1/upstream` | Reasoning/post-training release repository | README + DeepSeek-V3 base architecture |
| DeepSeek-V3 | `vendor/moe/deepseek-v3/upstream` | Official model architecture / inference implementation | `inference/model.py` |
| Kimi-K2.5 | `vendor/moe/kimi-k2-5/upstream` | Official release / report repository | README architecture table |
| Mamba | `vendor/ssm-hybrid/mamba/upstream` | Official architecture implementation | `mamba_ssm/modules/mamba_simple.py`, `mamba_ssm/modules/mamba2.py` |
| vLLM | `vendor/inference/vllm/upstream` | Serving runtime and inference implementations | `vllm/model_executor/models/*.py` |
| SGLang | `vendor/inference/sglang/upstream` | Serving runtime and inference implementations | `python/sglang/srt/models/*.py` |
| GLM-4.5/4.6/4.7 | `vendor/dense-transformer/glm-4-5/upstream` | Official release / deployment repository | README + SGLang/vLLM implementation references |

## Model-family source files

### Qwen3

Official release repository:

```text
vendor/dense-transformer/qwen3/upstream/README.md
```

Runtime implementation:

```text
vendor/inference/vllm/upstream/vllm/model_executor/models/qwen3.py
```

Key classes:

```text
Qwen3Attention
Qwen3DecoderLayer
Qwen3Model
Qwen3ForCausalLM
```

Architectural notes:

- decoder-only Transformer;
- QKV tensor-parallel projection;
- GQA/KV-head separation;
- RoPE;
- QK RMSNorm before rotary embedding;
- MLP inherited from Qwen2 implementation.

### DeepSeek-R1

Official release repository:

```text
vendor/reasoning/deepseek-r1/upstream/README.md
```

R1 is not a separate base architecture implementation. The local README states that DeepSeek-R1 and R1-Zero are trained based on DeepSeek-V3-Base and points to DeepSeek-V3 for architecture details.

Architecture source:

```text
vendor/moe/deepseek-v3/upstream/inference/model.py
vendor/inference/vllm/upstream/vllm/model_executor/models/deepseek_v2.py
vendor/inference/sglang/upstream/python/sglang/srt/models/deepseek_v2.py
```

### DeepSeek-V3

Official source-level implementation:

```text
vendor/moe/deepseek-v3/upstream/inference/model.py
```

Key classes / functions:

```text
ModelArgs
ParallelEmbedding
RMSNorm
MLA
MLP
Gate
Expert
MoE
Block
Transformer
```

Architectural notes:

- sparse MoE Transformer;
- MLA attention with low-rank query and key/value projections;
- split q/k dimensions into non-positional and RoPE dimensions;
- optional naive attention cache or absorbed latent-cache path;
- dense layers before MoE layers via `n_dense_layers`;
- routed experts + shared experts;
- top-k expert routing with group-limited routing;
- BF16 / FP8 linear execution hooks;
- YaRN-style RoPE scaling.

### Kimi-K2.5

Official release repository:

```text
vendor/moe/kimi-k2-5/upstream/README.md
```

Architecture evidence currently comes from the README table:

```text
1T total parameters
32B activated parameters
61 layers
1 dense layer
hidden dimension 7168
384 experts
8 selected experts per token
1 shared expert
MLA attention
SwiGLU
MoonViT vision encoder
256K context
```

A source-level Kimi implementation has not yet been located in the imported tree.

### Mamba

Official source-level implementation:

```text
vendor/ssm-hybrid/mamba/upstream/mamba_ssm/modules/mamba_simple.py
vendor/ssm-hybrid/mamba/upstream/mamba_ssm/modules/mamba2.py
```

Key classes:

```text
Mamba
Mamba2
```

Architectural notes:

- selective SSM sequence mixer;
- depthwise causal convolution;
- input-dependent dt/B/C generation;
- diagonal state transition through `A_log`;
- learned skip parameter `D`;
- fused selective scan / chunk scan kernels;
- explicit decoding cache with convolution and SSM state.

### GLM-4.5 / GLM-4.6 / GLM-4.7

Official release repository:

```text
vendor/dense-transformer/glm-4-5/upstream/README.md
```

SGLang implementation:

```text
vendor/inference/sglang/upstream/python/sglang/srt/models/glm4_moe.py
```

Expected vLLM implementation path:

```text
vendor/inference/vllm/upstream/vllm/model_executor/models/glm4_moe_mtp.py
```

Key SGLang classes:

```text
Glm4MoeMLP
Glm4MoeAttention
Glm4MoeGate
Glm4MoeSparseMoeBlock
Glm4MoeDecoderLayer
```

Architectural notes:

- MoE reasoning/coding model family;
- QKV projection + RoPE + optional QK norm;
- sparse MoE block with routed experts and shared experts;
- DeepSeekV3-like routing method in SGLang;
- expert parallel / all-to-all backends;
- speculative / MTP-oriented deployment.

### vLLM and SGLang as architecture objects

These two imports are not merely utilities. They are part of modern LLM architecture research because they determine how frontier architectures are actually executable.

vLLM contributes:

- PagedAttention;
- continuous batching;
- prefix caching;
- quantization kernels;
- model executors for Qwen3 and DeepSeek;
- MoE and MLA inference paths.

SGLang contributes:

- RadixAttention;
- DeepSeek / GLM MoE serving implementations;
- DeepEP / all-to-all MoE dispatch;
- batch-overlap execution;
- reasoning parser / tool-call parser runtime support;
- speculative and distributed serving paths.
