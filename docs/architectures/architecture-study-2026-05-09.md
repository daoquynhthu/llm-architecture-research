# Architecture Study: Imported Frontier Model Repositories

Date: 2026-05-09

This note studies the first six imported repositories:

- Qwen3
- DeepSeek-R1
- Kimi-K2.5
- Mamba
- vLLM
- GLM-4.5 / GLM-4.6 / GLM-4.7 repository

The main methodological point is that these imports are not homogeneous. Some repositories contain actual model implementation code; some are release / documentation repositories; some expose architecture indirectly through serving frameworks such as vLLM, SGLang, or Transformers. Therefore this study uses three evidence levels:

1. **Source-level architecture**: core modules/classes are present in the imported repository.
2. **Runtime implementation-level architecture**: model implementation is available in an imported runtime such as vLLM.
3. **Release-note architecture**: architectural parameters are stated in README/report files, but the full model implementation lives elsewhere.

## 1. Qwen3

Imported path:

```text
vendor/dense-transformer/qwen3/upstream/
```

Import record:

```text
vendor/dense-transformer/qwen3/IMPORT.md
```

Qwen3 was imported from `https://github.com/QwenLM/Qwen3.git` at commit `7a2f61ffc7a20d47efcd2bf97f6f2bf52729042e`.

### Evidence level

Qwen3 in this archive is primarily a **release / documentation repository**. The imported README describes the Qwen3 family, model releases, deployment routes, context capabilities, and supported inference frameworks. It does not, in the imported tree examined so far, expose a full native `modeling_qwen3.py` style architecture implementation.

The README states that Qwen3 includes dense and Mixture-of-Experts models, with sizes including 0.6B, 1.7B, 4B, 8B, 14B, 32B and MoE variants such as 30B-A3B and 235B-A22B. It also describes Qwen3-2507 as having 256K context understanding extendable to 1M tokens.

### Runtime architecture via vLLM

The actual inference-level Qwen3 implementation is present in the imported vLLM tree:

```text
vendor/inference/vllm/upstream/vllm/model_executor/models/qwen3.py
```

That file defines `Qwen3Attention`, `Qwen3DecoderLayer`, `Qwen3Model`, and `Qwen3ForCausalLM`.

The implementation shows Qwen3 as a decoder-only Transformer family with:

- tensor-parallel QKV projection through `QKVParallelLinear`;
- grouped-query style separation between total attention heads and key/value heads;
- RoPE via `get_rope`;
- per-head `q_norm` and `k_norm` RMSNorm before rotary embedding;
- an MLP inherited as `Qwen2MLP`;
- RMSNorm pre-attention and post-attention;
- LM head through `ParallelLMHead` unless weights are tied.

The key architectural difference visible in the vLLM implementation is QK normalization. Inside `Qwen3Attention.forward`, q and k are reshaped by head, normalized by `RMSNorm`, reshaped back, then rotary embedding is applied.

### Interpretation

Qwen3 should be treated as a modern decoder-only Transformer family with both dense and MoE variants. In this archive, the official Qwen3 repository is useful for release taxonomy and model-family metadata, while vLLM gives the more actionable architecture skeleton. For future study, import or reference Hugging Face Transformers' Qwen3 implementation if a full training/inference model class is needed outside vLLM.

## 2. DeepSeek-R1

Imported path:

```text
vendor/reasoning/deepseek-r1/upstream/
```

Import record:

```text
vendor/reasoning/deepseek-r1/IMPORT.md
```

DeepSeek-R1 was imported from `https://github.com/deepseek-ai/DeepSeek-R1.git` at commit `0cf78561f1d51c84a21b2190626b21116d5c68bb`.

### Evidence level

DeepSeek-R1 is primarily a **post-training / reasoning-model release repository**, not a separate new base architecture implementation repository. Its README explicitly states that DeepSeek-R1 and DeepSeek-R1-Zero are trained based on DeepSeek-V3-Base and that architectural details should be obtained from the DeepSeek-V3 repository.

The README states:

- DeepSeek-R1-Zero is produced by large-scale RL without preliminary SFT;
- DeepSeek-R1 adds cold-start data before RL;
- DeepSeek-R1 and R1-Zero are 671B total / 37B activated MoE models with 128K context;
- distilled variants are based on Qwen2.5 and Llama model families.

### Runtime architecture via vLLM

The imported vLLM tree contains the relevant DeepSeek implementation:

```text
vendor/inference/vllm/upstream/vllm/model_executor/models/deepseek_v2.py
```

The file describes itself as an inference-only DeepSeekV2/DeepSeekV3 model implementation. Architecturally, it is the most important local source for DeepSeek-style models.

Visible components include:

- `DeepseekV2MoE` with routed experts, shared experts, grouped top-k routing, expert parallelism, redundant expert support, and optional auxiliary-loss-free / correction-bias routing path;
- `DeepseekV2Attention`, which implements MLA-like low-rank query/key/value projections with separate `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, `q_lora_rank`, and `kv_lora_rank`;
- YaRN / long-context scaling helpers;
- vLLM-specific cache specifications for MLA attention;
- specialized indexer paths for DeepSeek-V3.2-style sparse attention indexing.

### Interpretation

DeepSeek-R1 should not be studied as an independent architecture in isolation. Its architecture is essentially DeepSeek-V3-style sparse MoE + MLA, with R1 being a reasoning/post-training variant. Therefore the archive should treat:

```text
DeepSeek-R1 = DeepSeek-V3 architecture + reasoning-oriented post-training pipeline
```

The key research object is not a new block type, but the interaction between:

- large sparse MoE capacity;
- MLA/compressed KV attention;
- long-context scaling;
- RL-derived reasoning behavior;
- distilled dense derivatives.

## 3. Kimi-K2.5

Imported path:

```text
vendor/moe/kimi-k2-5/upstream/
```

Import record:

```text
vendor/moe/kimi-k2-5/IMPORT.md
```

Kimi-K2.5 was imported from `https://github.com/MoonshotAI/Kimi-K2.5.git` at commit `3e60763b943e93c443287c383e0468ffe05b188f`.

### Evidence level

Kimi-K2.5 provides strong **release-note architecture evidence**. Its README contains a detailed architecture table.

### Architecture summary

According to the imported README, Kimi-K2.5 is:

- a native multimodal agentic model;
- built by continual pretraining on approximately 15T mixed visual and text tokens atop Kimi-K2-Base;
- a Mixture-of-Experts model;
- 1T total parameters;
- 32B activated parameters;
- 61 layers including one dense layer;
- hidden dimension 7168;
- per-expert MoE hidden dimension 2048;
- 64 attention heads;
- 384 experts;
- 8 selected experts per token;
- 1 shared expert;
- 160K vocabulary;
- 256K context length;
- MLA attention;
- SwiGLU activation;
- MoonViT vision encoder with approximately 400M parameters.

### Interpretation

Kimi-K2.5 is one of the more important architecture targets in the current archive because it combines several frontier tendencies:

1. very large sparse MoE total capacity;
2. relatively modest activated-parameter budget;
3. MLA-style attention rather than plain MHA/GQA;
4. native multimodal pretraining rather than post-hoc vision adapter only;
5. agentic execution and tool-use optimization as a first-class design goal.

The repository, however, currently functions more as a release/report repository than as a full source-level implementation. The architecture table is useful, but the next step is to locate runtime implementations in vLLM/SGLang/Transformers or Moonshot-provided inference code.

## 4. Mamba

Imported path:

```text
vendor/ssm-hybrid/mamba/upstream/
```

Import record:

```text
vendor/ssm-hybrid/mamba/IMPORT.md
```

Mamba was imported from `https://github.com/state-spaces/mamba.git` at commit `48582f0592f1d55b441cd963aa5f9226b083f02a`.

### Evidence level

Mamba provides **source-level architecture evidence**. The relevant files are present in the archive:

```text
vendor/ssm-hybrid/mamba/upstream/mamba_ssm/modules/mamba_simple.py
vendor/ssm-hybrid/mamba/upstream/mamba_ssm/modules/mamba2.py
```

### Mamba v1 architecture

`mamba_simple.py` defines the `Mamba` module. Key components include:

- `in_proj`: projects `d_model` into `2 * d_inner`, split into the state stream and gating stream;
- depthwise causal convolution over the inner dimension;
- `x_proj`: generates `dt`, `B`, and `C` parameters;
- `dt_proj`: maps low-rank `dt` representation to per-channel time steps;
- `A_log`: log-parameterized diagonal state transition;
- `D`: learned skip parameter;
- `selective_scan_fn` / `mamba_inner_fn`: fused selective scan execution;
- explicit inference cache with `conv_state` and `ssm_state`;
- token-by-token `step` path for autoregressive decoding.

This is a replacement for attention-based sequence mixing. Sequence information flows through data-dependent state transitions rather than attention over all prior tokens.

### Mamba2 architecture

`mamba2.py` defines `Mamba2`, which generalizes the module:

- larger default state dimension (`d_state=128`);
- head dimension `headdim=64` and `nheads = d_ssm / headdim`;
- group structure through `ngroups`;
- combined projection layout `[z, x, B, C, dt]`;
- memory-efficient fused path through `mamba_split_conv1d_scan_combined`;
- chunked scan through `mamba_chunk_scan_combined`;
- optional gated RMSNorm;
- optional tensor/sequence parallel support;
- explicit cache layout for convolution state and SSM state.

### Interpretation

Mamba is not merely another LLM release; it is a different sequence-mixing primitive. Its architectural significance is that it challenges the assumption that long-context modeling must be attention/KV-cache dominated. Compared with Transformer attention, Mamba shifts cost into recurrent state update and selective scan kernels.

In this archive, Mamba is currently the cleanest source-level architecture import.

## 5. vLLM

Imported path:

```text
vendor/inference/vllm/upstream/
```

Import record:

```text
vendor/inference/vllm/IMPORT.md
```

vLLM was imported from `https://github.com/vllm-project/vllm.git` at commit `d6563d693c817ec762ccd0bb6c44c56b4f3dc4a3`.

### Evidence level

vLLM is not a model family, but a **serving architecture / runtime architecture** repository. It is central to this research archive because many frontier model architectures are practically realized and studied through serving implementations.

The README characterizes vLLM as a fast LLM inference and serving library with PagedAttention, continuous batching, chunked prefill, prefix caching, CUDA/HIP graph execution, quantization, optimized attention kernels, optimized GEMM/MoE kernels, speculative decoding, torch.compile transformations, and disaggregated prefill/decode/encode.

### Architecture role

For this archive, vLLM has three distinct roles:

1. **Runtime substrate**: It determines how model architectures actually execute under batching, KV-cache pressure, quantization, and tensor/expert parallelism.
2. **Reference implementation source**: It contains inference-only implementations of Qwen3, DeepSeekV2/V3, GLM-like models, Mamba-like models, and many others.
3. **Systems architecture object**: Its own architecture—PagedAttention, continuous batching, speculative decoding, prefix caching, distributed parallelism—is itself part of modern LLM architecture research.

### Important local files already observed

```text
vllm/model_executor/models/qwen3.py
vllm/model_executor/models/deepseek_v2.py
```

These files show that vLLM is not merely a serving wrapper. It exposes architecture-level implementation decisions such as QK norm in Qwen3, MLA for DeepSeek, FusedMoE routing, KV/MLA cache specification, and quantized kernel pathways.

## 6. GLM-4.5 / GLM-4.6 / GLM-4.7 repository

Imported path:

```text
vendor/dense-transformer/glm-4-5/upstream/
```

Import record:

```text
vendor/dense-transformer/glm-4-5/IMPORT.md
```

GLM-4.5 was imported from `https://github.com/zai-org/GLM-4.5.git` at commit `170f20b2c10659008fdbc909d478bc2a75bc3627`.

### Evidence level

This is primarily a **release / deployment repository** with architecture information in README and external implementation references.

The README now covers GLM-4.7, GLM-4.6, and GLM-4.5. It states:

- GLM-4.5 has 355B total parameters and 32B active parameters;
- GLM-4.5-Air has 106B total and 12B active parameters;
- both are hybrid reasoning models with thinking and non-thinking modes;
- GLM-4.6 expands context from 128K to 200K;
- GLM-4.7 introduces / improves interleaved thinking, preserved thinking, and turn-level thinking;
- GLM-4.7-Flash is a 30B-A3B lightweight model;
- GLM-4.5/4.6/4.7 model code is referred to implementations in Transformers, vLLM, and SGLang.

### Interpretation

GLM in this archive should be studied as an agentic reasoning / coding model family with sparse active-parameter structure and inference features such as MTP/speculative decoding. The imported repository itself does not appear to be the canonical source-level implementation. It points to framework-level implementations:

- Transformers `glm4_moe`;
- vLLM `glm4_moe_mtp.py`;
- SGLang `glm4_moe.py`;
- lite model variants in corresponding `glm4_moe_lite` implementations.

Therefore, future architecture study should import or inspect those implementation files. At present the strongest local evidence is the README-level architecture and deployment documentation.

## Cross-model comparison

| Target | Evidence level | Core architecture | Key research value |
|---|---|---|---|
| Qwen3 | Release + vLLM runtime implementation | Decoder-only Transformer; dense and MoE family; QK-norm visible in vLLM | Modern general LLM family, long-context, thinking/non-thinking split |
| DeepSeek-R1 | Release + vLLM DeepSeek runtime implementation | DeepSeek-V3-style MoE + MLA; R1 is post-training/reasoning variant | RL-induced reasoning on sparse MoE base; distillation pipeline |
| Kimi-K2.5 | Release architecture table | 1T MoE, 32B active, MLA, MoonViT multimodal encoder | Native multimodal agentic MoE with large sparse capacity |
| Mamba | Source-level implementation | Selective SSM / SSM-hybrid; Mamba and Mamba2 modules | Non-attention sequence mixing; state-space alternative to KV-cache dominance |
| vLLM | Source-level serving architecture | PagedAttention, continuous batching, optimized kernels, model executors | Execution substrate for frontier architectures |
| GLM-4.5/4.7 | Release + external implementation references | Sparse active-parameter agentic reasoning/coding family | Thinking modes, MTP/speculative inference, coding-agent orientation |

## Immediate next steps

1. Import `sglang` and inspect its model files for GLM, Qwen, DeepSeek, and Kimi support.
2. Import Hugging Face `transformers` or selectively mirror the relevant model subtrees: `qwen3`, `glm4_moe`, DeepSeek configs if present.
3. Import DeepSeek-V3 directly if the goal is source-level study of R1's base architecture, because R1 explicitly points back to V3 for architecture.
4. Add a `docs/architectures/source-map.md` that maps each model family to the actual local files where architecture is implemented.
5. For Mamba, begin formal block-level notes because the source code is complete enough for a real architectural audit.
