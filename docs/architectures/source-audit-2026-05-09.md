# Source Audit: Imported Frontier Model Architectures

Date: 2026-05-09

This is a source-level audit of the imported architecture repositories. It deliberately separates three different objects that are often conflated:

1. **Model architecture**: the block structure, attention mechanism, MoE routing, normalization, positional encoding, and inference cache design.
2. **Post-training system**: RL, distillation, thinking-mode tuning, tool-use alignment, and agentic behavior training.
3. **Serving/runtime architecture**: batching, KV/cache layout, attention kernels, MoE dispatch, expert parallelism, speculative decoding, and parser support.

The main result is that modern frontier open-source “model releases” often do not place the full architecture implementation in the official release repository. The implementation is distributed across Hugging Face Transformers, vLLM, SGLang, and sometimes a small official inference folder.

## 0. Audited source set

Confirmed imports:

```text
vendor/dense-transformer/qwen3/
vendor/reasoning/deepseek-r1/
vendor/moe/deepseek-v3/
vendor/moe/kimi-k2-5/
vendor/ssm-hybrid/mamba/
vendor/dense-transformer/glm-4-5/
vendor/inference/vllm/
vendor/inference/sglang/
```

High-value source files:

```text
vendor/moe/deepseek-v3/upstream/inference/model.py
vendor/ssm-hybrid/mamba/upstream/mamba_ssm/modules/mamba_simple.py
vendor/ssm-hybrid/mamba/upstream/mamba_ssm/modules/mamba2.py
vendor/inference/vllm/upstream/vllm/model_executor/models/qwen3.py
vendor/inference/vllm/upstream/vllm/model_executor/models/deepseek_v2.py
vendor/inference/vllm/upstream/vllm/model_executor/models/glm4_moe_mtp.py
vendor/inference/sglang/upstream/python/sglang/srt/models/deepseek_v2.py
vendor/inference/sglang/upstream/python/sglang/srt/models/glm4_moe.py
```

## 1. DeepSeek-V3

### Source status

Evidence level: **official source-level implementation**.

Primary file:

```text
vendor/moe/deepseek-v3/upstream/inference/model.py
```

The file contains the core architecture classes:

```text
ModelArgs
ParallelEmbedding
Linear / ColumnParallelLinear / RowParallelLinear
RMSNorm
MLA
MLP
Gate
Expert
MoE
Block
Transformer
```

### Architectural skeleton

DeepSeek-V3 is a sparse MoE decoder architecture with MLA attention. Its block shape is:

```text
x -> RMSNorm -> MLA -> residual
  -> RMSNorm -> MLP or MoE -> residual
```

The feed-forward branch switches by layer id:

```python
self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
```

This means early layers can remain dense while later layers use sparse MoE routing. The default small `ModelArgs` in the inference file are not necessarily the full released model configuration, but they reveal the structural knobs: `n_layers`, `n_dense_layers`, `n_routed_experts`, `n_shared_experts`, `n_activated_experts`, expert groups, routing score function, and MLA low-rank ranks.

### MLA audit

`MLA` is the defining architectural component.

It decomposes query/key dimensions into:

```text
qk_nope_head_dim: non-positional q/k subspace
qk_rope_head_dim: rotary-positional q/k subspace
v_head_dim: value subspace
```

The query path has two modes:

```text
if q_lora_rank == 0:
    q = wq(x)
else:
    q = wq_b(q_norm(wq_a(x)))
```

The KV path is always compressed first:

```text
kv = wkv_a(x) -> [kv_lora_rank, qk_rope_head_dim]
kv_norm(kv_lora)
wkv_b(kv_lora) -> [qk_nope_head_dim + v_head_dim]
```

The architecture separates positional and non-positional channels. RoPE only touches the RoPE subspace:

```text
q_nope, q_pe = split(q)
kv, k_pe = split(wkv_a(x))
q_pe = apply_rotary_emb(q_pe)
k_pe = apply_rotary_emb(k_pe)
```

The implementation has two attention-cache regimes:

1. `naive`: materialize full K/V cache after expanding compressed KV.
2. `absorb`: cache compressed latent KV and positional key separately, then absorb the `wkv_b` projection into attention computation.

The `absorb` path is architecturally important because it shows the practical meaning of MLA: reduce KV-cache footprint by storing latent KV rather than fully expanded per-head K/V.

### MoE audit

`MoE` contains:

```text
Gate
local Expert list
shared_experts
```

The gate computes scores by a learned router matrix. It supports softmax or sigmoid scoring, group-limited routing, top-k expert selection, and route scaling. If `n_groups > 1`, scores are reshaped by group, only top groups are retained, and token-level experts are selected inside that restricted expert set.

Each expert is a SwiGLU-style MLP:

```text
w2(silu(w1(x)) * w3(x))
```

The final MoE output is:

```text
sum(weighted routed expert outputs) + shared_experts(x)
```

The shared expert is not a router choice; it is always applied. This is important because the model combines sparse conditional capacity with a dense shared pathway.

### Quantization / systems hooks

The official inference implementation includes BF16/FP8 linear execution hooks. `linear()` checks whether weights are quantized and dispatches either standard `F.linear`, dequantized BF16 GEMM, or FP8 GEMM.

This is not just an optimization detail. It affects how architecture is deployable: the architecture assumes expert and projection matrices can be served under low-precision kernels without changing high-level block semantics.

### Verdict

DeepSeek-V3 is the strongest imported source for studying modern sparse MoE + MLA. It is the base architecture needed to understand DeepSeek-R1. The model’s novelty lies less in the residual skeleton, which remains Transformer-like, and more in the combination of:

- MLA compressed KV attention;
- sparse routed experts;
- shared experts;
- group-limited top-k routing;
- dense-to-MoE layer transition;
- long-context RoPE scaling;
- FP8-aware inference path.

## 2. DeepSeek-R1

### Source status

Evidence level: **post-training release repository + DeepSeek-V3 architecture dependency**.

Primary file:

```text
vendor/reasoning/deepseek-r1/upstream/README.md
```

The README states that DeepSeek-R1 and DeepSeek-R1-Zero are trained based on DeepSeek-V3-Base. It points readers to DeepSeek-V3 for architectural details.

### Architecture audit

There is no evidence that R1 introduces a new low-level block architecture distinct from DeepSeek-V3. It should be audited as:

```text
DeepSeek-R1 = DeepSeek-V3 base architecture + reasoning-oriented post-training pipeline
```

The R1 repository is therefore primarily evidence for:

- RL without preliminary SFT in R1-Zero;
- cold-start data + RL in R1;
- reasoning behavior emergence;
- distillation into Qwen/Llama dense models;
- usage recommendations and prompting constraints.

### Verdict

R1 is architecturally downstream of V3. Treating it as a separate base architecture would be misleading. For architecture study, always route R1 claims back to:

```text
vendor/moe/deepseek-v3/upstream/inference/model.py
vendor/inference/vllm/upstream/vllm/model_executor/models/deepseek_v2.py
vendor/inference/sglang/upstream/python/sglang/srt/models/deepseek_v2.py
```

## 3. SGLang DeepSeek implementation

### Source status

Evidence level: **runtime implementation-level architecture**.

Primary file:

```text
vendor/inference/sglang/upstream/python/sglang/srt/models/deepseek_v2.py
```

Although named `deepseek_v2.py`, this file has evolved into a broad DeepSeek serving implementation. It contains DeepSeek V2/V3/V4-related branches, NSA support, DeepEP routing, HashTopK, and high-performance MoE dispatch paths.

### Architecture/runtime features

Important observed elements:

```text
DeepseekV2MLP
MoEGate
DeepseekV2MoE
Deepseek MLA forward mixins
NSA indexer support
DeepEP / all-to-all MoE dispatch
HashTopK / TopK routing
shared expert fusion
FP8/FP4/MoE backend specialization
```

The `MoEGate` class contains an `is_deepseek_v4` flag. When enabled, the routing path can switch away from grouped top-k toward a different scoring/layout branch. This is an important sign that the serving ecosystem is already preparing for, or supporting, post-V3 DeepSeek variants even when a clean official `DeepSeek-V4` source repository is not visible.

### Difference from official DeepSeek-V3 inference code

The official DeepSeek-V3 `inference/model.py` is cleaner and pedagogically better. SGLang’s implementation is operationally richer. It encodes:

- expert-parallel dispatch;
- hardware-specific router GEMM;
- DeepEP backend decisions;
- shared expert fusion;
- batch-overlap execution;
- deterministic inference modes;
- speculative / NSA context paths.

### Verdict

For understanding the mathematical architecture, read official DeepSeek-V3 first. For understanding how the architecture is actually made performant, read SGLang and vLLM.

## 4. Qwen3

### Source status

Evidence level: **release repository + vLLM implementation-level source**.

Primary release path:

```text
vendor/dense-transformer/qwen3/upstream/README.md
```

Primary implementation path:

```text
vendor/inference/vllm/upstream/vllm/model_executor/models/qwen3.py
```

### Architecture audit from vLLM

vLLM defines:

```text
Qwen3Attention
Qwen3DecoderLayer
Qwen3Model
Qwen3ForCausalLM
```

Qwen3’s vLLM implementation is recognizably decoder-only Transformer architecture. The attention module uses:

```text
QKVParallelLinear
RowParallelLinear
RoPE
Attention / EncoderOnlyAttention
RMSNorm q_norm
RMSNorm k_norm
```

The most visible architecture-specific distinction is QK normalization. The forward path computes Q/K/V, reshapes q and k by head, applies per-head RMSNorm to q and k, reshapes them back, then applies rotary embedding.

The decoder layer contains:

```text
input_layernorm
self_attn
post_attention_layernorm
Qwen3MLP = Qwen2MLP
```

Thus Qwen3, at least in this implementation, is a conservative modern decoder Transformer with an important QK-norm refinement rather than a radical block replacement.

### Official repository role

The official Qwen3 repository is valuable for model-family metadata: dense/MoE variants, Qwen3-2507 updates, thinking/non-thinking modes, long-context deployment, supported frameworks. It is not, in the imported snapshot, the strongest source for actual model class implementation.

### Verdict

Qwen3 should be studied as a modern Qwen2-lineage decoder architecture with QK norm and long-context/reasoning release variants. The actual source-level audit should rely on vLLM and, later, Hugging Face Transformers if imported.

## 5. Kimi-K2.5

### Source status

Evidence level: **release-note architecture only**.

Primary file:

```text
vendor/moe/kimi-k2-5/upstream/README.md
```

### Architecture facts available

The README provides a unusually complete summary table:

```text
Architecture: MoE
Total parameters: 1T
Activated parameters: 32B
Layers: 61
Dense layers: 1
Attention hidden dimension: 7168
MoE hidden dimension per expert: 2048
Attention heads: 64
Experts: 384
Selected experts per token: 8
Shared experts: 1
Vocabulary: 160K
Context length: 256K
Attention mechanism: MLA
Activation: SwiGLU
Vision encoder: MoonViT
Vision encoder parameters: 400M
```

### Audit limitation

No local source-level implementation has yet been located. Therefore we cannot audit the actual layer class, attention cache, routing code, or multimodal fusion mechanism from imported source.

### Architecture inference, with caveat

The published table suggests a DeepSeek-style large sparse MoE design: huge total capacity, small active parameter budget, MLA attention, shared expert, and one dense layer before the MoE stack. The hidden size 7168 and MLA indicate architectural proximity to the current Chinese frontier MoE lineage, but without source we should not infer exact code structure.

### Verdict

Kimi-K2.5 is architecturally important but not yet source-auditable in this repository. It should remain marked as release-level evidence until a runtime implementation is imported or located.

## 6. GLM-4.5 / GLM-4.6 / GLM-4.7

### Source status

Evidence level: **release repository + SGLang/vLLM implementation-level source**.

Release path:

```text
vendor/dense-transformer/glm-4-5/upstream/README.md
```

SGLang source:

```text
vendor/inference/sglang/upstream/python/sglang/srt/models/glm4_moe.py
```

vLLM MTP source:

```text
vendor/inference/vllm/upstream/vllm/model_executor/models/glm4_moe_mtp.py
```

### SGLang GLM MoE audit

SGLang’s `glm4_moe.py` declares itself as an inference-only GLM-4.5, GLM-4.6, GLM-4.7 implementation compatible with Hugging Face weights.

Key classes:

```text
Glm4MoeMLP
Glm4MoeAttention
Glm4MoeGate
Glm4MoeSparseMoeBlock
Glm4MoeDecoderLayer
```

Attention uses:

```text
QKVParallelLinear
RowParallelLinear
RoPE
RadixAttention
optional q_norm/k_norm
```

The sparse MoE block uses:

```text
Glm4MoeGate
TopK
get_moe_impl_class(quant_config)
shared_experts
routed_scaling_factor
DeepSeekV3 routing method
expert-parallel all-to-all backends
```

This means GLM-4.x serving architecture is converging on the same family of primitives as DeepSeek-style models: grouped routing, shared experts, correction bias, expert parallelism, quantization-aware experts, and specialized all-to-all backends.

### vLLM MTP audit

`glm4_moe_mtp.py` implements GLM-4.5/4.6/4.7 MTP compatible with Hugging Face weights.

The MTP layer contains:

```text
enorm: RMSNorm over current input embedding
hnorm: RMSNorm over previous hidden state
eh_proj: Linear(hidden_size * 2 -> hidden_size)
mtp_block: Glm4MoeDecoderLayer
shared_head: RMSNorm + ParallelLMHead
```

The forward path:

```text
inputs_embeds -> enorm
previous_hidden_states -> hnorm
concat -> eh_proj
-> Glm4MoeDecoderLayer
-> residual add
```

This is a speculative decoding / multi-token prediction architecture. It is not just a decoding trick outside the model; it adds auxiliary prediction layers that reuse the model’s MoE decoder layer structure.

### Release-level architecture

The GLM README states:

```text
GLM-4.5: 355B total / 32B active
GLM-4.5-Air: 106B total / 12B active
GLM-4.6: context expanded from 128K to 200K
GLM-4.7: coding, tool use, interleaved thinking, preserved thinking, turn-level thinking
GLM-4.7-Flash: 30B-A3B
```

### Verdict

GLM-4.x is best understood as a sparse MoE agentic reasoning/coding family with a runtime architecture strongly oriented toward:

- thinking modes;
- tool-call execution;
- expert parallel MoE serving;
- speculative/MTP acceleration;
- long-context coding-agent workloads.

Its distinctive feature in the imported source is not a radically new attention mechanism, but the integration of MoE execution and MTP into a production serving stack.

## 7. Mamba / Mamba2

### Source status

Evidence level: **official source-level implementation**.

Primary files:

```text
vendor/ssm-hybrid/mamba/upstream/mamba_ssm/modules/mamba_simple.py
vendor/ssm-hybrid/mamba/upstream/mamba_ssm/modules/mamba2.py
```

### Mamba v1 audit

`Mamba` contains:

```text
in_proj: d_model -> 2 * d_inner
conv1d: depthwise causal convolution
x_proj: d_inner -> dt_rank + 2 * d_state
dt_proj: dt_rank -> d_inner
A_log: log diagonal transition parameter
D: learned skip
out_proj: d_inner -> d_model
```

The forward path splits projected input into `x` and `z`, applies causal conv to `x`, generates `dt`, `B`, and `C`, and runs selective scan. The gate `z` modulates the output. During decoding, it maintains two state caches:

```text
conv_state
ssm_state
```

This is the core substitution for attention: sequence mixing happens through selective state updates rather than pairwise token attention.

### Mamba2 audit

`Mamba2` changes the structure substantially:

```text
projection order: [z, x, B, C, dt]
headdim
nheads
groups
chunk_size
RMSNormGated
mamba_split_conv1d_scan_combined
mamba_chunk_scan_combined
sequence parallel support
```

Mamba2’s state layout becomes head/group aware:

```text
ssm_state: [batch, nheads, headdim, d_state]
```

It also allows `d_ssm` to be smaller than the expanded inner dimension, with the remainder acting like a gated MLP path. This turns the block into a more flexible hybrid between SSM sequence mixing and gated feed-forward computation.

### Verdict

Mamba is the cleanest non-Transformer source in the archive. Its audit value is high because it gives an actual alternative to attention/KV-cache dominated architecture. The price is kernel dependence: the mathematical block is clear, but high performance depends on fused scan and causal conv kernels.

## 8. vLLM

### Source status

Evidence level: **serving/runtime architecture**.

Primary path:

```text
vendor/inference/vllm/upstream/
```

### Runtime architecture

vLLM is not a model, but it determines which architectures are practically executable at scale. Its README emphasizes:

```text
PagedAttention
continuous batching
chunked prefill
prefix caching
CUDA/HIP graphs
FP8/MXFP8/MXFP4/NVFP4/INT8/INT4 quantization
FlashAttention / FlashInfer / FlashMLA / Triton kernels
optimized GEMM/MoE kernels
speculative decoding
expert/tensor/pipeline/context parallelism
```

The imported tree also contains actual model executors for Qwen3 and DeepSeek, which makes it both a serving framework and a source of architecture implementations.

### Verdict

Modern LLM architecture cannot be audited only at the model block level. vLLM shows that attention form, KV-cache layout, quantization, parallelism, and speculative decoding are now architecture-level constraints.

## 9. SGLang

### Source status

Evidence level: **serving/runtime architecture**.

Primary path:

```text
vendor/inference/sglang/upstream/
```

### Runtime architecture

SGLang is especially important for DeepSeek/GLM-style MoE models. Its source contains:

```text
RadixAttention
ForwardBatch
DeepEP / MoE all-to-all dispatch
TopK / HashTopK
shared expert fusion
batch overlap
expert distribution recording
reasoning and tool-call parser integration
```

### Verdict

SGLang is currently the most useful imported source for studying frontier MoE serving. It exposes where model architecture ends and runtime architecture begins: the MoE block, router, top-k selection, expert dispatch, shared expert execution, and speculative execution path are fused into one operational design.

## 10. Comparative judgment

### Most source-auditable

```text
1. DeepSeek-V3
2. Mamba / Mamba2
3. GLM via SGLang + vLLM MTP
4. Qwen3 via vLLM
5. DeepSeek via SGLang/vLLM
```

### Most release-level only

```text
1. Kimi-K2.5
2. DeepSeek-R1 as a post-training release
3. Qwen3 official repo, unless paired with vLLM/Transformers
4. GLM official repo, unless paired with SGLang/vLLM/Transformers
```

### Architectural trends visible across sources

1. Sparse MoE is now central for frontier open-weight models.
2. Shared experts are common; sparse routing is rarely purely sparse.
3. MLA / latent KV compression is a major answer to KV-cache pressure.
4. QK normalization appears as a stabilizing refinement in modern decoder models.
5. Serving frameworks have become architecture repositories in practice.
6. Speculative decoding / MTP is becoming structurally integrated rather than merely a sampling trick.
7. Long context is not a single mechanism; it is a combination of RoPE scaling, cache layout, runtime memory management, and training/release policy.

## 11. Next audit tasks

1. Add a DeepSeek-V3 focused report: MLA equations, cache regimes, gate math, MoE routing path.
2. Add a GLM focused report: SGLang `Glm4MoeSparseMoeBlock` and vLLM `Glm4MoeMultiTokenPredictor`.
3. Add a Mamba focused report: compare Mamba v1 and Mamba2 state layout and scan semantics.
4. Add a runtime report: vLLM PagedAttention vs SGLang RadixAttention / DeepEP.
5. Locate or import Kimi-K2.5 implementation if available; otherwise keep it explicitly marked as non-source-audited.
