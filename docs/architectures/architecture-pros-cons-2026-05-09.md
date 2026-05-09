# Architecture Pros and Cons: Imported Frontier Model Sources

Date: 2026-05-09

This document evaluates the architectural strengths and weaknesses of the imported model/source families. It follows the source audit in `source-audit-2026-05-09.md` and keeps the same evidence discipline: source-level conclusions are made only where implementation files exist locally.

## Evaluation criteria

The comparison uses seven criteria:

1. **Expressive capacity**: how much modeling capacity the architecture can expose per token.
2. **Long-context economics**: how efficiently the architecture handles long context, especially KV/cache pressure.
3. **Training complexity**: how difficult the architecture is to train stably and efficiently.
4. **Inference complexity**: how difficult it is to serve with high throughput and low latency.
5. **Scalability**: whether the architecture scales cleanly across tensor, expert, pipeline, data, or context parallelism.
6. **Robustness / failure modes**: expected architectural fragilities.
7. **Auditability**: how transparent and locally inspectable the source is.

## 1. DeepSeek-V3

Primary source:

```text
vendor/moe/deepseek-v3/upstream/inference/model.py
```

Architecture type:

```text
Sparse MoE decoder Transformer + MLA + shared experts + group-limited routing
```

### Advantages

DeepSeek-V3 has one of the strongest architectures in the current archive because it attacks the two dominant cost centers of frontier LLMs at once: parameter scaling and long-context KV-cache pressure.

The sparse MoE design gives it high total capacity without activating all parameters per token. Routed experts let the model store heterogeneous sub-capabilities across expert weights, while shared experts preserve a dense always-on pathway. This avoids the brittle pure-sparse design where every token must rely entirely on router choices. The shared expert path is a practical architectural hedge: common transformations remain available to all tokens, while specialized transformations are conditionally activated.

MLA is the most important attention-side advantage. By compressing KV through a latent representation and splitting positional/non-positional qk dimensions, DeepSeek-V3 reduces the cost of storing and using attention history. The `absorb` path in the official inference implementation is especially meaningful: it shows that MLA is not just a paper-level factorization, but a cache-layout strategy. Long context becomes less dominated by full per-head K/V expansion.

The architecture also has a clean dense-to-MoE transition. Early dense layers can stabilize low-level token and syntax processing, while later sparse MoE layers carry broader semantic and task-specific capacity. This is a better structural compromise than making every layer sparse.

From an engineering perspective, the code is unusually auditable for a frontier architecture. The official file exposes the whole conceptual stack: `ModelArgs`, `MLA`, `Gate`, `Expert`, `MoE`, `Block`, and `Transformer`.

### Disadvantages

The architecture is complex in exactly the places where failures are hardest to diagnose: router dynamics, expert utilization, long-context cache semantics, and low-precision execution.

MoE routing introduces a second model inside the model. If the gate learns pathological distributions, expert collapse, underutilization, load imbalance, or domain-specific routing brittleness can occur. Group-limited top-k routing improves scalability, but also makes routing a constrained optimization problem. Some experts become inaccessible unless the correct group is selected first.

MLA reduces KV pressure but increases implementation subtlety. The model no longer has the simple mental model of full K and V caches per layer. Correctness now depends on whether latent cache, positional cache, projection absorption, RoPE split, and softmax scaling all remain aligned. This is a higher audit burden than standard GQA/MQA.

The architecture is also strongly coupled to high-performance kernels. FP8 hooks, expert dispatch, and absorbed MLA cache paths are part of its practical feasibility. A naive implementation can preserve semantics but fail economically. Thus DeepSeek-V3 is architecturally elegant but runtime-dependent.

### Best use case

DeepSeek-V3 is best when the goal is maximum capability per activated parameter under long-context and large-scale serving constraints.

### Main risk

The main risk is operational opacity: much of the real behavior depends on router/expert dynamics and serving-kernel correctness rather than on the visible residual block alone.

## 2. DeepSeek-R1

Primary source:

```text
vendor/reasoning/deepseek-r1/upstream/README.md
```

Architecture type:

```text
DeepSeek-V3 base architecture + reasoning/post-training system
```

### Advantages

DeepSeek-R1’s strength is not a new block architecture. Its strength is the demonstration that a strong sparse MoE base can be pushed into reasoning behavior through RL and cold-start/post-training design.

Architecturally, this is a useful separation. It implies that reasoning capability does not necessarily require a new attention primitive or a new residual block. A sufficiently strong base architecture, when paired with the right post-training regime, can express long chain-of-thought, self-verification, and reflective behavior.

The distilled model line is also significant. It suggests that reasoning behavior can be transferred into smaller dense models, though not without changing tokenization/config details.

### Disadvantages

R1 is weak as an architecture source. It does not give a new low-level design to audit. Its architecture inherits DeepSeek-V3’s strengths and weaknesses, while adding a separate post-training layer that is only partially visible from the release repository.

This creates a common interpretive trap: attributing all R1 behavior to architecture when much of it is actually training and sampling policy. The model may be architecturally V3-like while behaviorally very different due to RL, prompt conventions, and decoding policy.

### Best use case

R1 is best studied as an example of how post-training changes the behavioral regime of a sparse MoE model.

### Main risk

The main risk is conceptual misclassification: treating R1 as an architecture when it is primarily a post-training / reasoning release on top of V3.

## 3. Qwen3

Primary implementation source:

```text
vendor/inference/vllm/upstream/vllm/model_executor/models/qwen3.py
```

Architecture type:

```text
Modern decoder-only Transformer family with QK norm, RoPE, GQA/KV-head separation
```

### Advantages

Qwen3’s architectural advantage is conservatism with targeted refinement. It remains within the decoder-only Transformer lineage, which means tooling, inference, fine-tuning, quantization, and deployment are easier than for more exotic architectures.

The visible vLLM implementation shows a clean attention path: QKV projection, head reshaping, Q/K RMSNorm, RoPE, attention, output projection. QK normalization is the key refinement. It can improve attention stability by controlling query/key scale before rotary embedding and dot-product attention. This is especially relevant for long-context or high-capacity models where attention logits can become unstable.

The use of GQA/KV-head separation is practical. It reduces KV-cache cost relative to full multi-head attention without departing from standard attention semantics as radically as MLA.

Qwen3 also benefits from ecosystem maturity. Its architecture fits vLLM, SGLang, Transformers, llama.cpp, and common training stacks more naturally than highly custom MoE/MLA systems.

### Disadvantages

Qwen3’s weakness is that it is less architecturally radical. Standard Transformer attention remains the core sequence mixer, so long-context economics still depend heavily on KV-cache management, RoPE scaling, serving framework tricks, and runtime memory systems.

Compared with DeepSeek-V3 or Kimi-K2.5-style large sparse MoE, a dense Qwen3 model has less total parameter capacity per activated FLOP. MoE variants mitigate this, but the imported source evidence for Qwen3 MoE is weaker than the DeepSeek-V3 official implementation.

Another weakness is source fragmentation. The official Qwen3 repository is release-oriented; the model implementation has to be studied through vLLM or other frameworks. This makes source-level audit less direct.

### Best use case

Qwen3 is best when deployment compatibility, simplicity, and stable Transformer behavior matter more than maximal architectural novelty.

### Main risk

The main risk is that long-context and reasoning improvements may be over-attributed to architecture when they also depend on training data, post-training, chat templates, and runtime support.

## 4. Kimi-K2.5

Primary source:

```text
vendor/moe/kimi-k2-5/upstream/README.md
```

Architecture type:

```text
Release-described 1T total / 32B activated multimodal MoE with MLA and MoonViT
```

### Advantages

Kimi-K2.5 is architecturally ambitious. The release table describes a 1T-parameter MoE with only 32B activated parameters, 384 experts, 8 selected experts per token, one shared expert, MLA attention, SwiGLU, 256K context, and a MoonViT vision encoder. This is a clear frontier design pattern: very large sparse capacity, manageable activated compute, latent attention compression, and native multimodal input.

The inclusion of a vision encoder as part of the stated architecture is important. Unlike text-only MoE models with bolt-on visual adapters, Kimi-K2.5 is described as a native multimodal agentic model. If the implementation matches the release claims, its architecture is designed for visual grounding, coding from visual specifications, video/image reasoning, and tool use.

The agentic orientation is also notable. Its architecture is not only built for next-token prediction, but for long-horizon tool-mediated execution. This points toward a broader architecture concept where model, context manager, tool use, and execution scaffold interact.

### Disadvantages

The largest weakness is auditability. The imported repository currently gives a strong architecture table but not the full model implementation. Without source-level code, we cannot verify the exact attention cache, routing algorithm, multimodal fusion method, vision-language tokenization, or expert layout.

The described architecture is also extremely demanding operationally. A 1T total-parameter MoE requires expert parallelism, careful router balance, high-bandwidth distributed serving, and robust failure handling. MLA helps with context cost, but does not remove expert-dispatch complexity.

Multimodality increases architectural uncertainty. The source does not yet show whether MoonViT tokens are fused early, late, interleaved, compressed, or routed differently from text tokens. That is a central design question, not a detail.

### Best use case

Kimi-K2.5 is best treated as a high-priority target for further source discovery. Its release-level architecture is important, but not enough for rigorous implementation audit.

### Main risk

The main risk is over-inference from release metadata. It may be frontier architecture, but in the current archive it is not yet source-auditable.

## 5. GLM-4.5 / GLM-4.6 / GLM-4.7

Primary implementation sources:

```text
vendor/inference/sglang/upstream/python/sglang/srt/models/glm4_moe.py
vendor/inference/vllm/upstream/vllm/model_executor/models/glm4_moe_mtp.py
```

Architecture type:

```text
Sparse MoE reasoning/coding family + SGLang runtime implementation + vLLM MTP speculative layer
```

### Advantages

GLM-4.x’s advantage is integration. It is not merely a MoE block; it is a model family designed around coding, reasoning, tool use, long-context agent workflows, and high-performance serving.

The SGLang source shows a production-grade MoE runtime: QKV projection, RoPE, optional QK norm, RadixAttention, sparse MoE block, correction-bias routing, shared experts, top-k routing, expert-parallel dispatch, and all-to-all backends. This suggests the architecture is designed for real distributed inference rather than a clean but naive reference implementation.

The vLLM MTP file adds another important feature: multi-token prediction. The MTP layer normalizes current embeddings and previous hidden states separately, concatenates them, projects them back into hidden space, and passes them through a GLM MoE decoder layer. This is more architecturally integrated than an external speculative decoding heuristic. It means the model family has structural support for predicting future tokens beyond the normal one-token head.

For coding-agent use, this matters: latency and throughput affect whether long tool-interaction loops are economically viable. GLM’s architecture is shaped by this runtime objective.

### Disadvantages

The cost is complexity. GLM-4.x is hard to understand as a single clean mathematical object because its implementation is distributed across release documentation, SGLang serving code, vLLM MTP code, and framework-specific parsers.

MTP introduces additional coupling between the base model, hidden-state flow, shared head, speculative layers, and serving strategy. It can improve throughput, but it also increases correctness and evaluation complexity. Bugs or mismatches in MTP weight loading, step indexing, or hidden-state alignment could degrade generation in ways that are difficult to diagnose.

The MoE runtime is also backend-sensitive. Expert dispatch, shared expert fusion, quantized kernels, and all-to-all communication are not optional details. The architecture’s practical value depends on the serving stack being correct and efficient.

### Best use case

GLM-4.x is best for agentic coding/reasoning workloads where throughput, tool use, long horizon, and speculative acceleration are central.

### Main risk

The main risk is implementation entanglement: architecture, serving, parser behavior, and speculative decoding are tightly coupled.

## 6. Mamba / Mamba2

Primary sources:

```text
vendor/ssm-hybrid/mamba/upstream/mamba_ssm/modules/mamba_simple.py
vendor/ssm-hybrid/mamba/upstream/mamba_ssm/modules/mamba2.py
```

Architecture type:

```text
Selective state-space sequence mixer, non-attention or attention-alternative block
```

### Advantages

Mamba’s advantage is conceptual clarity and asymptotic ambition. It does not merely optimize attention; it replaces the dominant attention sequence mixer with selective state-space updates.

The Mamba v1 source shows a compact structure: input projection, depthwise causal convolution, generation of dt/B/C state parameters, diagonal state transition through `A_log`, learned skip `D`, selective scan, gating, and output projection. During decoding it keeps `conv_state` and `ssm_state`, avoiding full KV cache growth.

Mamba2 improves the architecture by introducing heads, groups, chunked scan, gated RMSNorm, tensor/sequence parallel options, and a more flexible split between SSM dimensions and gated MLP-like dimensions. It is more compatible with large-model engineering than the simpler v1 block.

The main architectural benefit is long-sequence economics. If the task can be represented well by selective state updates, Mamba-style models can avoid the quadratic/persistent-cache burden of attention.

### Disadvantages

The main weakness is expressivity uncertainty relative to attention. Attention gives each token direct content-addressed access to earlier tokens. SSMs compress the past into state. That can be efficient, but compression can lose information. For tasks requiring exact retrieval, multi-hop reference, or arbitrary long-range copying, attention has a natural advantage unless the SSM state is sufficiently expressive and trained appropriately.

Mamba also depends heavily on specialized kernels. The clean mathematical form is not the same as practical performance. Fused scan, causal conv, chunk scan, and state update kernels are central.

Another weakness is ecosystem maturity. Transformer-based models benefit from massive tooling, pretraining recipes, quantization support, serving support, and user experience. Mamba-like models are still less universally supported.

### Best use case

Mamba is best for research into long-sequence alternatives, efficient sequence modeling, and hybrid architectures that reduce attention dependence.

### Main risk

The main risk is that state compression may be too lossy for some LLM behaviors where explicit token-to-token attention remains important.

## 7. vLLM

Primary source:

```text
vendor/inference/vllm/upstream/
```

Architecture type:

```text
Serving/runtime architecture: PagedAttention, model executors, quantization, speculative decoding, parallelism
```

### Advantages

vLLM’s primary advantage is making model architecture economically real. It handles KV memory through PagedAttention, supports continuous batching, chunked prefill, prefix caching, quantization, optimized attention kernels, MoE kernels, tensor/pipeline/expert/context parallelism, and speculative decoding.

From an architecture research perspective, vLLM is valuable because it exposes inference-only model implementations for many model families. Qwen3 and DeepSeek are not just abstractly supported; their model executor files show concrete implementation decisions.

vLLM is also broad. It is useful as a comparative substrate: the same serving framework exposes different model families under similar execution assumptions.

### Disadvantages

vLLM is not a clean model architecture. It is a runtime. Studying it requires separating model semantics from serving optimizations. This can be difficult because modern serving optimizations alter cache layout, quantization behavior, batching strategy, and speculative execution.

The framework is also large and fast-moving. Code paths can be backend-specific, version-specific, and hardware-specific. A model architecture observed in vLLM may be an inference adaptation rather than the canonical training architecture.

### Best use case

vLLM is best for studying how frontier architectures are made deployable at scale.

### Main risk

The main risk is mistaking runtime implementation constraints for the model’s intrinsic architecture.

## 8. SGLang

Primary source:

```text
vendor/inference/sglang/upstream/
```

Architecture type:

```text
Serving/runtime architecture with strong MoE, DeepSeek, GLM, batch-overlap, and tool/reasoning support
```

### Advantages

SGLang is particularly strong for frontier MoE serving. It exposes the operational architecture of DeepSeek/GLM-style models: routing, top-k, hash top-k, DeepEP/all-to-all expert dispatch, shared expert fusion, batch overlap, hardware-specific kernels, parser integration, and speculative execution paths.

Compared with a clean reference implementation, SGLang is closer to how these architectures actually run under production pressure. It shows which parts of the model become systems problems: router GEMM, expert dispatch, KV/cache management, batch overlap, and backend selection.

It is also highly relevant to agentic models because it includes reasoning/tool-call parser infrastructure. For modern agent models, parsing and execution format are not external trivia; they are part of the effective deployed architecture.

### Disadvantages

SGLang’s disadvantage is opacity through engineering density. The more realistic the implementation becomes, the harder it is to isolate the conceptual model. Hardware branches, backend flags, distributed dispatch logic, and batch-overlap state can obscure the block-level architecture.

It is also easy to overfit architectural interpretation to SGLang-specific decisions. Some structures are intrinsic to DeepSeek/GLM; others are serving choices made by SGLang.

### Best use case

SGLang is best for analyzing real MoE deployment architecture and the boundary between model design and runtime systems.

### Main risk

The main risk is conflating the model with its serving policy.

## 9. Cross-architecture comparison

### Highest architectural novelty

```text
1. Mamba / Mamba2: replaces attention with selective SSM sequence mixing.
2. DeepSeek-V3: combines MLA + sparse MoE + shared experts + group routing.
3. GLM-4.x: integrates sparse MoE with MTP and agentic serving paths.
4. Kimi-K2.5: release-described native multimodal 1T MoE with MLA.
5. Qwen3: conservative Transformer refinement with QK norm and long-context support.
```

### Highest practical deployability

```text
1. Qwen3: standard Transformer lineage and broad ecosystem support.
2. vLLM/SGLang-supported DeepSeek/GLM: powerful but more runtime-dependent.
3. Mamba: efficient in principle, but kernel/ecosystem maturity matters.
4. Kimi-K2.5: high potential but source/runtime opacity remains.
```

### Best long-context economics

```text
1. Mamba/Mamba2: avoids standard KV-cache growth through recurrent state.
2. DeepSeek-V3 / Kimi-K2.5: MLA compresses KV pressure.
3. Qwen3: relies more on GQA, RoPE scaling, and serving cache systems.
4. GLM-4.x: long context depends on MoE runtime plus framework support.
```

### Best source auditability

```text
1. DeepSeek-V3 official inference/model.py
2. Mamba official source
3. vLLM Qwen3 / DeepSeek / GLM MTP implementations
4. SGLang DeepSeek / GLM implementations
5. Qwen3 official repo as release metadata
6. Kimi-K2.5 release table only
```

### Most fragile components

```text
DeepSeek-V3: router dynamics, MLA cache correctness, FP8/dispatch assumptions
DeepSeek-R1: post-training behavior attribution, prompt/decoding dependence
Qwen3: long-context scaling and QK norm interactions, release/source split
Kimi-K2.5: unknown implementation details, multimodal fusion opacity
GLM-4.x: MTP alignment, serving-stack coupling, expert dispatch complexity
Mamba: state compression limits, fused scan/kernel dependence
vLLM: model semantics vs runtime adaptation ambiguity
SGLang: engineering-density opacity, backend-specific behavior
```

## 10. Strategic conclusion

The imported sources show three competing architectural strategies:

### Strategy A: Scale capacity sparsely

Represented by DeepSeek-V3, GLM-4.x, and Kimi-K2.5.

This strategy increases total parameter capacity while controlling activated compute. It is currently the dominant frontier path. Its weakness is router/expert/runtime complexity.

### Strategy B: Reduce attention memory pressure

Represented by DeepSeek-V3/Kimi MLA and Mamba’s SSM design.

MLA preserves attention-like behavior while compressing KV. Mamba goes further by replacing attention sequence mixing with recurrent state. MLA is more compatible with existing Transformer behavior; Mamba is more radical but riskier.

### Strategy C: Move architecture into runtime

Represented by vLLM and SGLang.

Modern LLM architecture is no longer only the neural block. Serving architecture is now part of the model’s effective form: KV layout, batching, MoE dispatch, speculative decoding, parsers, and quantization determine what architectures are feasible.

The strongest near-term research path is therefore not to pick one model as “best,” but to study the interfaces between these strategies:

```text
MoE routing + MLA cache + serving runtime
MTP/speculative layers + MoE decoder blocks
SSM sequence mixing + Transformer/MoE hybrids
QK norm / RoPE scaling + long-context cache design
```

That interface layer is where current frontier architecture is actually changing.
