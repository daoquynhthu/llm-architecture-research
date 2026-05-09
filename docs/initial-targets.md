# Initial Import Targets

This document defines the first practical import targets for the repository.

## Priority A: safe architecture references and clean-room reimplementations

### Mamba family

- Family: state-space / selective SSM
- Upstream: https://github.com/state-spaces/mamba
- Suggested import mode: submodule or clean-room minimal implementation
- Rationale: Apache-2.0 source license and clear architecture focus.
- Core concepts: selective scan, hardware-aware SSM block, Mamba-2 SSD duality, Mamba-3 lineage.

### Dense Transformer baseline

- Family: decoder-only Transformer
- Suggested import mode: clean-room minimal implementation
- Rationale: needed as a local comparison baseline for attention, MoE, SSM, and hybrid architectures.
- Core concepts: RMSNorm, RoPE, causal attention, GQA/MQA switch, SwiGLU.

## Priority B: architecture archive with license caution

### DeepSeek-V3 family

- Family: MoE Transformer
- Upstream: https://github.com/deepseek-ai/DeepSeek-V3
- Suggested import mode: reference-only first; vendored source only after preserving code/model license boundaries.
- Core concepts: MLA, DeepSeekMoE, auxiliary-loss-free load balancing, MTP, FP8 inference kernels.

### Qwen family

- Family: dense Transformer and MoE Transformer
- Suggested import mode: reference-only first.
- Core concepts: modern decoder-only Transformer design, long-context variants, MoE variants depending on release.

### Mistral/Mixtral family

- Family: dense Transformer and sparse MoE
- Suggested import mode: reference-only first.
- Core concepts: sliding-window attention, GQA, sparse MoE routing.

## Priority C: systems architecture

### vLLM

- Family: inference runtime
- Suggested import mode: reference-only or submodule.
- Core concepts: PagedAttention, continuous batching, KV-cache management.

### FlashAttention

- Family: attention kernel / systems component
- Suggested import mode: reference-only or submodule.
- Core concepts: IO-aware exact attention, long-sequence efficient attention kernels.

## Rule for actual source import

The repository should not pretend that reference-only targets are already mirrored. Each imported target must move through this sequence:

1. record upstream URL and architecture relevance;
2. check license and model-weight terms;
3. create vendor model card;
4. choose import mode;
5. add submodule or vendored source;
6. preserve license and commit SHA;
7. add local architecture notes.
