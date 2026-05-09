# DeepSeek-V3 Architecture Archive

- Upstream: https://github.com/deepseek-ai/DeepSeek-V3
- Family: sparse Mixture-of-Experts Transformer
- Import mode: architecture card + selected source import when complete files can be fetched
- Code license: MIT, recorded in `LICENSE-CODE`
- Model license / weights: not mirrored
- Tokenizer/data: not mirrored

## Architecture elements

The public inference code exposes several architecture-relevant components:

1. `ModelArgs` records the structural hyperparameters: dense dimension, MoE dimension, number of layers, number of dense layers, attention heads, routed experts, shared experts, activated experts, MLA low-rank parameters, RoPE/YaRN scaling parameters.
2. `MLA` implements Multi-Head Latent Attention with compressed KV cache modes.
3. `Gate`, `Expert`, and `MoE` implement sparse expert routing with top-k activation and optional group-limited routing.
4. The block structure switches from dense MLP layers to MoE layers after `n_dense_layers`.
5. The inference path includes BF16/FP8-aware linear execution hooks.

## Important upstream files to track

```text
inference/model.py
inference/kernel.py
inference/generate.py
```

## Local status

`inference/model.py` was inspected, but the current connector returned a truncated payload for the full source file. To avoid storing an incomplete source as if it were complete, this directory currently records license and architecture metadata only.

A clean-room minimal DeepSeek-style MoE/MLA implementation should be placed under:

```text
models/minimal-deepseek-moe/
```
