# LLM Architecture Research

A public research repository for collecting, comparing, and re-implementing frontier large-model architectures.

This repository is intended to function as an architecture archive rather than a model-weight mirror. It stores:

- architecture notes and comparative analysis;
- minimal educational implementations;
- references to upstream repositories and papers;
- vendor manifests for third-party code imports;
- compatibility notes about licenses, checkpoints, tokenizer assets, and training recipes.

## Repository stance

The main repository is licensed under Apache-2.0. Third-party code, model definitions, configs, tokenizers, and checkpoints remain under their original upstream licenses. Imported code must preserve attribution and license notices.

This distinction is important because many modern model repositories combine permissive source code with separately governed model weights, data terms, or acceptable-use policies.

## Structure

```text
docs/                         Architecture notes and research memos
manifests/                    Machine-readable upstream inventory
vendor/                       Imported or mirrored architecture code, grouped by upstream
models/                       Minimal local implementations and experiments
templates/                    Review and import templates
scripts/                      Sync/check utilities
```

## Initial architecture targets

The first archive targets are grouped by architectural family:

1. Dense decoder-only Transformers: GPT-style, LLaMA-style, Qwen-style, Mistral-style.
2. Mixture-of-Experts Transformers: Mixtral-style, DeepSeek-style, Switch/GShard lineage.
3. Attention variants: MQA, GQA, sliding-window attention, latent attention, MLA-like decompositions.
4. State-space and hybrid models: Mamba-like selective SSM, RWKV-like recurrent hybrids, Jamba-like hybrids.
5. Long-context systems: RoPE scaling, ALiBi-like schemes, YaRN/NTK-aware scaling, recurrent memory, retrieval-augmented context.
6. Inference-oriented architecture: KV-cache design, paged attention, speculative decoding, quantization-aware blocks.

## Import rule

Do not paste third-party source code into this repository unless one of the following is true:

- the upstream license permits redistribution;
- the original license file is preserved under the imported subtree;
- the import is a clean-room minimal reimplementation;
- the import is only a submodule/link/reference rather than copied source.

For each architecture import, add one entry to `manifests/frontier-architectures.yaml` and one model card under `vendor/<family>/<project>/README.md`.
