# Minimal DeepSeek-style MoE / MLA

This directory is for a clean-room, educational implementation of the architectural ideas exposed by DeepSeek-V3-style inference code.

Planned modules:

- `model_args.py`: small dataclass for structural parameters;
- `mla.py`: compressed latent KV attention skeleton;
- `moe.py`: top-k routed experts plus shared expert path;
- `block.py`: dense-to-MoE block transition;
- `test_shapes.py`: shape-level tests.

This directory must not contain model weights or tokenizer assets.
