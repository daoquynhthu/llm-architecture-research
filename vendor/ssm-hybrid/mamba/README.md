# Mamba Architecture Archive

- Upstream: https://github.com/state-spaces/mamba
- Family: selective state-space model / SSM hybrid
- Import mode: vendored architecture notes + clean-room local implementation target
- Upstream license: Apache-2.0
- Weights: not mirrored
- Tokenizers/data: not mirrored

## Architecture elements

Mamba replaces attention-dominant sequence mixing with a selective state-space block. The core architectural pattern is:

1. input projection splits the hidden state into an SSM stream and a gating stream;
2. depthwise causal convolution provides local mixing;
3. input-dependent projections generate selective SSM parameters;
4. selective scan performs sequence mixing with data-dependent state updates;
5. the gated result is projected back to model dimension.

## Important upstream files to track

```text
mamba_ssm/modules/mamba_simple.py
mamba_ssm/modules/mamba2.py
mamba_ssm/ops/selective_scan_interface.py
csrc/selective_scan/
```

## Local status

A clean-room minimal implementation is stored under:

```text
models/minimal-mamba/mamba_block.py
```

This local implementation is intended for architecture reading and small experiments. It is not a drop-in replacement for the optimized upstream CUDA/Triton kernels.
