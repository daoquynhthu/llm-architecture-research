# Vendor Archive

This directory stores third-party architecture imports or references.

Default status: reference-only.

Recommended layout:

```text
vendor/
  dense-transformer/
    qwen/
      README.md
    llama/
      README.md
  moe/
    deepseek-v3/
      README.md
    mixtral/
      README.md
  ssm-hybrid/
    mamba/
      README.md
    rwkv/
      README.md
  inference/
    vllm/
      README.md
```

Use `templates/vendor-model-card.md` for each imported architecture.

Do not copy source code here until the upstream license has been checked and recorded.
