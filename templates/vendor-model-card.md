# Vendor Model Card Template

## Identity

- Project id:
- Family:
- Upstream repository:
- Upstream commit:
- Paper:
- Import mode: reference-only | submodule | vendored-source | clean-room
- Import date:
- Local path:

## License boundary

- Upstream code license:
- Model-weight license:
- Tokenizer/data license:
- Local modifications license:

## Architecture summary

Describe only the architecture-relevant facts:

- block type;
- attention type;
- positional encoding;
- normalization;
- FFN/MLP design;
- MoE routing, if any;
- recurrent/state-space component, if any;
- long-context mechanism;
- inference-specific mechanism.

## Imported files

List copied or vendored files and explain why each is needed.

```text
path | upstream path | reason | modified?
```

## Local modifications

Record all modifications relative to upstream.

## Research notes

Explain why this architecture is important to the archive.

## Verification checklist

- [ ] Upstream license checked.
- [ ] License file preserved if source code is copied.
- [ ] Commit SHA recorded.
- [ ] Model weights not mirrored unless explicitly permitted.
- [ ] Tokenizer/data assets not mirrored unless explicitly permitted.
- [ ] Local modifications documented.
