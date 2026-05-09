# LLM Architecture Research

A public archive for importing and storing open-source large-model architecture code.

This repository is for source code and architecture files only. It is not intended to mirror model weights, datasets, tokenizer artifacts, generated outputs, or cache files.

## Import workflow

Use the manual GitHub Action:

```text
Actions -> import-upstream -> Run workflow -> choose project
```

Supported import targets:

```text
mamba
deepseek-v3
flash-attention
vllm
```

The workflow runs `scripts/import_upstream.sh`, which clones the selected upstream repository, copies its source tree into `vendor/<family>/<project>/upstream`, excludes common weight/data artifacts, writes `IMPORT.md`, and commits the result back to this repository.

## License boundary

The repository-level license is Apache-2.0. Third-party code imported under `vendor/` remains governed by its upstream license. Each imported subtree must retain its original license files and import record.
