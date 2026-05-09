# Import Record: vllm

- Upstream: https://github.com/vllm-project/vllm.git
- Commit: d6563d693c817ec762ccd0bb6c44c56b4f3dc4a3
- Imported at: 2026-05-09T07:28:55Z
- Import mode: vendored source snapshot
- Weights: excluded
- Tokenizers/data: excluded by default when matched by script filters
- Destination: vendor/inference/vllm/upstream

## Exclusion policy

The import script excludes common checkpoint, tensor, dataset, output, and cache artifacts. This repository is intended to archive source code and architecture files, not model weights or datasets.

## Refresh command

```bash
scripts/import_upstream.sh vllm
```
