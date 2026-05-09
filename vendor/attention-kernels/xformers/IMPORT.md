# Import Record: xformers

- Upstream: https://github.com/facebookresearch/xformers.git
- Commit: ca6d2aa0d43241fc8a8dcd872debc2406160160d
- Imported at: 2026-05-09T07:38:13Z
- Import mode: vendored source snapshot
- Weights: excluded
- Tokenizers/data: excluded by default when matched by script filters
- Destination: vendor/attention-kernels/xformers/upstream

## Exclusion policy

The import script excludes common checkpoint, tensor, dataset, output, and cache artifacts. This repository is intended to archive source code and architecture files, not model weights or datasets.

## Refresh command

```bash
scripts/import_upstream.sh xformers
```
