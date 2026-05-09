#!/usr/bin/env bash
set -euo pipefail

# Import selected upstream architecture code into vendor/<family>/<project>/upstream.
# This script intentionally excludes common weight/checkpoint/data artifacts.
# It is designed to run inside GitHub Actions or a local clone of this repository.

PROJECT="${1:-}"
if [[ -z "$PROJECT" ]]; then
  echo "usage: scripts/import_upstream.sh <mamba|deepseek-v3|flash-attention|vllm>" >&2
  exit 2
fi

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

EXCLUDES=(
  "--exclude=.git"
  "--exclude=*.safetensors"
  "--exclude=*.bin"
  "--exclude=*.pt"
  "--exclude=*.pth"
  "--exclude=*.ckpt"
  "--exclude=*.gguf"
  "--exclude=*.onnx"
  "--exclude=*.tflite"
  "--exclude=*.npz"
  "--exclude=*.npy"
  "--exclude=checkpoints/"
  "--exclude=weights/"
  "--exclude=models--*/"
  "--exclude=.cache/"
  "--exclude=wandb/"
  "--exclude=outputs/"
  "--exclude=data/"
  "--exclude=datasets/"
)

case "$PROJECT" in
  mamba)
    URL="https://github.com/state-spaces/mamba.git"
    FAMILY="ssm-hybrid"
    NAME="mamba"
    ;;
  deepseek-v3)
    URL="https://github.com/deepseek-ai/DeepSeek-V3.git"
    FAMILY="moe"
    NAME="deepseek-v3"
    ;;
  flash-attention)
    URL="https://github.com/Dao-AILab/flash-attention.git"
    FAMILY="attention-kernels"
    NAME="flash-attention"
    ;;
  vllm)
    URL="https://github.com/vllm-project/vllm.git"
    FAMILY="inference"
    NAME="vllm"
    ;;
  *)
    echo "unknown project: $PROJECT" >&2
    exit 2
    ;;
esac

SRC="$TMP_ROOT/$NAME"
DEST="vendor/$FAMILY/$NAME/upstream"
IMPORT_FILE="vendor/$FAMILY/$NAME/IMPORT.md"

echo "cloning $URL"
git clone --depth=1 "$URL" "$SRC"
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
DATE_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
mkdir -p "$DEST"

rsync -a "${EXCLUDES[@]}" "$SRC/" "$DEST/"

cat > "$IMPORT_FILE" <<EOF
# Import Record: $NAME

- Upstream: $URL
- Commit: $COMMIT
- Imported at: $DATE_UTC
- Import mode: vendored source snapshot
- Weights: excluded
- Tokenizers/data: excluded by default when matched by script filters
- Destination: $DEST

## Exclusion policy

The import script excludes common checkpoint, tensor, dataset, output, and cache artifacts. This repository is intended to archive code and architecture, not model weights or datasets.

## Refresh command

\`\`\`bash
scripts/import_upstream.sh $PROJECT
\`\`\`
EOF

echo "imported $PROJECT at $COMMIT into $DEST"
