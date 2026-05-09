#!/usr/bin/env bash
set -euo pipefail

# Import upstream architecture/source code into vendor/<family>/<name>/upstream.
# This script intentionally excludes common model-weight/checkpoint/data artifacts.
# Usage:
#   scripts/import_upstream.sh <preset>
#   scripts/import_upstream.sh custom <git-url> <family> <name>

PROJECT="${1:-}"
if [[ -z "$PROJECT" ]]; then
  echo "usage: scripts/import_upstream.sh <mamba|deepseek-v3|deepseek-r1|flash-attention|vllm|custom> [url family name]" >&2
  exit 2
fi

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

EXCLUDES=(
  "--exclude=.git"
  "--exclude=.gitmodules"
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
  "--exclude=checkpoint*/"
  "--exclude=weights/"
  "--exclude=model_weights/"
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
  deepseek-r1)
    URL="https://github.com/deepseek-ai/DeepSeek-R1.git"
    FAMILY="reasoning"
    NAME="deepseek-r1"
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
  custom)
    URL="${2:-}"
    FAMILY="${3:-}"
    NAME="${4:-}"
    if [[ -z "$URL" || -z "$FAMILY" || -z "$NAME" ]]; then
      echo "usage: scripts/import_upstream.sh custom <git-url> <family> <name>" >&2
      exit 2
    fi
    ;;
  *)
    echo "unknown project: $PROJECT" >&2
    exit 2
    ;;
esac

if [[ ! "$URL" =~ ^https://github.com/[^/]+/[^/]+(\.git)?$ ]]; then
  echo "only public GitHub HTTPS repository URLs are accepted: $URL" >&2
  exit 2
fi

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

The import script excludes common checkpoint, tensor, dataset, output, and cache artifacts. This repository is intended to archive source code and architecture files, not model weights or datasets.

## Refresh command

\`\`\`bash
scripts/import_upstream.sh $PROJECT${PROJECT:+}
\`\`\`
EOF

echo "imported $PROJECT at $COMMIT into $DEST"
