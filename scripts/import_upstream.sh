#!/usr/bin/env bash
set -euo pipefail

# Import upstream architecture/source code into vendor/<family>/<name>/upstream.
# This script intentionally excludes common model-weight/checkpoint/data artifacts.
# Usage:
#   scripts/import_upstream.sh <preset>
#   scripts/import_upstream.sh custom <git-url> <family> <name>

PROJECT="${1:-}"
if [[ -z "$PROJECT" ]]; then
  echo "usage: scripts/import_upstream.sh <preset|custom> [url family name]" >&2
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
  # Dense / MoE LLM families
  qwen3)
    URL="https://github.com/QwenLM/Qwen3.git"; FAMILY="dense-transformer"; NAME="qwen3" ;;
  qwen3-coder)
    URL="https://github.com/QwenLM/Qwen3-Coder.git"; FAMILY="code-model"; NAME="qwen3-coder" ;;
  qwen3-vl)
    URL="https://github.com/QwenLM/Qwen3-VL.git"; FAMILY="multimodal"; NAME="qwen3-vl" ;;
  qwen3-omni)
    URL="https://github.com/QwenLM/Qwen3-Omni.git"; FAMILY="multimodal"; NAME="qwen3-omni" ;;
  qwen3-tts)
    URL="https://github.com/QwenLM/Qwen3-TTS.git"; FAMILY="speech"; NAME="qwen3-tts" ;;

  deepseek-v3)
    URL="https://github.com/deepseek-ai/DeepSeek-V3.git"; FAMILY="moe"; NAME="deepseek-v3" ;;
  deepseek-r1)
    URL="https://github.com/deepseek-ai/DeepSeek-R1.git"; FAMILY="reasoning"; NAME="deepseek-r1" ;;
  deepseek-ocr)
    URL="https://github.com/deepseek-ai/DeepSeek-OCR.git"; FAMILY="multimodal"; NAME="deepseek-ocr" ;;
  deepseek-vl2)
    URL="https://github.com/deepseek-ai/DeepSeek-VL2.git"; FAMILY="multimodal"; NAME="deepseek-vl2" ;;
  deepseek-coder-v2)
    URL="https://github.com/deepseek-ai/DeepSeek-Coder-V2.git"; FAMILY="code-model"; NAME="deepseek-coder-v2" ;;

  kimi-k2)
    URL="https://github.com/MoonshotAI/Kimi-K2.git"; FAMILY="moe"; NAME="kimi-k2" ;;
  kimi-k2-5)
    URL="https://github.com/MoonshotAI/Kimi-K2.5.git"; FAMILY="moe"; NAME="kimi-k2-5" ;;

  glm-4-5)
    URL="https://github.com/zai-org/GLM-4.5.git"; FAMILY="dense-transformer"; NAME="glm-4-5" ;;
  glm-v)
    URL="https://github.com/zai-org/GLM-V.git"; FAMILY="multimodal"; NAME="glm-v" ;;

  minicpm)
    URL="https://github.com/OpenBMB/MiniCPM.git"; FAMILY="small-llm"; NAME="minicpm" ;;
  minicpm-o)
    URL="https://github.com/OpenBMB/MiniCPM-o.git"; FAMILY="multimodal"; NAME="minicpm-o" ;;
  intern-s1)
    URL="https://github.com/InternLM/Intern-S1.git"; FAMILY="reasoning"; NAME="intern-s1" ;;
  internlm)
    URL="https://github.com/InternLM/InternLM.git"; FAMILY="dense-transformer"; NAME="internlm" ;;
  hunyuan-turbos)
    URL="https://github.com/Tencent/Hunyuan-TurboS.git"; FAMILY="dense-transformer"; NAME="hunyuan-turbos" ;;
  hunyuan-t1)
    URL="https://github.com/Tencent/llm.hunyuan.T1.git"; FAMILY="reasoning"; NAME="hunyuan-t1" ;;
  mistral-inference)
    URL="https://github.com/mistralai/mistral-inference.git"; FAMILY="dense-transformer"; NAME="mistral-inference" ;;

  # State-space / recurrent / hybrid architecture
  mamba)
    URL="https://github.com/state-spaces/mamba.git"; FAMILY="ssm-hybrid"; NAME="mamba" ;;
  rwkv-lm)
    URL="https://github.com/BlinkDL/RWKV-LM.git"; FAMILY="recurrent"; NAME="rwkv-lm" ;;

  # Inference / kernels / serving architecture
  flash-attention)
    URL="https://github.com/Dao-AILab/flash-attention.git"; FAMILY="attention-kernels"; NAME="flash-attention" ;;
  vllm)
    URL="https://github.com/vllm-project/vllm.git"; FAMILY="inference"; NAME="vllm" ;;
  sglang)
    URL="https://github.com/sgl-project/sglang.git"; FAMILY="inference"; NAME="sglang" ;;
  xformers)
    URL="https://github.com/facebookresearch/xformers.git"; FAMILY="attention-kernels"; NAME="xformers" ;;
  megatron-lm)
    URL="https://github.com/NVIDIA/Megatron-LM.git"; FAMILY="training-architecture"; NAME="megatron-lm" ;;

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
scripts/import_upstream.sh $PROJECT
\`\`\`
EOF

echo "imported $PROJECT at $COMMIT into $DEST"
