#!/usr/bin/env bash
#
# Extract SigLIP text weights (token + position embeddings) and download
# the SentencePiece tokenizer model for the Hailo SigLIP B/16 text encoder.
#
# Source: google/siglip-base-patch16-224 on HuggingFace
#
# Uses the hailo-base Docker image. Installs safetensors to extract weights
# from the model without downloading PyTorch.
#
# Usage:
#   HAILORT_VERSION=<ver> ./scripts/extract_siglip_weights.sh
#
# HAILORT_VERSION is required — it selects the hailo-base image to run in.
# setup.sh exports it; a standalone run must pass it.
#
# Output:
#   models/siglip_text_weights.npz
#   models/spiece.model
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_NPZ="$PROJECT_DIR/models/siglip_text_weights.npz"
OUTPUT_SPIECE="$PROJECT_DIR/models/spiece.model"
CONTAINER_NAME="siglip-extract-$$"

# Required, no default. This selects which hailo-base image the extraction runs
# in; defaulting would silently reach for a stale base image when this script is
# run on its own (setup.sh exports the variable, but a fresh shell does not).
if [[ -z "${HAILORT_VERSION:-}" ]]; then
    echo "ERROR: HAILORT_VERSION is required — there is no default."
    echo ""
    echo "  It selects the hailo-base image this extraction runs in, and must"
    echo "  match the hailo_pci kernel module on this host:"
    echo ""
    echo "    modinfo hailo_pci | grep '^version:'"
    echo ""
    echo "  Then re-run:  HAILORT_VERSION=4.24.0 $0"
    exit 1
fi
IMAGE="hailo-base:v${HAILORT_VERSION}"

HF_BASE="https://huggingface.co/google/siglip-base-patch16-224/resolve/main"
SAFETENSORS_URL="$HF_BASE/model.safetensors"
SPIECE_URL="$HF_BASE/spiece.model"

echo "=== Extract SigLIP text weights ==="
echo "Using image: $IMAGE"
echo "Output: $OUTPUT_NPZ, $OUTPUT_SPIECE"
echo ""

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: Docker image '$IMAGE' not found."
    echo "Build it first: docker build --build-arg HAILORT_VERSION=$HAILORT_VERSION -t $IMAGE -f Dockerfile.hailo-base ."
    exit 1
fi

# Download spiece.model directly (small file, no Docker needed)
if [[ ! -f "$OUTPUT_SPIECE" ]]; then
    echo "Downloading SentencePiece model (798 KB)..."
    curl -Lo "$OUTPUT_SPIECE" "$SPIECE_URL"
else
    echo "spiece.model already exists, skipping."
fi

# Extract weights via Docker
EXTRACT_PY=$(mktemp /tmp/extract_siglip_XXXXXX.py)
cat > "$EXTRACT_PY" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""Extract SigLIP token + position embeddings from safetensors."""

import numpy as np
from safetensors.numpy import load_file

SAFETENSORS_PATH = "/tmp/model.safetensors"
OUTPUT_PATH = "/output/siglip_text_weights.npz"

print(f"Loading safetensors: {SAFETENSORS_PATH}")
tensors = load_file(SAFETENSORS_PATH)

# Print text-related keys
text_keys = sorted(k for k in tensors if "text_model.embeddings" in k)
print(f"Embedding keys: {text_keys}")

token_emb = tensors["text_model.embeddings.token_embedding.weight"].astype(np.float32)
pos_emb = tensors["text_model.embeddings.position_embedding.weight"].astype(np.float32)

print(f"  token_embedding: {token_emb.shape} {token_emb.dtype}")
print(f"  position_embedding: {pos_emb.shape} {pos_emb.dtype}")

assert token_emb.shape == (32000, 768), f"Unexpected shape: {token_emb.shape}"
assert pos_emb.shape == (64, 768), f"Unexpected shape: {pos_emb.shape}"

np.savez(OUTPUT_PATH, token_embedding=token_emb, position_embedding=pos_emb)
print(f"\nSaved to {OUTPUT_PATH}")

# Verify
w = np.load(OUTPUT_PATH)
for k in w.files:
    print(f"  {k}: {w[k].shape} {w[k].dtype}")
print("Done.")
PYTHON_SCRIPT

echo "Starting extraction container..."
docker run --rm -d \
    --name "$CONTAINER_NAME" \
    "$IMAGE" \
    sleep 600

cleanup() {
    echo "Cleaning up container..."
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    rm -f "$EXTRACT_PY"
}
trap cleanup EXIT

docker cp "$EXTRACT_PY" "$CONTAINER_NAME:/tmp/extract.py"

echo "Installing safetensors..."
docker exec "$CONTAINER_NAME" \
    pip3 install --break-system-packages -q safetensors 2>&1 | tail -3

echo ""
echo "Downloading SigLIP model.safetensors (~813MB)..."
docker exec "$CONTAINER_NAME" \
    python3 -c "
import urllib.request, sys
print('Downloading...', flush=True)
def progress(count, block, total):
    mb = count * block / 1e6
    if total > 0:
        pct = count * block * 100 / total
        print(f'\r  {mb:.0f}/{total/1e6:.0f} MB ({pct:.0f}%)', end='', flush=True)
    else:
        print(f'\r  {mb:.0f} MB', end='', flush=True)
urllib.request.urlretrieve('$SAFETENSORS_URL', '/tmp/model.safetensors', reporthook=progress)
print('\n  Done.')
"

echo ""
echo "Extracting weights..."
docker exec "$CONTAINER_NAME" mkdir -p /output
docker exec "$CONTAINER_NAME" python3 /tmp/extract.py

echo ""
echo "Copying result to $OUTPUT_NPZ..."
docker cp "$CONTAINER_NAME:/output/siglip_text_weights.npz" "$OUTPUT_NPZ"

echo ""
echo "Done. Files saved:"
ls -lh "$OUTPUT_NPZ" "$OUTPUT_SPIECE"
