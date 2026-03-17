#!/usr/bin/env bash
#
# Extract TinyCLIP text weights (token/positional embeddings + text projection)
# from the official checkpoint and save as tinyclip_text_weights.npz.
#
# Downloads the TinyCLIP-ViT-39M-16-Text-19M-YFCC15M checkpoint (.pt) from
# GitHub, extracts the three CPU-side weight tensors needed for the Hailo
# text encoder pipeline, and saves them as a lightweight .npz file.
#
# Uses the hailo-base Docker image (has Python + numpy). Only needs torch
# for loading the checkpoint — no GPU required.
#
# Usage:
#   ./scripts/extract_tinyclip_weights.sh
#
# Output:
#   models/tinyclip_text_weights.npz
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT="$PROJECT_DIR/models/tinyclip_text_weights.npz"
CONTAINER_NAME="tinyclip-extract-$$"
IMAGE="hailo-base:v4.23.0"
CHECKPOINT_URL="https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M.pt"

echo "=== Extract TinyCLIP text weights ==="
echo "Using image: $IMAGE"
echo "Checkpoint: $CHECKPOINT_URL"
echo "Output: $OUTPUT"
echo ""

# Check that base image exists
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: Docker image '$IMAGE' not found."
    echo "Build it first: docker build -t hailo-base:v4.23.0 -f Dockerfile.hailo-base ."
    exit 1
fi

# Write the Python extraction script to a temp file
EXTRACT_PY=$(mktemp /tmp/extract_tinyclip_XXXXXX.py)
cat > "$EXTRACT_PY" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""Extract TinyCLIP text weights from checkpoint for Hailo inference.

Loads the .pt checkpoint (TinyCLIP/open_clip format) and extracts:
  - token_embedding:      (49408, 512) float32
  - positional_embedding: (77, 512) float32
  - text_projection:      (512, 512) float32
  - eot_token_id:         scalar int (49407)
"""

import sys
import numpy as np
import torch

CHECKPOINT_PATH = "/tmp/tinyclip.pt"
OUTPUT_PATH = "/output/tinyclip_text_weights.npz"

# TinyCLIP .pt checkpoint structure:
#   checkpoint['state_dict'] contains keys prefixed with '_text_encoder.' / '_image_encoder.'
#   Some checkpoints use 'module.' prefix from DDP training.

print(f"Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

# Extract state dict
if "state_dict" in checkpoint:
    state = checkpoint["state_dict"]
elif "model" in checkpoint:
    state = checkpoint["model"]
else:
    state = checkpoint

# Strip 'module.' prefix if present (from DDP)
state = {k.replace("module.", ""): v for k, v in state.items()}

# Print all text-related keys for debugging
text_keys = sorted(k for k in state if "_text_encoder" in k or "text" in k.lower())
print(f"Text-related keys ({len(text_keys)}):")
for k in text_keys:
    print(f"  {k}: {state[k].shape}")

# Extract weights — try TinyCLIP format first, then plain open_clip format
KEY_MAP = {
    "token_embedding": [
        "_text_encoder.token_embedding.weight",
        "token_embedding.weight",
        "text.token_embedding.weight",
    ],
    "positional_embedding": [
        "_text_encoder.positional_embedding",
        "positional_embedding",
        "text.positional_embedding",
    ],
    "text_projection": [
        "_text_encoder.text_projection",
        "text_projection",
        "text.text_projection",
    ],
}

results = {}
for name, candidates in KEY_MAP.items():
    found = False
    for key in candidates:
        if key in state:
            arr = state[key].cpu().numpy().astype(np.float32)
            results[name] = arr
            print(f"\n  {name} <- '{key}': shape={arr.shape}")
            found = True
            break
    if not found:
        print(f"\nERROR: Could not find {name}")
        print(f"  Tried: {candidates}")
        print(f"  Available keys: {list(state.keys())}")
        sys.exit(1)

# EOT token ID — standard CLIP BPE tokenizer: <|endoftext|> = 49407
eot_id = 49407
results["eot_token_id"] = np.array(eot_id, dtype=np.int64)

# Validate expected shapes
assert results["token_embedding"].shape == (49408, 512), \
    f"Unexpected token_embedding shape: {results['token_embedding'].shape}"
assert results["positional_embedding"].shape == (77, 512), \
    f"Unexpected positional_embedding shape: {results['positional_embedding'].shape}"
assert results["text_projection"].shape == (512, 512), \
    f"Unexpected text_projection shape: {results['text_projection'].shape}"

print(f"\nFinal shapes:")
for k, v in results.items():
    print(f"  {k}: shape={v.shape} dtype={v.dtype}")

np.savez(OUTPUT_PATH, **results)
print(f"\nSaved to {OUTPUT_PATH}")

# Verify
w = np.load(OUTPUT_PATH)
print(f"Verification — keys: {w.files}")
for k in w.files:
    print(f"  {k}: shape={w[k].shape} dtype={w[k].dtype}")

print("\nDone.")
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

# Copy extraction script into container
docker cp "$EXTRACT_PY" "$CONTAINER_NAME:/tmp/extract.py"

# Install torch (CPU-only, ~300MB) — we only need it to load the .pt file
echo "Installing PyTorch (CPU-only)..."
docker exec "$CONTAINER_NAME" \
    pip3 install --break-system-packages -q \
    torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3

# Download the checkpoint
echo ""
echo "Downloading TinyCLIP checkpoint (~330MB)..."
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
urllib.request.urlretrieve('$CHECKPOINT_URL', '/tmp/tinyclip.pt', reporthook=progress)
print('\n  Done.')
"

# Run extraction
echo ""
echo "Extracting weights..."
docker exec "$CONTAINER_NAME" mkdir -p /output
docker exec "$CONTAINER_NAME" python3 /tmp/extract.py

# Copy result out
echo ""
echo "Copying result to $OUTPUT..."
docker cp "$CONTAINER_NAME:/output/tinyclip_text_weights.npz" "$OUTPUT"

echo ""
echo "Done. File saved to:"
ls -lh "$OUTPUT"
