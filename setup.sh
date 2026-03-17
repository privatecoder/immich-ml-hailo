#!/usr/bin/env bash
#
# Full setup: check prerequisites, build images, extract weights, run tests.
#
# Usage:
#   ./setup.sh                    # uses tests/test.jpg for testing
#   ./setup.sh /path/to/img.jpg   # uses a custom test image
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HAILORT_VERSION="4.23.0"
IMAGE_BASE="hailo-base:v${HAILORT_VERSION}"
IMAGE_APP="immich-ml-hailo:v${HAILORT_VERSION}"
TEST_IMAGE="${1:-$SCRIPT_DIR/tests/test.jpg}"

red()   { printf "\033[31m%s\033[0m" "$*"; }
green() { printf "\033[32m%s\033[0m" "$*"; }
bold()  { printf "\033[1m%s\033[0m" "$*"; }

step() { echo ""; bold "[$1/$TOTAL] $2"; echo ""; }
ok()   { echo "  $(green OK): $1"; }
fail() { echo "  $(red FAIL): $1"; exit 1; }

TOTAL=6

# ── Step 1: Detect platform and check HailoRT files ──────────────────

step 1 "Checking HailoRT packages in hailo-rt-4/"

ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  DEB_ARCH="amd64";  WHL_ARCH="x86_64"  ;;
    aarch64) DEB_ARCH="arm64";  WHL_ARCH="aarch64"  ;;
    arm64)   DEB_ARCH="arm64";  WHL_ARCH="aarch64"  ;;
    *)       fail "Unsupported architecture: $ARCH" ;;
esac

echo "  Platform: $ARCH -> deb=$DEB_ARCH whl=$WHL_ARCH"

DEB_FILE="hailo-rt-4/hailort_${HAILORT_VERSION}_${DEB_ARCH}.deb"
WHL_FILE="hailo-rt-4/hailort-${HAILORT_VERSION}-cp312-cp312-linux_${WHL_ARCH}.whl"

MISSING=()
for f in "$DEB_FILE" "$WHL_FILE"; do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
        ok "$f"
    else
        MISSING+=("$f")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo ""
    echo "  $(red 'Missing HailoRT packages:')"
    for f in "${MISSING[@]}"; do
        echo "    - $f"
    done
    echo ""
    echo "  Download from https://hailo.ai/developer-zone (requires account):"
    echo "    - HailoRT Python package (whl) for Python 3.12, $ARCH"
    echo "    - HailoRT Ubuntu package (deb) for $DEB_ARCH"
    exit 1
fi

# ── Step 2: Check model files ────────────────────────────────────────

step 2 "Checking model files in models/"

MODEL_FILES=(
    "scrfd_2.5g.hef"
    "arcface_r50.hef"
    "tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder.hef"
    "tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder.hef"
    "paddle_ocr_v5_mobile_detection.hef"
    "paddle_ocr_v5_mobile_recognition.hef"
    "bpe_simple_vocab_16e6.txt.gz"
    "ppocrv5_dict.txt"
)

MISSING=()
for f in "${MODEL_FILES[@]}"; do
    if [[ -f "$SCRIPT_DIR/models/$f" ]]; then
        ok "$f"
    else
        MISSING+=("$f")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo ""
    echo "  $(red "Missing ${#MISSING[@]} model file(s):")"

    HEF_BASE="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8"
    HEF_BASE_V218="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8"
    BPE_URL="https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
    DICT_URL="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv5_dict.txt"

    # Build download list: (file, url) pairs
    DOWNLOADS=()
    for f in "${MISSING[@]}"; do
        case "$f" in
            scrfd_2.5g.hef|arcface_r50.hef|tinyclip_*)
                DOWNLOADS+=("$f|$HEF_BASE/$f") ;;
            paddle_ocr_*)
                DOWNLOADS+=("$f|$HEF_BASE_V218/$f") ;;
            bpe_simple_vocab_16e6.txt.gz)
                DOWNLOADS+=("$f|$BPE_URL") ;;
            ppocrv5_dict.txt)
                DOWNLOADS+=("$f|$DICT_URL") ;;
        esac
    done

    for entry in "${DOWNLOADS[@]}"; do
        echo "    - ${entry%%|*}"
    done

    echo ""
    read -rp "  Download all missing files now? [Y/n] " answer
    if [[ "${answer:-y}" =~ ^[Yy]$ ]]; then
        mkdir -p "$SCRIPT_DIR/models"
        for entry in "${DOWNLOADS[@]}"; do
            file="${entry%%|*}"
            url="${entry#*|}"
            echo "  Downloading $file..."
            curl -Lo "$SCRIPT_DIR/models/$file" "$url"
            ok "$file"
        done
    else
        echo ""
        echo "  Download manually with:"
        for entry in "${DOWNLOADS[@]}"; do
            file="${entry%%|*}"
            url="${entry#*|}"
            echo "    curl -Lo models/$file $url"
        done
        exit 1
    fi
fi

# ── Step 3: Build base image ─────────────────────────────────────────

step 3 "Building base image: $IMAGE_BASE"

docker build \
    --build-arg HAILORT_VERSION="$HAILORT_VERSION" \
    --build-arg DEB_ARCH="$DEB_ARCH" \
    --build-arg WHL_ARCH="$WHL_ARCH" \
    -t "$IMAGE_BASE" \
    -f "$SCRIPT_DIR/Dockerfile.hailo-base" \
    "$SCRIPT_DIR"

ok "$IMAGE_BASE"

# ── Step 4: Extract TinyCLIP text weights ─────────────────────────────

step 4 "Extracting TinyCLIP text weights"

if [[ -f "$SCRIPT_DIR/models/tinyclip_text_weights.npz" ]]; then
    echo "  models/tinyclip_text_weights.npz already exists, skipping."
    echo "  (Delete it to re-extract.)"
else
    "$SCRIPT_DIR/scripts/extract_tinyclip_weights.sh"
fi

ok "models/tinyclip_text_weights.npz"

# ── Step 5: Build application image ───────────────────────────────────

step 5 "Building application image: $IMAGE_APP"

docker build \
    -t "$IMAGE_APP" \
    -f "$SCRIPT_DIR/Dockerfile.immich-ml-hailo" \
    "$SCRIPT_DIR"

ok "$IMAGE_APP"

# ── Step 6: Run tests ────────────────────────────────────────────────

step 6 "Running tests"

if [[ ! -f "$TEST_IMAGE" ]]; then
    echo "  $(red 'Test image not found:')" "$TEST_IMAGE"
    echo "  Skipping tests. Provide a test image to run them:"
    echo "    ./setup.sh /path/to/photo.jpg"
    echo ""
    bold "Setup complete (tests skipped)."; echo ""
    echo "Run the container with:"
    echo "  docker run -itd --device=/dev/hailo0:/dev/hailo0 --group-add=0 -p 3003:3003 $IMAGE_APP"
    exit 0
fi

# Stop any existing container on port 3003
EXISTING=$(docker ps -q --filter "publish=3003" 2>/dev/null || true)
if [[ -n "$EXISTING" ]]; then
    echo "  Stopping existing container on port 3003..."
    docker rm -f "$EXISTING" >/dev/null 2>&1 || true
    sleep 2
fi

# Start a test container
CONTAINER_NAME="immich-ml-setup-test-$$"
echo "  Starting test container: $CONTAINER_NAME"
if ! docker run -d \
    --device=/dev/hailo0:/dev/hailo0 \
    --group-add=0 \
    -p 3003:3003 \
    --name "$CONTAINER_NAME" \
    "$IMAGE_APP"; then
    fail "Failed to start container (port 3003 in use?)"
fi

cleanup() {
    echo "  Stopping test container..."
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Wait for pipeline to fully initialize (loading 6 HEFs can take >30s)
echo "  Waiting for pipeline to initialize..."
for i in $(seq 1 120); do
    READY=$(docker exec "$CONTAINER_NAME" \
        python3 -c "
import urllib.request, json, sys
try:
    req = urllib.request.Request('http://localhost:3003/predict', method='POST')
    # Can't easily do multipart from urllib, just check /ping + a quick test
    r = urllib.request.urlopen('http://localhost:3003/').read()
    d = json.loads(r)
    if d.get('message') == 'Immich ML':
        print('ready')
except:
    pass
" 2>/dev/null || true)
    if [[ "$READY" == "ready" ]]; then
        break
    fi
    if [[ $i -eq 120 ]]; then
        echo "  $(red 'Service did not become ready within 120 seconds.')"
        docker logs "$CONTAINER_NAME" 2>&1 | tail -30
        exit 1
    fi
    if (( i % 10 == 0 )); then
        echo "  ... still waiting (${i}s)"
    fi
    sleep 1
done
echo "  Service ready."

# Copy test files into container and run tests inside
echo ""
docker cp "$SCRIPT_DIR/tests/test.sh" "$CONTAINER_NAME:/tmp/test.sh"
docker cp "$TEST_IMAGE" "$CONTAINER_NAME:/tmp/test_image.jpg"
docker exec "$CONTAINER_NAME" bash /tmp/test.sh /tmp/test_image.jpg

echo ""
bold "Setup complete. All tests passed."; echo ""
echo "Run the container with:"
echo "  docker run -itd --device=/dev/hailo0:/dev/hailo0 --group-add=0 -p 3003:3003 $IMAGE_APP"
