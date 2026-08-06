#!/usr/bin/env bash
#
# Full setup: check prerequisites, choose CLIP backend, build images,
# extract weights, run tests.
#
# Usage:
#   HAILORT_VERSION=<ver> ./setup.sh                    # uses tests/test.jpg
#   HAILORT_VERSION=<ver> ./setup.sh /path/to/img.jpg   # uses a custom image
#
# HAILORT_VERSION is REQUIRED — see the check below for why there is no default.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_IMAGE="${1:-$SCRIPT_DIR/tests/test.jpg}"

red()   { printf "\033[31m%s\033[0m" "$*"; }
green() { printf "\033[32m%s\033[0m" "$*"; }
yellow(){ printf "\033[33m%s\033[0m" "$*"; }
bold()  { printf "\033[1m%s\033[0m" "$*"; }

step() { echo ""; bold "[$1/$TOTAL] $2"; echo ""; }
ok()   { echo "  $(green OK): $1"; }
fail() { echo "  $(red FAIL): $1"; exit 1; }

TOTAL=6

# ── HailoRT version (required) ────────────────────────────────────────
#
# The HailoRT userspace library in the image must match the hailo_pci kernel
# module on the host EXACTLY — any difference, even a patch bump, fails at
# VDevice() with HAILO_INVALID_DRIVER_VERSION(76).
#
# There is deliberately no default. A guessed version builds and tags cleanly
# and only dies at runtime, which is the failure mode this whole script exists
# to prevent. Both Dockerfiles already refuse to build without it; this keeps
# setup.sh symmetric with them.

if [[ -z "${HAILORT_VERSION:-}" ]]; then
    echo ""
    echo "  $(red 'ERROR'): HAILORT_VERSION is required — there is no default."
    echo ""
    echo "  It must match the hailo_pci kernel module on this host exactly."
    echo "  Read the host's version with either of:"
    echo ""
    echo "    modinfo hailo_pci | grep '^version:'"
    echo "    cat /sys/module/hailo_pci/version"
    echo ""
    echo "  Then re-run with that version, for example:"
    echo ""
    echo "    HAILORT_VERSION=4.24.0 $0"
    echo "    HAILORT_VERSION=4.24.0 $0 /path/to/photo.jpg"
    echo ""
    echo "  The matching hailort_<ver>_<arch>.deb and"
    echo "  hailort-<ver>-cp312-cp312-linux_<arch>.whl must be in hailo-rt-4/."
    echo ""
    exit 1
fi

export HAILORT_VERSION   # scripts/extract_*_weights.sh require this too
IMAGE_BASE="hailo-base:v${HAILORT_VERSION}"
IMAGE_APP="immich-ml-hailo:v${HAILORT_VERSION}"

# URLs
HEF_BASE="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8"
HEF_V218="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8"
HF_SIGLIP="https://huggingface.co/google/siglip-base-patch16-224/resolve/main"
BPE_URL="https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
DICT_URL="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv5_dict.txt"

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

# ── Step 2: Check and download model files ───────────────────────────

step 2 "Checking model files in models/"

# Common models (always needed)
COMMON_FILES=(
    "scrfd_2.5g.hef|$HEF_BASE/scrfd_2.5g.hef"
    # Both face detectors are fetched, for the same reason both CLIP backends
    # are: scrfd_10g is 6.9 MB, and having it present makes FACE_DETECTOR a
    # restart rather than a download.
    "scrfd_10g.hef|$HEF_BASE/scrfd_10g.hef"
    "arcface_r50.hef|$HEF_BASE/arcface_r50.hef"
    "paddle_ocr_v5_mobile_detection.hef|$HEF_V218/paddle_ocr_v5_mobile_detection.hef"
    "paddle_ocr_v5_mobile_recognition.hef|$HEF_V218/paddle_ocr_v5_mobile_recognition.hef"
    "ppocrv5_dict.txt|$DICT_URL"
)

# Backend-specific models
TINYCLIP_FILES=(
    "tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder.hef|$HEF_BASE/tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder.hef"
    "tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder.hef|$HEF_BASE/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder.hef"
    "bpe_simple_vocab_16e6.txt.gz|$BPE_URL"
)

SIGLIP_FILES=(
    "siglip_b_16_image_encoder.hef|$HEF_V218/siglip_b_16_image_encoder.hef"
    "siglip_b_16_text_encoder.hef|$HEF_V218/siglip_b_16_text_encoder.hef"
    "spiece.model|$HF_SIGLIP/spiece.model"
)

# Both backends are included in the image so you can switch at runtime
REQUIRED_FILES=("${COMMON_FILES[@]}" "${TINYCLIP_FILES[@]}" "${SIGLIP_FILES[@]}")

DOWNLOADS=()
for entry in "${REQUIRED_FILES[@]}"; do
    file="${entry%%|*}"
    if [[ -f "$SCRIPT_DIR/models/$file" ]]; then
        ok "$file"
    else
        DOWNLOADS+=("$entry")
    fi
done

if [[ ${#DOWNLOADS[@]} -gt 0 ]]; then
    echo ""
    echo "  $(red "Missing ${#DOWNLOADS[@]} file(s):")"
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
            # -f so an HTTP error is a failure, not an HTML error page
            # written into a .hef that then fails cryptically at load time.
            curl -fLo "$SCRIPT_DIR/models/$file" "$url" \
                || fail "download failed ($url) — check the URL and your connection"
            ok "$file"
        done
    else
        echo ""
        echo "  Download manually with:"
        for entry in "${DOWNLOADS[@]}"; do
            file="${entry%%|*}"
            url="${entry#*|}"
            echo "    curl -fLo models/$file $url"
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

# ── Step 4: Extract CLIP text weights ─────────────────────────────────

step 4 "Extracting CLIP text weights (both backends)"

if [[ -f "$SCRIPT_DIR/models/tinyclip_text_weights.npz" ]]; then
    echo "  models/tinyclip_text_weights.npz already exists, skipping."
else
    "$SCRIPT_DIR/scripts/extract_tinyclip_weights.sh"
fi
ok "models/tinyclip_text_weights.npz"

if [[ -f "$SCRIPT_DIR/models/siglip_text_weights.npz" ]]; then
    echo "  models/siglip_text_weights.npz already exists, skipping."
else
    "$SCRIPT_DIR/scripts/extract_siglip_weights.sh"
fi
ok "models/siglip_text_weights.npz"

# ── Step 5: Build application image ───────────────────────────────────

step 5 "Building application image: $IMAGE_APP"

docker build \
    --build-arg HAILORT_VERSION="$HAILORT_VERSION" \
    -t "$IMAGE_APP" \
    -f "$SCRIPT_DIR/Dockerfile.immich-ml-hailo" \
    "$SCRIPT_DIR"

ok "$IMAGE_APP"

# ── Step 6: Run tests (both backends) ────────────────────────────────

step 6 "Running tests"

if [[ ! -f "$TEST_IMAGE" ]]; then
    echo "  $(red 'Test image not found:')" "$TEST_IMAGE"
    echo "  Skipping tests. Provide a test image to run them:"
    echo "    ./setup.sh /path/to/photo.jpg"
    echo ""
    bold "Setup complete (tests skipped)."; echo ""
    echo "Run the container with one of:"
    echo "  docker run -d --device=/dev/hailo0:/dev/hailo0 --group-add=0 -p 3003:3003 -e CLIP_BACKEND=siglip $IMAGE_APP"
    echo "  docker run -d --device=/dev/hailo0:/dev/hailo0 --group-add=0 -p 3003:3003 -e CLIP_BACKEND=tinyclip $IMAGE_APP"
    exit 0
fi

# Containers started by this script, removed on ANY exit.
#
# Without this, a failing test suite tripped `set -e` before the cleanup line
# below could run, leaving a container bound to port 3003. The next run then
# died with "port is already allocated" — a confusing second failure masking
# the first. The trap covers every exit path: test failure, Ctrl-C, or a fault
# anywhere else in the script.
TEST_CONTAINERS=()

cleanup_test_containers() {
    local c
    for c in ${TEST_CONTAINERS[@]+"${TEST_CONTAINERS[@]}"}; do
        docker rm -f "$c" >/dev/null 2>&1 || true
    done
}
trap cleanup_test_containers EXIT

# Wait until nothing holds port 3003. `docker rm -f` returns before the port is
# actually released, which is the other half of "port is already allocated".
wait_for_port_free() {
    local i
    for i in $(seq 1 30); do
        if [[ -z "$(docker ps -q --filter "publish=3003" 2>/dev/null || true)" ]]; then
            sleep 1   # brief settle for the port to be released by the daemon
            return 0
        fi
        sleep 1
    done
    return 1
}

run_test_with_backend() {
    local backend="$1"
    local container="immich-ml-setup-test-${backend}-$$"

    echo ""
    bold "  Testing CLIP_BACKEND=$backend"; echo ""

    # Stop any existing container on port 3003
    EXISTING=$(docker ps -q --filter "publish=3003" 2>/dev/null || true)
    if [[ -n "$EXISTING" ]]; then
        docker rm -f "$EXISTING" >/dev/null 2>&1 || true
        wait_for_port_free || echo "  $(yellow 'WARNING'): port 3003 still held; starting anyway"
    fi

    TEST_CONTAINERS+=("$container")

    if ! docker run -d \
        --device=/dev/hailo0:/dev/hailo0 \
        --group-add=0 \
        -p 3003:3003 \
        -e CLIP_BACKEND="$backend" \
        --name "$container" \
        "$IMAGE_APP"; then
        fail "Failed to start container for $backend"
    fi

    # Wait for pipeline to initialize
    echo "  Waiting for pipeline to initialize..."
    for i in $(seq 1 120); do
        READY=$(docker exec "$container" \
            python3 -c "
import urllib.request, json
try:
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
            echo "  $(red "Service did not become ready within 120 seconds ($backend).")"
            docker logs "$container" 2>&1 | tail -30
            docker rm -f "$container" >/dev/null 2>&1 || true
            return 1
        fi
        if (( i % 10 == 0 )); then
            echo "  ... still waiting (${i}s)"
        fi
        sleep 1
    done
    echo "  Service ready."

    # Copy test files and run
    echo ""
    docker cp "$SCRIPT_DIR/tests/test.sh" "$container:/tmp/test.sh"
    docker cp "$TEST_IMAGE" "$container:/tmp/test_image.jpg"
    # `|| result=$?` keeps `set -e` from aborting here — the removal below must
    # run whether the suite passed or failed.
    local result=0
    docker exec "$container" bash /tmp/test.sh /tmp/test_image.jpg || result=$?

    docker rm -f "$container" >/dev/null 2>&1 || true
    return "$result"
}

# `if !` so a failure is reported rather than silently aborting via set -e.
# The EXIT trap still removes the container either way.
if ! run_test_with_backend "tinyclip"; then
    fail "test suite failed for CLIP_BACKEND=tinyclip"
fi
if ! run_test_with_backend "siglip"; then
    fail "test suite failed for CLIP_BACKEND=siglip"
fi

echo ""
bold "Setup complete. All tests passed."; echo ""
echo "Run the container with one of:"
echo ""
echo "  $(bold 'SigLIP') (better quality, Immich-compatible embeddings):"
echo "  docker run -d --device=/dev/hailo0:/dev/hailo0 --group-add=0 -p 3003:3003 -e CLIP_BACKEND=siglip $IMAGE_APP"
echo ""
echo "  $(bold 'TinyCLIP') (faster inference):"
echo "  docker run -d --device=/dev/hailo0:/dev/hailo0 --group-add=0 -p 3003:3003 -e CLIP_BACKEND=tinyclip $IMAGE_APP"
