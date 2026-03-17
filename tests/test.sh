#!/usr/bin/env bash
#
# End-to-end test for immich-ml-hailo.
# Requires: a running container on port 3003, a test image.
#
# Usage:
#   ./tests/test.sh                  # uses tests/test.jpg
#   ./tests/test.sh /path/to/img.jpg # uses a custom image
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_URL="${BASE_URL:-http://localhost:3003}"
IMAGE="${1:-$SCRIPT_DIR/test.jpg}"
PASS=0
FAIL=0
SKIP=0

TMPDIR_TEST=$(mktemp -d)
trap "rm -rf $TMPDIR_TEST" EXIT

# ── helpers ───────────────────────────────────────────────────────────

red()   { printf "\033[31m%s\033[0m" "$*"; }
green() { printf "\033[32m%s\033[0m" "$*"; }
yellow(){ printf "\033[33m%s\033[0m" "$*"; }
bold()  { printf "\033[1m%s\033[0m" "$*"; }

pass() { PASS=$((PASS+1)); echo "  $(green PASS): $1"; }
fail() { FAIL=$((FAIL+1)); echo "  $(red FAIL): $1"; }
skip() { SKIP=$((SKIP+1)); echo "  $(yellow SKIP): $1"; }

# check LABEL RESP_FILE PYTHON_CODE
# Runs python code with the response file path available as sys.argv[1].
check() {
    local label="$1" resp_file="$2"; shift 2
    if python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    RESP = f.read()
try:
    r = json.loads(RESP)
except:
    sys.exit(1)
$*
" "$resp_file" 2>/dev/null; then
        pass "$label"
    else
        fail "$label"
    fi
}

# ── preflight ─────────────────────────────────────────────────────────

echo ""
bold "=== immich-ml-hailo test suite ==="; echo ""
echo "Target: $BASE_URL"
echo "Image:  $IMAGE"
echo ""

if [[ ! -f "$IMAGE" ]]; then
    echo "$(red ERROR): Test image not found: $IMAGE"
    echo "Provide a JPEG with faces and visible text for best coverage."
    exit 1
fi

if ! curl -sf "$BASE_URL/ping" >/dev/null 2>&1; then
    echo "$(red ERROR): Service not reachable at $BASE_URL/ping"
    echo "Start the container first:"
    echo "  docker run -itd --device=/dev/hailo0:/dev/hailo0 --group-add=0 -p 3003:3003 immich-ml-hailo:v4.23.0"
    exit 1
fi

# ── Test 1: GET / ─────────────────────────────────────────────────────

bold "1. Service discovery (GET /)"; echo ""
R="$TMPDIR_TEST/t1.json"
curl -s "$BASE_URL/" -o "$R"
check "returns JSON with message" "$R" "
assert r.get('message') == 'Immich ML', f'unexpected: {r}'
"

# ── Test 2: GET /ping ─────────────────────────────────────────────────

bold "2. Health check (GET /ping)"; echo ""
RESP=$(curl -s "$BASE_URL/ping")
if [[ "$RESP" == "pong" ]]; then
    pass "returns plain text 'pong'"
else
    fail "expected 'pong', got '$RESP'"
fi

# ── Test 3: CLIP visual ──────────────────────────────────────────────

bold "3. CLIP visual (image -> embedding)"; echo ""
R="$TMPDIR_TEST/t3.json"
curl -s -X POST "$BASE_URL/predict" \
    -F 'entries={"clip":{"visual":{"modelName":"test"}}}' \
    -F "image=@$IMAGE" -o "$R"

check "response has clip key" "$R" "
assert 'clip' in r, f'missing clip key: {list(r.keys())}'
"

check "clip is 512-dim embedding" "$R" "
emb = json.loads(r['clip'])
assert isinstance(emb, list) and len(emb) == 512, f'expected 512-dim, got {len(emb)}'
"

check "imageHeight and imageWidth present" "$R" "
assert 'imageHeight' in r and 'imageWidth' in r, f'missing dimensions: {list(r.keys())}'
assert isinstance(r['imageHeight'], int) and r['imageHeight'] > 0
assert isinstance(r['imageWidth'], int) and r['imageWidth'] > 0
"

# ── Test 4: CLIP text ─────────────────────────────────────────────────

bold "4. CLIP text (text -> embedding)"; echo ""
R="$TMPDIR_TEST/t4.json"
curl -s -X POST "$BASE_URL/predict" \
    -F 'entries={"clip":{"textual":{"modelName":"test"}}}' \
    -F 'text=a photo of a dog' -o "$R"

check "response has clip key" "$R" "
assert 'clip' in r, f'missing clip key: {list(r.keys())}'
"

check "clip is 512-dim embedding" "$R" "
emb = json.loads(r['clip'])
assert isinstance(emb, list) and len(emb) == 512, f'expected 512-dim, got {len(emb)}'
"

check "no imageHeight/imageWidth for text-only" "$R" "
assert 'imageHeight' not in r, 'imageHeight should not be present for text-only'
"

# ── Test 5: CLIP similarity sanity check ──────────────────────────────

bold "5. CLIP similarity sanity check"; echo ""
R_DOG="$TMPDIR_TEST/t5_dog.json"
R_CAT="$TMPDIR_TEST/t5_cat.json"
curl -s -X POST "$BASE_URL/predict" \
    -F 'entries={"clip":{"textual":{"modelName":"test"}}}' \
    -F 'text=a dog' -o "$R_DOG"
curl -s -X POST "$BASE_URL/predict" \
    -F 'entries={"clip":{"textual":{"modelName":"test"}}}' \
    -F 'text=a cat' -o "$R_CAT"

# Special check comparing two response files
if python3 -c "
import json, sys
with open(sys.argv[1]) as f: r1 = json.loads(f.read())
with open(sys.argv[2]) as f: r2 = json.loads(f.read())
e1 = json.loads(r1['clip'])
e2 = json.loads(r2['clip'])
assert e1 != e2, 'dog and cat embeddings should differ'
" "$R_DOG" "$R_CAT" 2>/dev/null; then
    pass "different queries produce different embeddings"
else
    fail "different queries produce different embeddings"
fi

# ── Test 6: Facial recognition ────────────────────────────────────────

bold "6. Facial recognition (image -> faces)"; echo ""
R="$TMPDIR_TEST/t6.json"
curl -s -X POST "$BASE_URL/predict" \
    -F 'entries={"facial-recognition":{"detection":{"modelName":"test","options":{"minScore":0.5}},"recognition":{"modelName":"test"}}}' \
    -F "image=@$IMAGE" -o "$R"

check "response has facial-recognition key" "$R" "
assert 'facial-recognition' in r, f'missing key: {list(r.keys())}'
"

check "facial-recognition is a list" "$R" "
fr = r['facial-recognition']
assert isinstance(fr, list), f'expected list, got {type(fr).__name__}'
"

check "imageHeight and imageWidth present" "$R" "
assert 'imageHeight' in r and 'imageWidth' in r
"

# Check face structure if any faces were found
HAS_FACES=0
python3 -c "
import json, sys
with open(sys.argv[1]) as f: r = json.loads(f.read())
sys.exit(0 if len(r.get('facial-recognition', [])) > 0 else 1)
" "$R" 2>/dev/null && HAS_FACES=1 || true

if [[ "$HAS_FACES" == "1" ]]; then
    check "each face has boundingBox, score, embedding" "$R" "
for f in r['facial-recognition']:
    assert 'boundingBox' in f, 'missing boundingBox'
    bb = f['boundingBox']
    for k in ('x1','y1','x2','y2'):
        assert k in bb, f'missing {k} in boundingBox'
    assert 'score' in f and 0 <= f['score'] <= 1, f'bad score: {f.get(\"score\")}'
    emb = json.loads(f['embedding'])
    assert len(emb) == 512, f'expected 512-dim embedding, got {len(emb)}'
"
else
    skip "no faces detected in test image — face structure checks skipped"
fi

# ── Test 7: OCR ───────────────────────────────────────────────────────

bold "7. OCR (image -> text)"; echo ""
R="$TMPDIR_TEST/t7.json"
curl -s -X POST "$BASE_URL/predict" \
    -F 'entries={"ocr":{"detection":{"modelName":"test","options":{"minScore":0.3}},"recognition":{"modelName":"test","options":{"minScore":0.5}}}}' \
    -F "image=@$IMAGE" -o "$R"

# Check if OCR is available (models loaded)
OCR_AVAILABLE=0
python3 -c "
import json, sys
with open(sys.argv[1]) as f: r = json.loads(f.read())
ocr = r.get('ocr', {})
sys.exit(1 if isinstance(ocr, dict) and 'error' in ocr else 0)
" "$R" 2>/dev/null && OCR_AVAILABLE=1 || true

if [[ "$OCR_AVAILABLE" == "1" ]]; then
    check "response has ocr key with correct structure" "$R" "
ocr = r['ocr']
assert isinstance(ocr, dict), f'expected dict, got {type(ocr).__name__}'
for k in ('text', 'box', 'boxScore', 'textScore'):
    assert k in ocr, f'missing key: {k}'
assert isinstance(ocr['text'], list)
assert isinstance(ocr['boxScore'], list)
assert isinstance(ocr['textScore'], list)
assert len(ocr['text']) == len(ocr['boxScore']) == len(ocr['textScore'])
"

    check "imageHeight and imageWidth present" "$R" "
assert 'imageHeight' in r and 'imageWidth' in r
"

    NTEXT=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f: r = json.loads(f.read())
texts = r['ocr']['text']
print(len(texts))
for i, t in enumerate(texts):
    s = r['ocr']['textScore'][i]
    print(f'  [{i}] score={s:.3f}: {t}', flush=True)
" "$R" 2>/dev/null || echo "0")
    echo "  info: $NTEXT text region(s) recognized"
else
    skip "OCR models not loaded — add PaddleOCR HEFs + ppocrv5_dict.txt to models/"
fi

# ── Test 8: Combined request ─────────────────────────────────────────

bold "8. Combined request (face + clip in one call)"; echo ""
R="$TMPDIR_TEST/t8.json"
curl -s -X POST "$BASE_URL/predict" \
    -F 'entries={"facial-recognition":{"detection":{"modelName":"test","options":{"minScore":0.7}},"recognition":{"modelName":"test"}},"clip":{"visual":{"modelName":"test"}}}' \
    -F "image=@$IMAGE" -o "$R"

check "response has both facial-recognition and clip keys" "$R" "
assert 'facial-recognition' in r, f'missing facial-recognition: {list(r.keys())}'
assert 'clip' in r, f'missing clip: {list(r.keys())}'
"

check "clip embedding is valid" "$R" "
emb = json.loads(r['clip'])
assert len(emb) == 512
"

# ── Test 9: Error handling ────────────────────────────────────────────

bold "9. Error handling"; echo ""

# Missing image for visual task
R="$TMPDIR_TEST/t9_missing.json"
curl -s -X POST "$BASE_URL/predict" \
    -F 'entries={"clip":{"visual":{"modelName":"test"}}}' -o "$R"
check "missing image returns error" "$R" "
clip = r.get('clip', {})
assert isinstance(clip, dict) and 'error' in clip, f'expected error: {r}'
"

# Invalid entries JSON
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/predict" \
    -F 'entries=not-valid-json')
if [[ "$HTTP_CODE" == "400" ]]; then
    pass "invalid JSON returns 400"
else
    fail "invalid JSON returned $HTTP_CODE, expected 400"
fi

# ── Summary ───────────────────────────────────────────────────────────

echo ""
bold "=== Results ==="; echo ""
echo "  $(green "PASS: $PASS")  $(red "FAIL: $FAIL")  $(yellow "SKIP: $SKIP")"
echo ""

if [[ "$FAIL" -gt 0 ]]; then
    echo "$(red 'Some tests failed.')"
    exit 1
elif [[ "$SKIP" -gt 0 ]]; then
    echo "$(yellow 'All tests passed, some skipped.')"
    exit 0
else
    echo "$(green 'All tests passed.')"
    exit 0
fi
