#!/usr/bin/env bash
#
# Golden-embedding regression test.
#
# test.sh asserts embedding *dimensions*. This asserts embedding *values*. The
# failure mode that matters — wrong quantization parameters, a mis-shaped batch,
# a silently swapped model — produces a correctly-shaped, correctly-normalised,
# subtly-wrong vector that passes all of test.sh's assertions. Only comparing
# against a known-good reference catches it.
#
# Usage:
#   ./tests/golden.sh generate [image]   # capture references from the running container
#   ./tests/golden.sh check    [image]   # compare against the stored references
#
# Environment:
#   BASE_URL    default http://localhost:3003
#   CONTAINER   container name/id; auto-detected from the published port if unset
#   GOLDEN_DIR  where references live; default tests/golden/
#   SAMPLES     repeats used to measure the noise floor during generate; default 10
#
# REFERENCES ARE NOT PORTABLE. They are specific to one HEF build on one device.
# A Hailo-8 reference will not match a Hailo-8L, and a Model Zoo version bump
# recompiles the HEF and changes the numbers. Regenerate deliberately whenever
# the model, the HEF version, or the device changes — never to "make the test
# pass". They are gitignored for this reason.
#
# Run this on the Docker host.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="${1:-}"
IMAGE="${2:-$PROJECT_DIR/tests/test.jpg}"
BASE_URL="${BASE_URL:-http://localhost:3003}"
GOLDEN_DIR="${GOLDEN_DIR:-$PROJECT_DIR/tests/golden}"
SAMPLES="${SAMPLES:-10}"

TEXT_QUERY="a photo of a dog"   # fixed so the textual reference is reproducible

red()   { printf "\033[31m%s\033[0m" "$*"; }
green() { printf "\033[32m%s\033[0m" "$*"; }
yellow(){ printf "\033[33m%s\033[0m" "$*"; }
bold()  { printf "\033[1m%s\033[0m" "$*"; }

die() { echo "  $(red FAIL): $1"; exit 1; }

usage() {
    echo "usage: $0 {generate|check} [image]"
    exit 2
}

case "$MODE" in
    generate|check) ;;
    *) usage ;;
esac

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# ── Preflight ─────────────────────────────────────────────────────────

echo ""
bold "=== golden embeddings: $MODE ==="; echo ""

command -v docker >/dev/null 2>&1 || die "docker not found — run this on the Docker host"
command -v curl   >/dev/null 2>&1 || die "curl not found"
[[ -f "$IMAGE" ]] || die "test image not found: $IMAGE"

curl -sf "$BASE_URL/ping" >/dev/null 2>&1 \
    || die "service not reachable at $BASE_URL/ping — start the container first"

CONTAINER="${CONTAINER:-$(docker ps -q --filter "publish=3003" 2>/dev/null | head -1)}"
[[ -n "$CONTAINER" ]] || die "no running container publishing port 3003 — set CONTAINER=<name>"

IMAGE_TAG=$(docker inspect -f '{{.Config.Image}}' "$CONTAINER" 2>/dev/null || echo "unknown")
BACKEND=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" 2>/dev/null \
          | awk -F= '/^CLIP_BACKEND=/{print $2}' | head -1)
BACKEND="${BACKEND:-tinyclip}"
DETECTOR=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" 2>/dev/null \
           | awk -F= '/^FACE_DETECTOR=/{print $2}' | head -1)
DETECTOR="${DETECTOR:-scrfd_2.5g}"
RECOGNIZER=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" 2>/dev/null \
             | awk -F= '/^FACE_RECOGNIZER=/{print $2}' | head -1)
RECOGNIZER="${RECOGNIZER:-arcface_r50}"
IMAGE_SUM=$(cksum < "$IMAGE" | awk '{print $1"-"$2}')

# Keyed by BOTH the CLIP backend and the face detector.
#
# The CLIP references depend only on the backend, but the face reference
# depends on the detector: a different detector finds different faces, at
# slightly different boxes, so the crops differ and therefore the ArcFace
# embeddings differ — and the face count itself may change. The recognizer
# changes the face embedding directly and completely. A single-key reference
# would make either switch look like an embedding regression.
#
# Keying the whole file by both duplicates the CLIP vectors across detectors,
# which is a few KB and buys a much better failure mode: an unseen combination
# SKIPS with an explicit message instead of FAILING.
REF_FILE="$GOLDEN_DIR/${BACKEND}__${DETECTOR}__${RECOGNIZER}.json"

# Python does the vector maths. Prefer the host interpreter; fall back to the
# container's, since a bare Unraid host has no python3. Programs are fed on
# stdin with their data embedded, so no path has to be valid in both places.
if command -v python3 >/dev/null 2>&1; then
    py() { python3 - "$@"; }
else
    py() { docker exec -i "$CONTAINER" python3 - "$@"; }
fi

echo "  Target:    $BASE_URL"
echo "  Container: $IMAGE_TAG"
echo "  Backend:   $BACKEND"
echo "  Detector:  $DETECTOR"
echo "  Recognizer: $RECOGNIZER"
echo "  Image:     $IMAGE"
echo "  Reference: $REF_FILE"
echo ""

# ── Request helpers ───────────────────────────────────────────────────

# Appends "<kind>\t<response-json>" to $WORK/samples.tsv
sample() {
    local kind="$1" entries="$2" body_kind="$3" body="$4" out
    out=$(mktemp "$WORK/resp.XXXXXX")
    if [[ "$body_kind" == "image" ]]; then
        curl -sf -X POST "$BASE_URL/predict" -F "entries=$entries" -F "image=@$body" -o "$out" \
            || die "request failed: $kind"
    else
        curl -sf -X POST "$BASE_URL/predict" -F "entries=$entries" -F "text=$body" -o "$out" \
            || die "request failed: $kind"
    fi
    printf '%s\t' "$kind" >> "$WORK/samples.tsv"
    tr -d '\n' < "$out" >> "$WORK/samples.tsv"
    printf '\n' >> "$WORK/samples.tsv"
    rm -f "$out"
}

E_VISUAL='{"clip":{"visual":{"modelName":"golden"}}}'
E_TEXTUAL='{"clip":{"textual":{"modelName":"golden"}}}'
E_FACE='{"facial-recognition":{"detection":{"modelName":"golden","options":{"minScore":0.7}},"recognition":{"modelName":"golden"}}}'

collect_round() {
    sample clip_visual  "$E_VISUAL"  image "$IMAGE"
    sample clip_textual "$E_TEXTUAL" text  "$TEXT_QUERY"
    sample face         "$E_FACE"    image "$IMAGE"
}

# Embeds a file as a raw Python string. Safe here: the payloads are JSON, which
# can contain neither a triple quote nor a trailing backslash.
emit_blob() {
    local name="$1" file="$2"
    printf "%s = r'''\n" "$name"
    cat "$file"
    printf "'''\n"
}

# ── generate ──────────────────────────────────────────────────────────

if [[ "$MODE" == "generate" ]]; then
    case "$SAMPLES" in ''|*[!0-9]*) die "SAMPLES must be a positive integer" ;; esac
    [[ "$SAMPLES" -ge 3 ]] || die "SAMPLES must be at least 3 to measure a spread"

    echo "  Measuring run-to-run spread over $SAMPLES repeats..."
    : > "$WORK/samples.tsv"
    for ((i = 0; i < SAMPLES; i++)); do
        collect_round
        printf '.'
    done
    echo ""; echo ""

    mkdir -p "$GOLDEN_DIR"

    {
        emit_blob SAMPLES_BLOB "$WORK/samples.tsv"
        cat <<'PY'
import json, sys, datetime

backend, image_tag, image_sum, text_query, detector, recognizer = sys.argv[1:7]

def unit(v):
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v

def cos(a, b):
    return sum(x * y for x, y in zip(unit(a), unit(b)))

# Collect every sample per kind.
per_kind = {}
face_counts = []
for line in SAMPLES_BLOB.strip().split("\n"):
    if not line.strip():
        continue
    kind, _, payload = line.partition("\t")
    r = json.loads(payload)
    if kind == "face":
        faces = r.get("facial-recognition")
        if not isinstance(faces, list):
            continue
        face_counts.append(len(faces))
        if not faces:
            continue
        emb = json.loads(faces[0]["embedding"])
    else:
        v = r.get("clip")
        if not isinstance(v, str):
            continue
        emb = json.loads(v)
    per_kind.setdefault(kind, []).append(emb)

report, refs = [], {}
worst_noise = 0.0

for kind in ("clip_visual", "clip_textual", "face"):
    runs = per_kind.get(kind, [])
    if len(runs) < 2:
        report.append((kind, None, None, len(runs)))
        continue
    dims = {len(r) for r in runs}
    if len(dims) != 1:
        print(f"ERROR: {kind} returned inconsistent dimensions {sorted(dims)}", file=sys.stderr)
        sys.exit(1)
    # Every run against the first: the minimum is the observed noise floor.
    sims = [cos(runs[0], r) for r in runs[1:]]
    lo = min(sims)
    noise = max(0.0, 1.0 - lo)
    worst_noise = max(worst_noise, noise)
    report.append((kind, lo, dims.pop(), len(runs)))
    refs[kind] = runs[0]

# Threshold: ten times the worst observed deviation, never tighter than 1e-5.
# A real regression (wrong quantization, a mis-shaped batch, a swapped model)
# moves cosine similarity by orders of magnitude more than device jitter, so
# generous headroom costs nothing in sensitivity and avoids false alarms.
margin = max(worst_noise * 10.0, 1e-5)
threshold = 1.0 - margin

print("  measured run-to-run cosine similarity:", file=sys.stderr)
for kind, lo, dim, n in report:
    if lo is None:
        print(f"    {kind:14s} not captured ({n} usable sample(s)) — skipped", file=sys.stderr)
    else:
        print(f"    {kind:14s} min={lo:.12f}  dim={dim}  n={n}", file=sys.stderr)

if worst_noise == 0.0:
    print("\n  Device is bit-exact across repeats (all similarities == 1.0).", file=sys.stderr)
else:
    print(f"\n  Worst observed deviation: {worst_noise:.3e}", file=sys.stderr)
print(f"  Threshold set to {threshold:.9f} "
      f"(margin {margin:.3e} = 10x observed, floor 1e-5)", file=sys.stderr)
if threshold < 0.99:
    print("  WARNING: that is a very loose threshold. This device is unusually "
          "non-deterministic; investigate before trusting the test.", file=sys.stderr)

if not refs:
    print("ERROR: captured no usable embeddings", file=sys.stderr)
    sys.exit(1)

doc = {
    "_comment": "Golden reference embeddings. Specific to this HEF build and this "
                "device — NOT portable across Hailo-8/8L or Model Zoo versions. "
                "Regenerate deliberately when the model, HEF version or device changes.",
    "backend": backend,
    "face_detector": detector,
    "face_recognizer": recognizer,
    "image_tag": image_tag,
    "test_image_cksum": image_sum,
    "text_query": text_query,
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "samples": len(report and per_kind.get("clip_visual", [])),
    "worst_observed_deviation": worst_noise,
    "threshold": threshold,
    "face_count": max(face_counts) if face_counts else 0,
    "embeddings": refs,
}
json.dump(doc, sys.stdout, separators=(",", ":"))
PY
    } | py "$BACKEND" "$IMAGE_TAG" "$IMAGE_SUM" "$TEXT_QUERY" "$DETECTOR" "$RECOGNIZER" > "$WORK/ref.json"

    mv "$WORK/ref.json" "$REF_FILE"
    echo ""
    echo "  $(green OK): wrote $REF_FILE"
    echo ""
    echo "  Regenerate this whenever the model, the HEF version, or the device"
    echo "  changes. A stale reference fails with a large similarity drop that"
    echo "  looks like a regression."
    echo ""
    exit 0
fi

# ── check ─────────────────────────────────────────────────────────────

if [[ ! -f "$REF_FILE" ]]; then
    echo "  $(yellow SKIP): no reference for CLIP backend '$BACKEND' + detector '$DETECTOR' + recognizer '$RECOGNIZER'"
    echo ""
    echo "  Expected: $REF_FILE"
    echo "  Create it against a build you trust:"
    echo ""
    echo "      ./tests/golden.sh generate"
    echo ""
    echo "  References are per-device and per-HEF, so they are not shipped in"
    echo "  the repo and must be generated on this deployment."
    echo ""
    echo "  Each CLIP-backend / face-detector combination needs its own reference:"
    echo "  the detector changes which faces are found and where, and the"
    echo "  recognizer changes the face embedding outright. Switching either is a"
    echo "  configuration change, not a regression — regenerate rather than debug."
    echo ""
    exit 0
fi

: > "$WORK/samples.tsv"
collect_round

{
    emit_blob SAMPLES_BLOB "$WORK/samples.tsv"
    emit_blob REF_BLOB "$REF_FILE"
    cat <<'PY'
import json, sys

image_tag, image_sum = sys.argv[1:3]
ref = json.loads(REF_BLOB)
threshold = float(ref["threshold"])

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

drift = []
if ref.get("image_tag") != image_tag:
    drift.append(f"container image {ref.get('image_tag')!r} -> {image_tag!r}")
if ref.get("test_image_cksum") != image_sum:
    drift.append("test image contents changed")
for d in drift:
    print(f"  {YELLOW}NOTE{RESET}: reference was captured against a different {d}")
if drift:
    print("        A failure below may mean the reference is stale, not that the "
          "worker regressed.\n")

def unit(v):
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v

def cos(a, b):
    return sum(x * y for x, y in zip(unit(a), unit(b)))

live, face_count = {}, None
for line in SAMPLES_BLOB.strip().split("\n"):
    if not line.strip():
        continue
    kind, _, payload = line.partition("\t")
    r = json.loads(payload)
    if kind == "face":
        faces = r.get("facial-recognition")
        if isinstance(faces, list):
            face_count = len(faces)
            if faces:
                live[kind] = json.loads(faces[0]["embedding"])
    else:
        v = r.get("clip")
        if isinstance(v, str):
            live[kind] = json.loads(v)

print(f"  threshold: cosine >= {threshold:.9f} "
      f"(from a measured deviation of {ref.get('worst_observed_deviation', 0.0):.3e})\n")

failed = skipped = passed = 0

for kind, reference in ref["embeddings"].items():
    got = live.get(kind)
    if got is None:
        print(f"  {YELLOW}SKIP{RESET}: {kind} — not present in this response")
        skipped += 1
        continue
    if len(got) != len(reference):
        print(f"  {RED}FAIL{RESET}: {kind} dimension {len(got)} != reference {len(reference)}")
        failed += 1
        continue
    c = cos(reference, got)
    if c >= threshold:
        print(f"  {GREEN}PASS{RESET}: {kind:14s} cos={c:.12f}  (dim {len(got)})")
        passed += 1
    else:
        print(f"  {RED}FAIL{RESET}: {kind:14s} cos={c:.12f} < {threshold:.9f}  (dim {len(got)})")
        failed += 1

exp_faces = ref.get("face_count")
if exp_faces is not None and face_count is not None:
    if face_count == exp_faces:
        print(f"  {GREEN}PASS{RESET}: face_count      {face_count}")
        passed += 1
    else:
        print(f"  {RED}FAIL{RESET}: face_count      {face_count} != reference {exp_faces}")
        failed += 1

print(f"\n  {GREEN}PASS: {passed}{RESET}  {RED}FAIL: {failed}{RESET}  {YELLOW}SKIP: {skipped}{RESET}")
if failed:
    print(f"\n  {RED}Embeddings differ from the reference.{RESET}")
    print("  Either something changed how tensors reach the device, or the")
    print("  reference is stale. Confirm which before regenerating it —")
    print("  regenerating to make the test pass discards the only signal.")
    sys.exit(1)
print(f"\n  {GREEN}Embeddings match the reference.{RESET}")
PY
} | py "$IMAGE_TAG" "$IMAGE_SUM"

echo ""
