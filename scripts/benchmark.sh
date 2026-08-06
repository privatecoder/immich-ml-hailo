#!/usr/bin/env bash
#
# Per-stage latency benchmark for a running immich-ml-hailo container.
#
# Fires N requests per task and reports p50/p95 for every pipeline stage, so a
# change can be attributed to a specific stage rather than to an end-to-end
# total. Stage timings come from the structured "/predict ..." summary line the
# worker emits at INFO — this script fires requests, then reads them back out of
# `docker logs`.
#
# Why parse the log rather than instrument the response: it needs no code
# change and no rebuild, so it measures the container you already have running,
# in production, exactly as it is. Adding a timing header would mean rebuilding
# the very thing under measurement.
#
# READ-ONLY. It sends /predict requests (inference has no side effects on the
# worker) and reads `docker logs` / `docker inspect`. It never restarts,
# rebuilds, or reconfigures anything.
#
# Usage:
#   ./scripts/benchmark.sh                      # tests/test.jpg, 20 iterations
#   ./scripts/benchmark.sh /path/to/img.jpg     # custom image, 20 iterations
#   ./scripts/benchmark.sh /path/to/img.jpg 50  # custom image, 50 iterations
#
# Environment:
#   BASE_URL    default http://localhost:3003
#   CONTAINER   container name/id; auto-detected from the published port if unset
#
# Run this on the Docker host — it needs `docker logs`.
#
set -euo pipefail

# Force the C locale. Under a comma-decimal locale (de_DE and friends) both
# `sort -n` and awk misparse "71.2" as 71, which silently corrupts every
# percentile. Not hypothetical — it happened on the first run of this script.
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE="${1:-$PROJECT_DIR/tests/test.jpg}"
ITERATIONS="${2:-20}"
BASE_URL="${BASE_URL:-http://localhost:3003}"

red()   { printf "\033[31m%s\033[0m" "$*"; }
green() { printf "\033[32m%s\033[0m" "$*"; }
yellow(){ printf "\033[33m%s\033[0m" "$*"; }
bold()  { printf "\033[1m%s\033[0m" "$*"; }

fail() { echo "  $(red FAIL): $1"; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# ── Preflight ─────────────────────────────────────────────────────────

echo ""
bold "=== immich-ml-hailo benchmark ==="; echo ""

command -v docker >/dev/null 2>&1 || fail "docker not found — run this on the Docker host"
command -v curl   >/dev/null 2>&1 || fail "curl not found"
command -v awk    >/dev/null 2>&1 || fail "awk not found"

[[ -f "$IMAGE" ]] || fail "test image not found: $IMAGE"

case "$ITERATIONS" in
    ''|*[!0-9]*) fail "iterations must be a positive integer, got '$ITERATIONS'" ;;
esac
[[ "$ITERATIONS" -ge 1 ]] || fail "iterations must be at least 1"

if ! curl -sf "$BASE_URL/ping" >/dev/null 2>&1; then
    echo "  $(red ERROR): service not reachable at $BASE_URL/ping"
    echo "  Start the container first, or set BASE_URL."
    exit 1
fi

CONTAINER="${CONTAINER:-$(docker ps -q --filter "publish=3003" 2>/dev/null | head -1)}"
[[ -n "$CONTAINER" ]] || fail "no running container publishing port 3003 — set CONTAINER=<name>"

IMAGE_TAG=$(docker inspect -f '{{.Config.Image}}' "$CONTAINER" 2>/dev/null || echo "unknown")
CONTAINER_NAME=$(docker inspect -f '{{.Name}}' "$CONTAINER" 2>/dev/null | sed 's|^/||' || echo "$CONTAINER")
BACKEND=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" 2>/dev/null \
          | awk -F= '/^CLIP_BACKEND=/{print $2}' | head -1)
BACKEND="${BACKEND:-tinyclip (default, not set in env)}"

# The detector changes how many faces are found and therefore the size of the
# recognition batch — two runs are only comparable if it is held constant.
DETECTOR=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" 2>/dev/null \
           | awk -F= '/^FACE_DETECTOR=/{print $2}' | head -1)
DETECTOR="${DETECTOR:-scrfd_2.5g (default, not set in env)}"

# The worker only emits the summary line at INFO or below.
LOGLEVEL=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" 2>/dev/null \
           | awk -F= '/^LOG_LEVEL=/{print $2}' | head -1)
LOGLEVEL="${LOGLEVEL:-INFO}"
case "$(printf '%s' "$LOGLEVEL" | tr '[:lower:]' '[:upper:]')" in
    DEBUG|INFO|NOTSET) ;;
    *) fail "container LOG_LEVEL=$LOGLEVEL suppresses the per-request summary line this script parses. Restart it with LOG_LEVEL=INFO." ;;
esac

echo "  Target:     $BASE_URL"
echo "  Container:  $CONTAINER_NAME ($IMAGE_TAG)"
echo "  Backend:    $BACKEND"
echo "  Detector:   $DETECTOR"
echo "  Image:      $IMAGE"
echo "  Iterations: $ITERATIONS per task"
echo ""
echo "  $(yellow NOTE): pause Immich's ML jobs while benchmarking. Concurrent"
echo "        traffic both contends for the device and lands in the same log."
echo ""

# ── Fire one task's requests, then recover its stage timings ──────────

# bench_task TASK_LABEL ENTRIES_JSON SEND_IMAGE SEND_TEXT [EXTRA_MATCH]
#
# EXTRA_MATCH disambiguates tasks that share a `tasks=` key. CLIP visual and
# CLIP textual are both `tasks=clip`, so without it each would scoop up the
# other's lines and average two unrelated stages together.
bench_task() {
    local label="$1" entries="$2" send_image="$3" send_text="$4" extra="${5:-}"
    local start_epoch elapsed i

    bold "  $label"; echo ""

    start_epoch=$(date +%s)

    for ((i = 0; i < ITERATIONS; i++)); do
        if [[ "$send_image" == "yes" ]]; then
            curl -sf -X POST "$BASE_URL/predict" \
                -F "entries=$entries" \
                -F "image=@$IMAGE" -o /dev/null || fail "request failed for $label"
        else
            curl -sf -X POST "$BASE_URL/predict" \
                -F "entries=$entries" \
                -F "text=$send_text" -o /dev/null || fail "request failed for $label"
        fi
    done

    # +5s of slack covers rounding and any log-flush lag.
    elapsed=$(( $(date +%s) - start_epoch + 5 ))

    local task_key="${entries#*\"}"; task_key="${task_key%%\"*}"
    local raw="$WORK/raw.log"
    if [[ -n "$extra" ]]; then
        docker logs --since "${elapsed}s" "$CONTAINER" 2>&1 \
            | grep -F '/predict ' | grep -F "tasks=$task_key" \
            | grep -F "$extra" > "$raw" || true
    else
        docker logs --since "${elapsed}s" "$CONTAINER" 2>&1 \
            | grep -F '/predict ' | grep -F "tasks=$task_key" > "$raw" || true
    fi

    local seen
    seen=$(wc -l < "$raw" | tr -d ' ')

    if [[ "$seen" -eq 0 ]]; then
        echo "    $(red 'no matching log lines') — is LOG_LEVEL=INFO and is this the right container?"
        echo ""
        return
    fi
    if [[ "$seen" -ne "$ITERATIONS" ]]; then
        echo "    $(yellow WARNING): fired $ITERATIONS requests but matched $seen log lines."
        echo "    Other traffic is hitting this worker; treat these numbers as contaminated."
    fi

    # Split each summary line into stage/value pairs. Facts (no 'ms' suffix)
    # go to a separate file so they can be reported as context.
    awk '{
        for (i = 1; i <= NF; i++) {
            if ($i ~ /^[A-Za-z_]+=[0-9.]+ms$/) {
                split($i, kv, "=");
                v = kv[2]; sub(/ms$/, "", v);
                print kv[1], v > "'"$WORK"'/stages.txt";
            } else if ($i ~ /^[A-Za-z_]+=/ && $i !~ /^total=/) {
                print $i > "'"$WORK"'/facts.txt";
            }
        }
    }' "$raw"

    if [[ -s "$WORK/facts.txt" ]]; then
        echo -n "    context: "
        sort "$WORK/facts.txt" | uniq -c | sort -rn \
            | awk '$2 !~ /^(status|error)=/ {printf "%s ", $2} END {print ""}'
    fi

    printf "    %-20s %10s %10s %6s\n" "stage" "p50" "p95" "n"
    printf "    %-20s %10s %10s %6s\n" "--------------------" "----------" "----------" "------"

    # Stage order follows first appearance in the log line, i.e. pipeline order.
    awk '{print $1}' "$WORK/stages.txt" | awk '!seen[$0]++' > "$WORK/order.txt"

    while read -r stage; do
        awk -v s="$stage" '$1 == s {print $2}' "$WORK/stages.txt" | sort -n > "$WORK/vals.txt"
        awk -v stage="$stage" '
            {v[NR] = $1}
            END {
                if (NR == 0) exit
                # Nearest-rank percentile: index = ceil(p * N), 1-based.
                p50 = v[int((NR * 50 + 99) / 100)]
                p95 = v[int((NR * 95 + 99) / 100)]
                printf "    %-20s %9.1fms %9.1fms %6d\n", stage, p50, p95, NR
            }' "$WORK/vals.txt"
    done < "$WORK/order.txt"

    rm -f "$WORK/stages.txt" "$WORK/facts.txt" "$WORK/order.txt" "$WORK/vals.txt"
    echo ""
}

# ── Run ───────────────────────────────────────────────────────────────

bench_task "clip visual (image -> embedding)" \
    '{"clip":{"visual":{"modelName":"bench"}}}' yes "" "clip=visual"

bench_task "clip textual (text -> embedding)" \
    '{"clip":{"textual":{"modelName":"bench"}}}' no "a photo of a dog" "clip=textual"

bench_task "facial-recognition (detect + recognise)" \
    '{"facial-recognition":{"detection":{"modelName":"bench","options":{"minScore":0.7}},"recognition":{"modelName":"bench"}}}' yes ""

bench_task "ocr (detect + recognise)" \
    '{"ocr":{"detection":{"modelName":"bench","options":{"minScore":0.5}},"recognition":{"modelName":"bench","options":{"minScore":0.9}}}}' yes ""

bold "Done."; echo ""
echo "  Stage names map onto pipeline steps. To compare two runs, keep the image,"
echo "  the iteration count, the backend and the detector identical — face and OCR timings scale"
echo "  with how many faces and text regions the image contains."
echo ""
