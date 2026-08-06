#!/usr/bin/env bash
#
# Concurrency benchmark: does firing requests in parallel improve throughput?
#
# The question is open because `app.py` declares `async def predict` with a
# fully synchronous body, so every request serialises on the event loop — no
# overlap even between one request's CPU work and another's device wait. There
# is only one Hailo device, so device time can never overlap device time; the
# only available win is hiding CPU work (JPEG decode ~49 ms) behind device work.
# This measures whether that win exists before anything is changed to chase it.
#
# READ-ONLY. Sends /predict requests and reads `docker logs` / `docker inspect`.
# Never restarts, rebuilds or reconfigures anything.
#
# Usage:
#   ./scripts/benchmark_concurrency.sh                        # clip, 20 reqs, C=1,2,4,8
#   ./scripts/benchmark_concurrency.sh face                   # facial-recognition
#   ./scripts/benchmark_concurrency.sh ocr /path/img.jpg 40   # custom image and count
#   ./scripts/benchmark_concurrency.sh clip tests/test.jpg 40 "1 2 4 8 16"
#
#   task    clip | face | ocr    (default clip)
#   image   path to a JPEG       (default tests/test.jpg)
#   count   requests per level   (default 20)
#   levels  concurrency list     (default "1 2 4 8")
#
# Environment:
#   BASE_URL    default http://localhost:3003
#   CONTAINER   container name/id; auto-detected from the published port if unset
#
# Run this on the Docker host.
#
set -euo pipefail

# Comma-decimal locales make `sort -n` and awk misparse "71.2" as 71, silently
# corrupting every percentile. Same trap as scripts/benchmark.sh.
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TASK="${1:-clip}"
IMAGE="${2:-$PROJECT_DIR/tests/test.jpg}"
COUNT="${3:-20}"
LEVELS="${4:-1 2 4 8}"
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
bold "=== immich-ml-hailo concurrency benchmark ==="; echo ""

for c in docker curl xargs awk sort; do
    command -v "$c" >/dev/null 2>&1 || fail "$c not found"
done
[[ -f "$IMAGE" ]] || fail "test image not found: $IMAGE"

case "$COUNT" in ''|*[!0-9]*) fail "count must be a positive integer, got '$COUNT'" ;; esac
[[ "$COUNT" -ge 1 ]] || fail "count must be at least 1"
for c in $LEVELS; do
    case "$c" in ''|*[!0-9]*) fail "concurrency levels must be positive integers, got '$c'" ;; esac
    [[ "$c" -ge 1 ]] || fail "concurrency levels must be at least 1"
done

LEVEL_COUNT=$(printf '%s\n' $LEVELS | grep -c .)
BASE_LEVEL=$(printf '%s\n' $LEVELS | head -1)

case "$TASK" in
    clip) ENTRIES='{"clip":{"visual":{"modelName":"bench"}}}'
          TASK_KEY="clip"; TASK_EXTRA="clip=visual"; TASK_LABEL="clip visual (device-dominated)" ;;
    face) ENTRIES='{"facial-recognition":{"detection":{"modelName":"bench","options":{"minScore":0.7}},"recognition":{"modelName":"bench"}}}'
          TASK_KEY="facial-recognition"; TASK_EXTRA=""; TASK_LABEL="facial-recognition (CPU-dominated)" ;;
    ocr)  ENTRIES='{"ocr":{"detection":{"modelName":"bench","options":{"minScore":0.5}},"recognition":{"modelName":"bench","options":{"minScore":0.9}}}}'
          TASK_KEY="ocr"; TASK_EXTRA=""; TASK_LABEL="ocr (mixed)" ;;
    *)    fail "unknown task '$TASK' — expected clip, face or ocr" ;;
esac

curl -sf "$BASE_URL/ping" >/dev/null 2>&1 \
    || fail "service not reachable at $BASE_URL/ping — start the container first"

CONTAINER="${CONTAINER:-$(docker ps -q --filter "publish=3003" 2>/dev/null | head -1)}"
[[ -n "$CONTAINER" ]] || fail "no running container publishing port 3003 — set CONTAINER=<name>"

_env_of() {
    docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" 2>/dev/null \
        | awk -F= -v k="^$1=" '$0 ~ k {print $2}' | head -1
}

IMAGE_TAG=$(docker inspect -f '{{.Config.Image}}' "$CONTAINER" 2>/dev/null || echo unknown)
CONTAINER_NAME=$(docker inspect -f '{{.Name}}' "$CONTAINER" 2>/dev/null | sed 's|^/||' || echo "$CONTAINER")
BACKEND="$(_env_of CLIP_BACKEND)";     BACKEND="${BACKEND:-tinyclip (default)}"
DETECTOR="$(_env_of FACE_DETECTOR)";   DETECTOR="${DETECTOR:-scrfd_2.5g (default)}"
RECOGNIZER="$(_env_of FACE_RECOGNIZER)"; RECOGNIZER="${RECOGNIZER:-arcface_r50 (default)}"
LOGLEVEL="$(_env_of LOG_LEVEL)";       LOGLEVEL="${LOGLEVEL:-INFO}"

case "$(printf '%s' "$LOGLEVEL" | tr '[:lower:]' '[:upper:]')" in
    DEBUG|INFO|NOTSET) ;;
    *) fail "container LOG_LEVEL=$LOGLEVEL suppresses the per-request summary line this script parses. Restart it with LOG_LEVEL=INFO." ;;
esac

# ── Timing helper ─────────────────────────────────────────────────────
#
# Millisecond wall clock, in descending order of preference:
#   1. GNU `date +%s%3N`      — the Linux/Unraid case, no subprocess concerns
#   2. bash 5 $EPOCHREALTIME  — builtin, no fork
#   3. perl                   — present on most systems including macOS
#   4. whole seconds          — last resort, loudly flagged
#
# Resolution matters more than it looks: a C=8 run of 20 short requests can
# finish inside one second, and second-granularity would report 0 ms wall, which
# turns every derived figure into a divide-by-zero.
CLOCK=""
if [[ "$(date +%s%3N 2>/dev/null)" =~ ^[0-9]+$ ]]; then
    CLOCK="date"
    now_ms() { date +%s%3N; }
elif [[ -n "${EPOCHREALTIME:-}" ]]; then
    CLOCK="bash"
    now_ms() { local t="${EPOCHREALTIME/,/.}"; echo $(( ${t%.*} * 1000 + 10#${t#*.} / 1000 )); }
elif command -v perl >/dev/null 2>&1; then
    CLOCK="perl"
    now_ms() { perl -MTime::HiRes=time -e 'printf "%.0f", time()*1000'; }
else
    CLOCK="seconds"
    echo "  $(yellow WARNING): no sub-second clock available — using whole seconds."
    echo "        Raise the request count so each level runs for several seconds,"
    echo "        or throughput figures will be badly quantised."
    echo ""
    now_ms() { echo $(( $(date +%s) * 1000 )); }
fi

# A zero-length run means the clock is too coarse for this workload, not that
# the work was instant. Say which, rather than emitting RPS 0.00 and letting the
# generator check blame itself.
guard_wall() {
    if [[ "$1" -le 0 ]]; then
        echo "    $(red ERROR): measured 0 ms wall clock (clock=$CLOCK)." >&2
        echo "    The run finished below this clock's resolution. Increase the" >&2
        echo "    request count, or run where a sub-second clock is available." >&2
        exit 1
    fi
}

echo "  Target:      $BASE_URL"
echo "  Container:   $CONTAINER_NAME ($IMAGE_TAG)"
echo "  Backend:     $BACKEND"
echo "  Detector:    $DETECTOR"
echo "  Recognizer:  $RECOGNIZER"
echo "  Image:       $IMAGE"
echo "  Task:        $TASK_LABEL"
echo "  Requests:    $COUNT per concurrency level"
echo "  Levels:      $LEVELS"
echo "  Clock:       $CLOCK"
echo ""
echo "  $(yellow NOTE): pause Immich's ML jobs. Concurrent traffic contends for the"
echo "        device and lands in the same log."
echo ""

# ── Load generator ────────────────────────────────────────────────────
#
# xargs -P holds exactly C curl processes in flight. Each writes its own
# time_total to its own file, so nothing interleaves and no lock is needed.
#
# Process spawn cost is the obvious worry with this approach — it is why the
# control phase below exists rather than being assumed away.
#
# run_load CONCURRENCY COUNT ENTRIES OUTDIR -> echoes "wall_ms ok_count"
run_load() {
    local conc="$1" n="$2" entries="$3" outdir="$4"
    mkdir -p "$outdir"
    local t0 t1
    t0=$(now_ms)
    seq 1 "$n" | xargs -P "$conc" -I{} sh -c '
        curl -s -o /dev/null \
             -w "%{time_total} %{http_code}\n" \
             -X POST "$1/predict" \
             -F "entries=$2" \
             -F "image=@$3" > "$4/r{}" 2>/dev/null || echo "0 000" > "$4/r{}"
    ' _ "$BASE_URL" "$entries" "$IMAGE" "$outdir"
    t1=$(now_ms)
    local ok
    ok=$(cat "$outdir"/r* 2>/dev/null | awk '$2 ~ /^[0-9]{3}$/ {c++} END {print c+0}')
    echo "$(( t1 - t0 )) $ok"
}

# Nearest-rank percentile over a column of numbers on stdin.
pct() { sort -n | awk -v p="$1" '{v[NR]=$1} END {if (NR) printf "%.1f", v[int((NR*p+99)/100)]; else printf "0.0"}'; }

client_ms() { cat "$1"/r* 2>/dev/null | awk '{printf "%.1f\n", $1*1000}'; }

# Server-side handler time from the worker's own summary lines, so queue wait
# (client) can be separated from work (server).
# `since` is an absolute RFC3339 timestamp so each level reads only its own
# lines. A relative duration always looks back from now and re-counts earlier
# levels, which is the bug this replaced.
server_ms() {
    local since="$1" status_filter="$2" extra="$3"
    docker logs --since "$since" "$CONTAINER" 2>&1 \
        | grep -F '/predict ' | grep -F "$status_filter" \
        | { [[ -n "$extra" ]] && grep -F "$extra" || cat; } \
        | grep -o 'total=[0-9.]*ms' | sed 's/total=//; s/ms//'
}

# ── Phase 1: control — is the generator itself the bottleneck? ────────
#
# Fires the same multipart upload of the same image, but with deliberately
# invalid `entries`. FastAPI still receives the whole body; the handler rejects
# it at JSON parsing and returns 400 before any inference. So this measures
# everything except the worker's actual work: process spawn, TLS-less HTTP,
# ~780 KB upload, framework overhead.
#
# The worker logs these too, with status=400 error=entries-parse, which is what
# keeps them distinguishable from the real measurements below.
#
# The container healthcheck cannot contaminate anything: GET /ping emits no
# summary line at all, so it is structurally uncountable in the log parsing.

bold "  Phase 1 — control (full upload, 400 before inference)"; echo ""
printf "    %-6s %10s %10s %12s\n" "C" "wall" "RPS" "vs C=$BASE_LEVEL"
printf "    %-6s %10s %10s %12s\n" "------" "----------" "----------" "------------"

CTRL_BASE=""
CTRL_MAX=""
for c in $LEVELS; do
    read -r wall ok <<<"$(run_load "$c" "$COUNT" 'not-valid-json' "$WORK/ctrl.$c")"
    guard_wall "$wall"
    [[ "$ok" -eq "$COUNT" ]] || echo "    $(yellow WARN): only $ok/$COUNT control requests completed at C=$c"
    rps=$(awk -v n="$COUNT" -v w="$wall" 'BEGIN{printf "%.2f", (w>0)? n*1000/w : 0}')
    [[ -z "$CTRL_BASE" ]] && CTRL_BASE="$rps"
    CTRL_MAX="$rps"
    spd=$(awk -v a="$rps" -v b="$CTRL_BASE" 'BEGIN{printf "%.2fx", (b>0)? a/b : 0}')
    printf "    %-6s %9sms %10s %12s\n" "$c" "$wall" "$rps" "$spd"
done
echo ""

CTRL_SCALING=$(awk -v a="$CTRL_MAX" -v b="$CTRL_BASE" 'BEGIN{printf "%.2f", (b>0)? a/b : 0}')
GEN_OK=1
if [[ "$LEVEL_COUNT" -lt 2 ]]; then
    # A one-level sweep compares the control against itself and always
    # yields 1.00x, which the check below would read as a stalled generator
    # and use to void a perfectly good run. Scaling is only meaningful
    # across two or more levels.
    echo "    $(yellow NOTE): only one concurrency level, so there is nothing to"
    echo "    compare the control against. The scaling check is skipped — it needs"
    echo "    at least two levels. The control itself ran fine (row above); it"
    echo "    simply cannot prove the generator outpaces the worker from one point."
    echo "    Add a second level to validate it, e.g. levels \"1 $BASE_LEVEL\"."
elif awk -v s="$CTRL_SCALING" 'BEGIN{exit !(s < 1.5)}'; then
    GEN_OK=0
    echo "    $(red 'GENERATOR WARNING'): control throughput scaled only ${CTRL_SCALING}x across the sweep."
    echo "    The load generator may not be issuing requests in parallel — on some"
    echo "    xargs implementations -I disables -P. Everything below is then"
    echo "    measuring the generator, not the worker. Treat the results as void"
    echo "    until this is explained."
else
    echo "    $(green OK): control scales ${CTRL_SCALING}x — the generator can outpace the worker,"
    echo "    so a flat result below is a property of the worker and not of this script."
fi
echo ""

# ── Phase 2: the real task ────────────────────────────────────────────

bold "  Phase 2 — $TASK_LABEL"; echo ""
printf "    %-4s %9s %8s %9s %12s %12s %12s\n" \
    "C" "wall" "RPS" "vs C=$BASE_LEVEL" "client p50" "client p95" "server p50"
printf "    %-4s %9s %8s %9s %12s %12s %12s\n" \
    "----" "---------" "--------" "---------" "------------" "------------" "------------"

BASE_RPS=""
for c in $LEVELS; do
    out="$WORK/task.$c"

    # Absolute start timestamp, not a duration. `docker logs --since <N>s` is
    # measured from *now*, so a window wide enough to cover this level also
    # covered the previous one — counts came out cumulative (20, 40, 60, ...),
    # contaminating server p50 and making the contamination warning cry wolf.
    # Sleep first so the previous level's lines are more than one second old,
    # since these timestamps have second resolution.
    sleep 1
    level_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    read -r wall ok <<<"$(run_load "$c" "$COUNT" "$ENTRIES" "$out")"
    guard_wall "$wall"
    [[ "$ok" -eq "$COUNT" ]] || echo "    $(yellow WARN): only $ok/$COUNT requests completed at C=$c"

    server_ms "$level_start" "status=200" "$TASK_EXTRA" | grep -c . > "$WORK/seen.$c" || true
    seen=$(cat "$WORK/seen.$c")
    if [[ "$seen" -ne "$COUNT" ]]; then
        echo "    $(yellow WARN): fired $COUNT requests but matched $seen log lines at C=$c —"
        echo "          other traffic is hitting this worker; treat these numbers as contaminated."
    fi

    rps=$(awk -v n="$COUNT" -v w="$wall" 'BEGIN{printf "%.2f", (w>0)? n*1000/w : 0}')
    [[ -z "$BASE_RPS" ]] && BASE_RPS="$rps"
    spd=$(awk -v a="$rps" -v b="$BASE_RPS" 'BEGIN{printf "%.2fx", (b>0)? a/b : 0}')

    cp50=$(client_ms "$out" | pct 50)
    cp95=$(client_ms "$out" | pct 95)
    sp50=$(server_ms "$level_start" "status=200" "$TASK_EXTRA" | pct 50)

    printf "    %-4s %8sms %8s %9s %11sms %11sms %11sms\n" \
        "$c" "$wall" "$rps" "$spd" "$cp50" "$cp95" "$sp50"
done

# ── How to read it ────────────────────────────────────────────────────

echo ""
bold "  Reading this"; echo ""
cat <<'EOF'
    RPS flat, client p50/p95 rising roughly linearly with C, server p50 flat
      -> requests are QUEUEING, not slowing down. The worker serialises them.
         This is what `async def predict` with a synchronous body produces, and
         it is the case where moving to a threadpool handler could help.

    RPS rising with C
      -> the worker already overlaps work. No concurrency change is needed, and
         the premise behind doing one is wrong.

    RPS flat AND server p50 rising with C
      -> contention rather than queueing — the device or the host is saturated.
         A threadpool would not help; it would only move the queue.

    Client p50 minus server p50 is the queue wait. At C=1 they should nearly
    match; a widening gap as C rises is the clearest evidence of serialisation.
EOF
echo ""
[[ "$GEN_OK" -eq 1 ]] || echo "  $(red 'Results void: see the generator warning above.')"
echo ""
