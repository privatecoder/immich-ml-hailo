# Benchmarks

What this hardware actually does, and how to measure it yourself.

[MODELS.md](MODELS.md) answers *which model should I use*. This answers *what does this device do, and how do I reproduce these numbers*. The two overlap; neither duplicates the other's tables.

**The reproduction commands are the most valuable part of this file.** Every number below can be re-derived in minutes. If you change hardware, HailoRT version, image, or model, re-run rather than trusting what is written here.

---

## Conditions

Unless stated otherwise, everything was measured on:

| | |
|---|---|
| Device | one Hailo-8, PCIe Gen3 x2 |
| Runtime | HailoRT 4.24.0 |
| CLIP backend | `CLIP_BACKEND=siglip` |
| Image | `tests/test.jpg`, 2360×2360 |
| Sampling | p50 over 20 iterations |

One device, one runtime version, one image. Not a survey.

## Provenance

| Tag | Meaning |
|---|---|
| **Measured** | Read off the device or the worker's own timings. |
| **Derived** | Arithmetic on measured values (a fitted model, a ratio, a per-frame cost). |
| **Inferred** | An explanation consistent with the data but not directly observed. |

Every claim below carries one of these. Where a figure is a single data point rather than a curve, it says so.

---

## Device benchmarks — `hailortcli benchmark`

**Measured.** The accelerator alone, with no host pipeline around it.

| Model | HEF MiB | Latency | FPS | Hailo published | Ratio |
|---|---:|---:|---:|---:|---|
| `scrfd_2.5g` | 3.84 | 2.53 ms | 569 | 1058 | **1.86× off** |
| `scrfd_10g` | 6.87 | 4.40 ms | 441 | 440 | exact |
| `arcface_r50` | 29.7 | 20.36 ms | 35.2 | 113 | **3.2× off** |
| `arcface_mobilefacenet` | 3.91 | 1.09 ms | 5191 | 5191 | exact |
| `tinyclip_39m` image | 46.9 | 46.66 ms | 18.9 | — | — |
| `siglip_b_16` image | 181.1 | 205.70 ms | 4.46 | — | — |
| `siglip2_b_32_256` image | 187.6 | 205.64 ms | 4.39 | — | — |

Not benchmarked: both PaddleOCR models. Their device cost is unknown, so the host/device split for the OCR stages cannot be stated.

### Reproduce

The worker holds the VDevice exclusively, so it must be stopped first. `hailortcli` already ships in the base image — nothing to install.

```bash
docker stop immich-ml-hailo

docker run --rm --device=/dev/hailo0:/dev/hailo0 \
  -v "$PWD/models:/models:ro" \
  "hailo-base:v$HAILORT_VERSION" \
  hailortcli benchmark /models/scrfd_2.5g.hef

docker start immich-ml-hailo
```

---

## Published figures are unreliable

**Measured.** Three of the five checkable models are off by 1.86× to 3.2×; two match to the digit.

**Nothing distinguishes them in advance.** It is tempting to say small models match and large ones do not — the data refuses that: `scrfd_2.5g` (3.84 MiB) is 1.86× off while `arcface_mobilefacenet` (3.91 MiB) is exact, at essentially the same size. Nor does it split by task or by Model Zoo version.

**Anyone sizing work from the Model Zoo tables will sometimes be badly wrong, with no warning.** Benchmark the specific HEF you intend to run; it takes one command.

---

## In-pipeline stage timings — v1.0.0 baseline

**Measured.** The worker's own per-stage timers, which include host work the device benchmark excludes.

| Task | Stages (ms) | Total |
|---|---|---:|
| clip visual | `decode_image` 48.0 · `clip_preprocess` 0.2 · `clip_image_infer` 223.9 | **272.8** |
| clip text | `clip_text_infer` 217.9 | **218.7** |
| face | `decode_image` 47.7 · `letterbox` 0.9 · `det_infer` 8.8 · `det_decode` 0.8 · `crop_faces` 0.2 · `rec_infer_batch` 144.0 | **204.3** |
| ocr | `decode_image` 47.7 · `ocr_letterbox` 0.8 · `ocr_det_infer` 75.0 · `ocr_det_decode` 5.4 · `ocr_crop` 5.1 · `ocr_rec_batch` 159.9 | **294.7** |

Two things worth reading off this directly:

- **JPEG decode is ~48 ms on every image task**, pure host CPU. It is the single largest non-device cost and the reason request pipelining is worth anything.
- **Host share varies enormously by model size.** `det_infer` is 8.8 ms in-pipeline against 2.53 ms on the device — **71% host**. `clip_image_infer` is 223.9 ms against 205.70 ms — **8% host**. Small models are host-bound; large ones are device-bound. *(Derived from the two tables.)*

### Reproduce

```bash
./scripts/benchmark.sh                      # tests/test.jpg, 20 iterations
./scripts/benchmark.sh /path/to/img.jpg 50
```

Reads the worker's per-request summary lines out of `docker logs`, so it needs `LOG_LEVEL=INFO` (the default) and no rebuild. Pause Immich's ML jobs first — concurrent traffic contends for the device and lands in the same log; the script warns when the line count does not match what it fired.

---

## Effect of each change

**Measured.** Same conditions, one change at a time.

| Change | Stage effect | Face total |
|---|---|---:|
| v1.0.0 baseline | — | 204.3 |
| Device batching `B=8` | `rec_infer_batch` 144.0 → **35.2** | **96.8** |
| `FACE_DETECTOR=scrfd_10g` | `det_infer` 8.8 → **13.0** | **101.2** |
| `FACE_RECOGNIZER=arcface_mobilefacenet` | `rec_infer_batch` 35.2 → **8.0** | **69.2** |

Batching also took `ocr_rec_batch` 160.1 → **58.7** (ocr total 294.7 → **196.3**).

> **The last two rows are alternative configurations, not sequential steps.** Both are measured against the 96.8 ms batched baseline. 101.2 ms is `scrfd_10g` + `arcface_r50`; 69.2 ms is `scrfd_2.5g` + `mobilefacenet`. Applying both would be roughly **74 ms** — *derived, not measured*.

> **Small anomaly worth re-measuring.** `det_infer` rose 8.8 → 13.0 ms (+4.2) when the device cost rose only 2.53 → 4.40 ms (+1.87). The extra ~2.3 ms of host time is unexplained; the decode path is identical. Could be run-to-run noise, could be that a larger HEF costs more per activation host-side too. One repeat run would settle it.

---

## Device batch size

**Measured**, 5 faces, `rec_infer_batch` p50, `arcface_r50`.

| batch | p50 | per frame | speedup |
|---|---:|---:|---:|
| unset | 144.2 ms | 28.84 ms | — |
| 2 | 88.4 ms | 14.73 ms | 1.96× |
| 4 | 61.5 ms | 7.69 ms | 3.75× |
| 8 | 35.2 ms | 4.40 ms | **6.55×** |

**Derived** — fits to within 0.1% at B=4 and B=8, 3.2% at B=2:

> **cost = ceil(N / B) × (26.3 ms burst overhead + B × 1.11 ms per frame)**

Actual per-face compute is ~1.1 ms. The unbatched path paid the whole 26.3 ms burst overhead on *every frame*. That is why batching is worth 6.55×, and why padding a short batch up to the burst size costs almost nothing.

No batch sweep exists for OCR — only `B=8` against unset — so its burst overhead is unquantified and its optimum is unknown.

### Reproduce

Batch size is fixed at model-configure time, so each point needs a restart:

```bash
for B in default 2 4 8; do
  docker rm -f immich-ml-hailo
  docker run -d --device=/dev/hailo0:/dev/hailo0 --group-add=0 -p 3003:3003 \
    -e CLIP_BACKEND=siglip -e HAILO_BATCH_SIZE_FACE=$B \
    --name immich-ml-hailo "immich-ml-hailo:v$HAILORT_VERSION"
  sleep 30
  echo "=== B=$B ==="; ./scripts/benchmark.sh
done
```

---

## Concurrency

**Measured.** RPS by concurrency level, 20 requests per level.

| Configuration | C=1 | C=2 | C=4 | C=8 | gain |
|---|---:|---:|---:|---:|---|
| `serial`, face | 9.03 | 9.73 | 9.70 | 9.73 | 1.08× |
| `serial`, clip | 3.46 | — | — | 3.55 | 1.03× |
| `threadpool` + `arcface_r50`, face | 9.88 | — | — | 15.59 | 1.58× |
| `threadpool` + `mobilefacenet`, 4 threads | 12.67 | 20.86 | 26.53 | **28.37** | **2.24×** |
| `threadpool` + `mobilefacenet`, 8 threads | 12.77 | 22.05 | 28.25 | 29.28 | 2.29× |
| `threadpool`, clip | 3.53 | — | — | 4.12 | 1.17× |

The `serial` rows are the control: throughput flat across every level while client latency rose linearly, which is queueing rather than work.

**Against v1.0.0's 205.7 ms face request (4.86 RPS), 28.37 RPS is 5.8×** on that path. *(Derived.)*

> That 5.8× is **not available at unchanged accuracy.** It includes `mobilefacenet`, which trades 99.7% → 99.4% LFW and forces Immich to re-run its face jobs. Holding accuracy constant — batching plus threadpool on `arcface_r50` — gives 15.59 RPS, **3.2×**. Quote whichever is honest for the configuration you are actually running.

**`REQUEST_THREADS=4` is the optimum**: 8 threads gives 29.28 against 28.37, **+3% for double the peak memory**. The wall is host-side serialisation, not thread count.

The clip curve has only two points (C=1 and C=8). Its saturation level is **not measured**.

### Reproduce

```bash
./scripts/benchmark_concurrency.sh face                  # 20 requests, C=1,2,4,8
./scripts/benchmark_concurrency.sh clip tests/test.jpg 40 "1 2 4 8"
```

Phase 1 is a control: the same full multipart upload with deliberately invalid `entries`, so the request is rejected before any inference. If the control does not scale with C, the load generator is the bottleneck rather than the worker and the script says so and voids the run. Without that check a flat result is ambiguous.

---

## Two derived models

### 1. Large models are bound by weight streaming, not compute

**Inferred**, from a correlation plus one decisive pair.

Device latency for models above ~30 MiB tracks HEF size at roughly 1 GB/s. The evidence that makes it more than a correlation: **SigLIP B/16 processes 196 patches and SigLIP2 B/32-256 processes 64 — 3× the computation — and they take identical time** (205.70 vs 205.64 ms) at file sizes within 4% of each other. Compute varies threefold and changes nothing; size predicts everything.

Hailo-8 is PCIe Gen3 x2, theoretical ~1.97 GB/s. An effective ~1 GB/s is about half of that, ordinary for real DMA. A 181 MiB model cannot stay resident, so its weights cross the bus on every inference.

Not directly observed — no bus counters were read. See MODELS.md for the full per-model rate table.

### 2. The batch cost model

**Derived** (fitted above). Beyond predicting batch behaviour it explains *why* batching works: a burst pays the weight load once instead of once per frame. That also connects the two models — the 26.3 ms burst overhead is the same order as `arcface_r50`'s 20.36 ms standalone latency, which is 29.7 MiB of weights moving.

---

## Test tooling — what each catches

Four tools, four different failure modes. None subsumes another.

| Tool | Detects | Blind to |
|---|---|---|
| `tests/test.sh` | Response **shapes** — endpoints, keys, types, embedding dimensions, error handling. 19 assertions (18 without OCR) | Values. A correctly-shaped, wrong embedding passes every assertion |
| `tests/golden.sh` | Embedding **values**, bit-exactly. The device is deterministic, so the threshold sits at the 1e-5 floor and any change at all fails | Anything not an embedding. Also only makes serial requests |
| `tests/verify_ocr_align.py` | **Text-to-box correspondence** — that `text[i]` names the cell `box[i]` sits in | Timing, embeddings, response shape |
| `scripts/benchmark*.sh` | **Timing** and throughput | Correctness of any kind |

### The alignment grid is the only tool for its failure mode

`tests/ocr-align.jpg` is a 64-cell grid in which **every token encodes its own position** — `R3C5` means row 3, column 5. So for each returned region the checker computes which cell the *box* centre falls in, and asserts the *text* at that index names that cell.

That makes it the only thing here that detects **text attached to the wrong box** — the failure mode of OCR chunked batching, of batch padding, and of concurrency. It reports `SHIFTED=n` with the exact offset when indices slip, and distinguishes a genuine shift from ordinary OCR misreads. It has passed every time it has been run, which is precisely why it must not be lost.

64 regions against a chunk size of 32 means chunk boundaries are always crossed — a shift confined to one chunk would still be caught.

`tests/ocr-align.svg` is the source it was rendered from, so the grid can be regenerated or altered.

### Reproduce

```bash
curl -s -X POST http://localhost:3003/predict \
  -F 'entries={"ocr":{"detection":{"modelName":"x","options":{"minScore":0.3}},"recognition":{"modelName":"x","options":{"minScore":0.5}}}}' \
  -F 'image=@tests/ocr-align.jpg' -o /tmp/ocr.json

python3 tests/verify_ocr_align.py /tmp/ocr.json
```

Pure stdlib, so it runs anywhere with python3. On a host without one:

```bash
docker cp tests/verify_ocr_align.py immich-ml-hailo:/tmp/
docker cp /tmp/ocr.json immich-ml-hailo:/tmp/
docker exec immich-ml-hailo python3 /tmp/verify_ocr_align.py /tmp/ocr.json
```

Expected: `VERDICT: ALIGNED — every region's text names its own cell, across 64 regions`.

### The check that matters most: alignment under load

Run the alignment check **while** a concurrency sweep is in flight. That is the only configuration that exercises concurrency against a correctness test:

```bash
./scripts/benchmark_concurrency.sh face tests/test.jpg 60 "8" &
sleep 5
# ... the curl + verify above, two or three times ...
wait
```

This has been done: three separate alignment checks returned `exact=64 SHIFTED=0` against a genuinely saturated worker (server p50 1253 ms, client p50 2648 ms), alongside a bit-exact golden check and 19/19.

### What none of them catch

**A rare race under sustained real-world load.** Every correctness tool here makes serial requests; the one concurrent run above was 60 requests, not a library scan. `golden.sh` would catch corruption deterministically but only in requests it issues itself. Nothing here would catch a fault that needs hours of mixed traffic to surface.

That is the honest reason `REQUEST_MODE=threadpool` is opt-in.

Also uncovered: p99 latency (the benchmarks report p50/p95), steady-state memory under sustained concurrency, and any path exercised only by unusual images — the golden and alignment checks both use fixed inputs.

---

## Measurements we should still take

Gaps noticed while assembling this:

1. **Device benchmarks for both PaddleOCR models.** Without them the host/device split for `ocr_det_infer` (75.0 ms) and `ocr_rec_batch` cannot be stated, and OCR is the second most expensive task.
2. **An OCR batch-size sweep.** Only `B=8` versus unset exists; the optimum is assumed by analogy with faces, not measured.
3. **Intermediate clip concurrency points** (C=2, C=4). The saturation level is currently inferred from its device-bound ratio.
4. **An OCR concurrency sweep.** Neither `serial` nor `threadpool` has one.
5. **The real faces-per-photo distribution.** MODELS.md's weighted batch table assumes one; `faces=` in the request log already records the truth, so a histogram over one library scan would replace an assumption with a fact.
6. **Steady-state RSS during a concurrent sweep**, to confirm the 4-thread memory bound in practice.
7. **A repeat of `det_infer` under `scrfd_10g`**, to settle the +2.3 ms of unexplained host time noted above.
8. **Host CPU core count and utilisation during a threadpool sweep.** The worker reaches roughly half the device ceiling and the cause is not established — see below.

### On the ~50% ceiling

It has been suggested that the GIL explains why throughput plateaus at roughly half the device's theoretical rate. **That explanation is weaker than it looks and should not be written down as fact.** The dominant host cost is JPEG decode via Pillow, and both Pillow's decoders and NumPy's array operations *release* the GIL — they are among the libraries most likely to parallelise cleanly.

At least as plausible: the host simply does not have enough cores to run four concurrent decodes, or memory bandwidth saturates. Both are cheap to test (`nproc`, and `top` during a sweep) and neither has been checked. Until one is, the ceiling's cause is **unexplained**, not "the GIL".
