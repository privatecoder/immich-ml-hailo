# Model Selection

Why this project runs the models it does, which alternatives were evaluated, and what is worth changing.

Every model must exist as a compiled `.hef` for Hailo-8 in the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) — this project does not train or compile models. Within that constraint the choices below are deliberate, and the alternatives were evaluated against four criteria:

1. **Immich embedding compatibility** — whether a change forces users to re-index their library.
2. **Accuracy** on the task.
3. **Measured cost on this Hailo-8** — not published figures, and not GPU numbers. All models share one 26 TOPS device.
4. **Whether it is the bottleneck.** Making a fast stage faster buys nothing.

> **Raw numbers and reproduction commands live in [BENCHMARKS.md](BENCHMARKS.md).** This file explains the model choices; that one records what the hardware does and how to measure it again.

> **Read this first: the Model Zoo's published FPS figures do not predict what you will get.** Of the five models here with a published figure, three are off by 1.9× to 3.2× and two match exactly — with nothing to tell you in advance which kind you are looking at. Every performance number in this document was measured on the production device. See [Measured device performance](#measured-device-performance).

---

## What ships today

| Role | Model | Source | Device latency | Output |
|---|---|---|---|---|
| Face detection | `scrfd_2.5g` *(default)* or `scrfd_10g` | Model Zoo v2.17.0 | 2.53 / 4.40 ms | 9 output streams, 3 strides |
| Face recognition | `arcface_r50` *(default)* or `arcface_mobilefacenet` | v2.17.0 | 20.36 / 1.09 ms | 512-dim, L2-normalized |
| Smart search (default) | `tinyclip_vit_39m_16_text_19m_yfcc15m` | v2.17.0 | 46.66 ms | 512-dim |
| Smart search (quality) | `siglip_b_16` | v2.18.0 | 205.70 ms | 768-dim |
| OCR detection | `paddle_ocr_v5_mobile_detection` | v2.18.0 | not benchmarked | 544×960 probability map |
| OCR recognition | `paddle_ocr_v5_mobile_recognition` | v2.18.0 | not benchmarked | CTC over 18,383 characters |

Latencies are `hailortcli benchmark` on the production Hailo-8, HailoRT 4.24.0.

---

## Measured device performance

All figures below are `hailortcli benchmark` on one production Hailo-8 running HailoRT 4.24.0. Single device, single runtime version — not a survey.

| Model | HEF | MiB | Latency | FPS | Published FPS | Ratio |
|---|---:|---:|---:|---:|---:|---|
| `scrfd_2.5g` *(current)* | 4,027,374 | 3.8 | 2.53 ms | 569 | 1058 | **1.86× off** |
| `scrfd_10g` | 7,206,072 | 6.9 | 4.40 ms | 441 | 440 | exact |
| `arcface_r50` *(current)* | 31,190,468 | 29.7 | 20.36 ms | 35.2 | 113 | **3.21× off** |
| `arcface_mobilefacenet` | 4,099,978 | 3.9 | 1.09 ms | 5191 | 5191 | exact |
| `tinyclip_39m` image | 49,127,796 | 46.9 | 46.66 ms | 18.9 | — | — |
| `siglip_b_16` image | 189,897,833 | 181.1 | 205.70 ms | 4.46 | — | — |
| `siglip2_b_32_256` image | 196,662,515 | 187.6 | 205.64 ms | 4.39 | — | — |

### Published figures are unreliable here

Three of the five checkable models are off by 1.86× to 3.21×; two match to the digit.

**There is no rule for telling them apart in advance.** It is tempting to say small models match and large ones do not — the data refuses that: `scrfd_2.5g` (3.8 MiB) is 1.86× off while `arcface_mobilefacenet` (3.9 MiB) is exact, at essentially the same size. Nor does it split by task, or by Model Zoo version.

The practical consequence: **anyone sizing work from the Model Zoo tables will sometimes be badly wrong, with no warning.** Benchmark the specific HEF you intend to run. It takes one command — see [Evaluating a candidate on this hardware](#evaluating-a-candidate-on-this-hardware).

### Large models are bound by weight streaming, not compute

*(This is an inference from the measurements below, not a directly observed mechanism. It is a strong inference, and it is labelled as one.)*

Divide HEF size by latency and an effective transfer rate falls out:

| Model | MiB | Latency | Effective rate |
|---|---:|---:|---:|
| `arcface_mobilefacenet` | 3.9 | 1.09 ms | 3.76 GB/s |
| `scrfd_10g` | 6.9 | 4.40 ms | 1.64 GB/s |
| `scrfd_2.5g` | 3.8 | 2.53 ms | 1.59 GB/s |
| `arcface_r50` | 29.7 | 20.36 ms | 1.53 GB/s |
| `tinyclip_39m` | 46.9 | 46.66 ms | 1.05 GB/s |
| `siglip2_b_32_256` | 187.6 | 205.64 ms | 0.96 GB/s |
| `siglip_b_16` | 181.1 | 205.70 ms | 0.92 GB/s |

The three largest converge tightly on ~0.9–1.05 GB/s. Small models sit well above that line, as you would expect of models whose cost is dominated by something other than moving weights. `arcface_r50` at 1.53 GB/s sits between the two regimes.

**The decisive evidence is the SigLIP pair.** SigLIP B/16 at 224px processes 196 patches; SigLIP2 B/32 at 256px processes 64 — **3.06× the compute** for B/16. Their latencies are **205.70 ms and 205.64 ms**: identical to within 0.03%, at file sizes within 3.6% of each other. Compute varies threefold and changes nothing; size predicts everything.

Hailo-8 is a PCIe Gen3 x2 device, theoretical ~1.97 GB/s. An effective ~1 GB/s is about half of that, which is an ordinary result for real DMA. A 181 MiB model cannot stay resident on the accelerator, so its weights cross the bus on every inference.

**This retroactively explains Tier 2's batching result.** The 26.3 ms per-burst overhead fitted to the ArcFace batch sweep is the same order as its 20.36 ms standalone latency — it is ~29.7 MiB of weights being loaded. That is precisely why batching bought 6.5×: one weight load amortised across eight frames rather than paid eight times.

### Device cost versus pipeline cost

In-pipeline stage timings (`scripts/benchmark.sh`, p50, 2360×2360, siglip, 20 iterations, shipped defaults) against the device latency for the same model:

| Stage | In pipeline | Device | Host-side | Host share |
|---|---:|---:|---:|---:|
| `det_infer` | 8.8 ms | 2.53 ms | 6.3 ms | **71%** |
| `clip_image_infer` | 224.2 ms | 205.70 ms | 18.5 ms | **8%** |
| `rec_infer_batch` (5 faces, batch 8) | 35.2 ms | — | — | — |
| `clip_text_infer` | 218.3 ms | — | — | — |
| `ocr_det_infer` | 74.9 ms | not benchmarked | — | — |
| `ocr_rec_batch` (12 regions, batch 8) | 58.7 ms | — | — | — |
| `decode_image` | ~49 ms | — | 100% | pure CPU JPEG decode |

Request totals: **clip visual 274.1 ms · facial-recognition 96.8 ms · ocr 196.3 ms.**

The two ends of that table are the useful part. **Small models are host-bound** — detection spends 71% of its stage time outside the device, so optimising the host path is what would help it. **Large models are device-bound** — CLIP spends 92% on the device, so no amount of host-side work will move it.

### CLIP has no speed lever except a smaller model

Worth stating plainly, because it is the question a reader will keep returning to.

`clip_image_infer` is 224.2 ms, of which **205.7 ms is the device**. Therefore:

- **Not plumbing.** Only 18.5 ms is host-side. Eliminating *all* of it — virtual-stream reuse, activation caching, everything — buys at most 8%.
- **Not batching.** Immich sends one asset per request, so the encoder receives exactly one frame per call. There is no batch to form, and configuring a device batch size would only pad one real frame up to eight and make it worse. This is why CLIP is left at HailoRT's default.
- **Not a different SigLIP variant.** SigLIP2 B/32-256 is the same 205 ms, measured. Any encoder of this size costs this much here.

205 ms is what a ~180 MiB model costs on a PCIe Gen3 x2 accelerator. The only lever is a **smaller model** — and the smaller model that exists, TinyCLIP at 46.9 MiB and 46.66 ms, is 4.4× faster and Immich-incompatible. That trade is the CLIP backend decision, and it is not improvable by engineering.

---

## The CLIP backend decision

This is the only choice with consequences beyond this worker, because CLIP embeddings are written into Immich's search index.

**`siglip` — 768-dim, the quality option.** Embeddings are interchangeable with Immich's own `ViT-B-16-SigLIP__webli`. A user can switch between this worker and Immich's built-in ML worker without re-running Smart Search. That interoperability is the reason SigLIP exists here. Measured cost: **205.70 ms device, 224.2 ms in pipeline.**

**`tinyclip` — 512-dim, the throughput option.** Measured **46.66 ms device (18.9 FPS)** against SigLIP's 205.70 ms — **4.4× faster**, and a simpler code path (it requests `FLOAT32` output, so no dequantization). It matches **no** Immich CLIP model, so a user on TinyCLIP is in a private embedding space: moving to or from Immich's own worker, or to SigLIP, means re-processing the entire library.

Note that the vendor's published CLIP figures (~60 and ~14 FPS) are both roughly 3× optimistic against measurement (18.9 and 4.46). Their *ratio* survives — 4.3× published, 4.2× measured — so the published numbers were useful for choosing between the two models and useless for predicting how long a scan takes.

TinyCLIP remains the default for historical reasons and because it is the fastest path for an initial index on a large library. **SigLIP is the better recommendation for most users**, and the README's `docker run` examples lead with it.

Changing CLIP backend on an existing library always requires re-running Smart Search — the dimensions differ (512 vs 768), and even at equal dimensions the vector spaces are unrelated.

---

## Face recognition: the batching cost model

Measured on the production Hailo-8. `rec_infer_batch` for 5 faces, varying the configured device batch size:

| batch | frames sent | p50 | per frame | speedup |
|---|---|---|---|---|
| unset | 5 | 144.2 ms | 28.84 ms | — |
| 2 | 6 | 88.4 ms | 14.73 ms | 1.96× |
| 4 | 8 | 61.5 ms | 7.69 ms | 3.75× |
| 8 | 8 | 35.2 ms | 4.40 ms | **6.55×** |

These fit a two-term model to within 0.1% at batch 4 and 8, and 3.2% at batch 2:

> **cost = ceil(N / B) × (26.3 ms burst overhead + B × 1.11 ms per frame)**

Actual per-face compute is **~1.1 ms**; the unbatched path was paying the entire 26.3 ms burst overhead on *every single frame*. As established above, that overhead is weight loading — ~29.7 MiB across the PCIe bus — which is why amortising it across a burst is worth 6.5×, and why padding a short batch up to the burst size costs almost nothing.

Weighted over an **assumed** distribution of faces per photo (1: 40%, 2: 25%, 3: 15%, 5: 12%, 8: 6%, 20: 2%) — the per-N costs are measured, the weighting is a guess, and these totals inherit that:

| batch | weighted mean | vs unbatched |
|---|---|---|
| unset | 81.6 ms | — |
| 2 | 49.9 ms | 1.64× |
| 4 | 38.7 ms | 2.11× |
| **8** | **36.6 ms** | **2.23×** |
| 16 | 44.9 ms | 1.82× |

**8 is the optimum.** 16 is worse only because single-face photos pad wastefully into a 16-frame burst. Batch 8 beats unbatched at every N ≥ 2; at N = 1 it costs 28.8 → 35.2 ms. At the `max_faces` cap of 100, batching gives 13 bursts totalling ~457 ms against ~2884 ms — 6.3×.

**OCR recognition** shows the same mechanism: `ocr_rec_batch` for 12 regions goes **160.1 ms → 58.7 ms** at batch 8. Backing out the 1.33× padding tax gives a per-frame speedup of ~3.6×. One data point, so its burst overhead is not separately quantified and its optimal batch size is unknown.

---

## Evaluated alternatives

### Shipped as selectable options

#### `arcface_mobilefacenet` — **shipped**, selectable with `FACE_RECOGNIZER=arcface_mobilefacenet`

Both recognition models ship and `setup.sh` downloads both, so switching is a container restart. **`arcface_r50` remains the default**, and the reason is not speed.

| | LFW accuracy | Device latency | Device FPS |
|---|---|---|---|
| `arcface_r50` *(default)* | **99.7%** | 20.36 ms | 35.2 |
| `arcface_mobilefacenet` | 99.4% | **1.09 ms** | **5191** |

**The accuracy trade is the headline.** 0.3 pp on LFW reads as noise and is not, because this is face *identity* and the failure mode is user-visible: two people merged into one cluster, or one person split across two, each needing manual correction in Immich. That is more annoying than a slow initial scan, and it is why this is an option rather than a new default.

**The speed is real but secondary.** 18.7× lower latency, measured. An earlier revision of this document rated this "low to medium value-for-effort" on the reasoning that recognition was not the bottleneck and that a smaller model would only reduce the ~1.1 ms of per-face compute while leaving the 26.3 ms burst overhead. **That reasoning was wrong** — the burst overhead is weight loading, and mobilefacenet is 3.9 MiB against R50's 29.7 MiB, so a smaller model removes almost all of it rather than almost none. `rec_infer_batch` should fall from 35.2 ms for 5 faces to a few milliseconds.

**The default-vs-option reasoning is unchanged, but the reason has shifted.** It was previously "not worth the effort because recognition is cheap". It is now "worth real speed, but the cost lands on the user's data": switching changes the face **embeddings themselves**, so every vector stored in Immich becomes incomparable with new ones. Immich must re-run its face jobs across the whole library — clusters rebuilt from scratch, every named person reconfirmed by hand. Hours of processing plus manual work, and nobody should have their people-tagging reset by pulling an update.

A legitimate choice for someone indexing a large library from scratch who accepts the accuracy trade knowingly. Not a default.

**The model choice and the request mode compound, and anyone choosing models should know it.** Shrinking device time raises the ceiling on what request pipelining can hide behind it, so the two multiply rather than add. Measured with `REQUEST_MODE=threadpool` at C=8: **1.58×** with `arcface_r50`, **2.24×** with `arcface_mobilefacenet`. The faster model does not merely make each request quicker — it makes concurrency worth more, because a smaller share of the request is the serialised device section. Evaluate the pair together rather than each alone.

Note mobilefacenet's 5191 FPS matches Hailo's published figure exactly, while `arcface_r50` is 3.2× off its published 113. That corroborates this one number; it is not a rule — `scrfd_2.5g` is equally small and 1.86× off.

**Golden references are keyed by recognition model**, since the face embedding changes completely. Switching makes `golden.sh check` skip rather than fail, and needs a fresh `generate`.

> **Unverified against the HEF:** output layer name, embedding dimension and input crop size are carried in `ARCFACE_VARIANTS` as ArcFace-family defaults (`fc1`, 512, 112×112) and marked TODO in `config.py`. Confirm with `hef_inspect` before shipping — see [Evaluating a candidate on this hardware](#evaluating-a-candidate-on-this-hardware).

#### `scrfd_10g` — **shipped**, selectable with `FACE_DETECTOR=scrfd_10g`

No longer a recommendation: both detectors ship in the image and `setup.sh` downloads both, so switching is a container restart.

| | mAP (Hailo-8) | Device latency (measured) | Share of a 96.8 ms face request |
|---|---|---|---|
| `scrfd_2.5g` *(default)* | 76.4 | 2.53 ms | 2.6% |
| `scrfd_10g` | **82.1** | 4.40 ms | 4.5% |

**+5.7 mAP for +1.87 ms**, measured — under 2% of a face request. Better on small, occluded, and profile faces. `scrfd_2.5g` stays the default because it is the tested configuration and the one every existing golden reference was generated against.

> **Correction.** A previous revision warned that 10g's cost would "rise by more than the 2.4× the FPS ratio suggests", reasoning that per-burst overhead grows with model size. Measured, it is 1.74× (4.40 vs 2.53 ms) — *less* than the published ratio implied, and trivial in absolute terms. Note also that 10g's published figure (440 FPS) matched measurement exactly while 2.5g's did not, which is what made the published ratio misleading.

**The output layer names differ between the two variants, and this is why `config.py` carries them per model.** Read from each HEF with `hef_inspect`:

| stride | `scrfd_2.5g` cls / box | `scrfd_10g` cls / box |
|---|---|---|
| 8 | `conv42` / `conv43` | `conv41` / `conv42` |
| 16 | `conv49` / `conv50` | `conv49` / `conv50` — **identical** |
| 32 | `conv55` / `conv56` | `conv56` / `conv57` |

Stride 16 happens to match; strides 8 and 32 are shifted by one. Copying 2.5g's names onto 10g would match only the stride-16 pair, so detection would return mid-size faces and silently miss everything larger and smaller — `decode_scrfd` warns only when *nothing* matches, not when some do. That near-miss is exactly the trap the per-model registry exists to prevent, and the reason these names must be read from the HEF rather than assumed.

More detections also means more ArcFace crops — cheap now that recognition is batched, since extra faces mostly fill bursts already being paid for, and bounded by `ScrfdConfig.max_faces` (default 100, at most 13 bursts).

**No re-index required.** The recognition model is unchanged, so face embeddings are unaffected by the detector choice and existing clusters survive. CLIP compatibility is untouched.

**But it is user-visible.** Detecting more faces is the point, and Immich will surface them: re-running face detection over the library will find new faces in already-scanned photos, so expect new people to appear and existing people to gain photos.

**Golden references are keyed by detector**, because a different detector produces different crops and therefore different face embeddings. Switching detectors makes `golden.sh check` skip rather than fail, and needs a fresh `generate`.

### Worth doing

#### SigLIP2 B/32-256 — a quality upgrade at zero speed cost

`siglip2_b_32_256_image_encoder.hef` and `siglip2_b_32_256_text_encoder.hef` exist for Hailo-8 in v2.18.0 and v2.19.0 (verified, HTTP 200; ~196 MB and ~189 MB).

Immich offers a matching model, **`ViT-B-32-SigLIP2-256__webli`** — same patch size, same resolution. If Hailo's HEF is built from the same weights, this preserves the Immich interoperability that makes SigLIP valuable, while improving on it:

| | English recall | Multilingual | Host memory (Immich's runtime) | Device latency (measured) |
|---|---|---|---|---|
| `ViT-B-16-SigLIP__webli` *(current)* | 81.9% | baseline | 1,081 MiB | 205.70 ms |
| `ViT-B-32-SigLIP2-256__webli` | **82.28%** | **+1–3 pp** | 3,061 MiB | 205.64 ms |

The memory column is why Immich's own docs treat B/32-SigLIP2 as a trade-off — and **it does not apply here.** On Hailo the weights live in a compiled HEF on the accelerator, and the two HEFs are within 4% of each other in size.

> **Correction.** A previous revision of this document claimed SigLIP2 B/32-256 "should also be *faster*: 64 patches against 196 — roughly a third of the tokens." **That is measured false: 205.64 ms against 205.70 ms, a 0.03% difference.** The prediction assumed compute determines latency; on this device size does. That pair is now the strongest single piece of evidence for the weight-streaming explanation above.

So the case is a **pure quality upgrade at identical cost**: +0.4 pp English recall, +1–3 pp multilingual, no speed penalty, no memory penalty on this hardware.

**Still unproven:** that Hailo's HEF matches Immich's weights, preprocessing, tokenizer, and pooling. Verify before shipping — extract from HuggingFace `google/siglip2-base-patch32-256`, run both on identical images, and compare cosine similarity. Near-1.0 confirms it; anything lower means a private embedding space and the upgrade forces a re-index it cannot justify.

### Researched, deferred

**Face landmark alignment before ArcFace.** `face_landmarks_lite` was added in v2.18.0. Aligning crops by landmarks typically improves embeddings for rotated and profile faces. Cost is a third model in the cascade and another inference per detected face — and by the analysis above, that cost is mostly its HEF size, which should be benchmarked before any integration work. Plausible for v2; unquantified.

### Not worth doing

| Candidate | Why not |
|---|---|
| `scrfd_500m` | 68.7 mAP — a large accuracy loss. Measured, `scrfd_2.5g` costs 2.53 ms of a 96.8 ms request, so there is at most ~2 ms to win. Not a trade worth making |
| `retinaface_mobilenet_v1` | 81.2 mAP against `scrfd_10g`'s 82.1 at 4.40 ms measured. Its own published 163 FPS is **unverified** — and published figures have proven unreliable — but it would have to be extraordinarily fast to justify lower accuracy |
| TinyCLIP 8m / 40m / 61m | HEFs exist, but every one leaves the user in a private embedding space. A better TinyCLIP is still not an Immich-compatible one |
| `siglip_l_16_256`, `siglip2_l_16_256` | No Hailo-8 HEF exists (verified: 403). Also: an L-sized model would be larger still, and on this device that means proportionally slower |
| YOLO / object detection | Immich's remote-ML protocol exposes no object-detection results. A parallel feature, not a better model for existing jobs |
| `all-MiniLM-L6-v2` (v2.19.0) | Text-only sentence embeddings. Not a joint image-text space, so it cannot serve smart search |
| PaddleOCR replacements | No better Hailo-8 OCR model found. v5.3 improvements target Hailo-10H/15H/15L |
| Whisper | Exists in v2.18.0, but audio transcription is a different product surface than this worker serves |

---

## Model Zoo versions

**v2.19.0 is the newest release** for Hailo-8 (verified; v2.20.0 and above do not exist). Hailo-8 and 8L stay on the v2.x branch with Dataflow Compiler v3.x — the v5.x branch targets Hailo-10/15 and is not a drop-in path.

**All eight models this project uses are available under a single prefix**, `v2.18.0/hailo8` or `v2.19.0/hailo8` (verified, 8/8 HTTP 200 at both). `setup.sh` currently straddles v2.17.0 and v2.18.0 for historical reasons — SCRFD, ArcFace and TinyCLIP were added when v2.17.0 was current. Collapsing to one prefix is a safe simplification:

- The HEFs at each version are **recompiles, not copies** — sizes and ETags differ at every version.
- **SCRFD output layer names are identical** across v2.17.0, v2.18.0 and v2.19.0 (verified by extracting layer strings from all three HEFs), so `ScrfdConfig.output_layers` needs no change.
- **Quantization parameters do differ** between recompiles, but the pipeline reads them from the loaded HEF at startup, so this is handled automatically. See the README on `CLIP_QUANT_SOURCE`.

Since HEF size predicts latency for large models, a recompile that changes file size may change performance. The benchmark below takes seconds; run it after any version change rather than assuming parity.

There is no evidence of face, CLIP, or OCR improvements in v2.19.0 relevant to these roles. Treat it as a source for new optional models, not a migration to make for its own sake.

---

## Evaluating a candidate on this hardware

This is the reusable part of everything above. **Benchmark the HEF before writing any integration code** — one command tells you latency and throughput on your own device, and it costs nothing to be wrong.

```bash
# 1. Stop the worker. It holds the VDevice exclusively; hailortcli cannot
#    open the device while the container is running.
docker stop immich-ml-hailo

# 2. Benchmark the HEF. hailortcli ships in the base image, so no install.
docker run --rm --device=/dev/hailo0:/dev/hailo0 \
  -v "$PWD/models:/models:ro" \
  "hailo-base:v$HAILORT_VERSION" \
  hailortcli benchmark /models/<candidate>.hef

# 3. Restart the worker.
docker start immich-ml-hailo
```

Read the **latency** and **FPS** from the output, then:

1. **Compare against the published figure, and expect to be surprised.** Three of five checked here were off by 1.9–3.2×.
2. **Check the HEF size.** Above ~30 MiB, expect latency ≈ size ÷ ~1 GB/s. A model twice the size will cost roughly twice as much regardless of how much cheaper its compute looks on paper.
3. **Compare against the stage it replaces**, using the in-pipeline table above — not against the model it replaces in isolation. A detector that is 2 ms slower on a 96.8 ms request does not matter; a CLIP encoder that is 50 ms slower does.
4. **Then, and only then, wire it up.**

## Evaluating a swap yourself

1. **Check the HEF exists** for `hailo8` (or `hailo8l`) at the version you want.
2. **Benchmark it**, per the section above.
3. **Inspect it**: `python3 -m ml_target.hef_inspect /app/models/<model>.hef` prints stream names, shapes, and quantization parameters.
4. **Layer names** — only SCRFD hardcodes them, in `ScrfdConfig.output_layers`. Wrong names do not crash; face detection silently returns zero faces. See the README's Hailo-8L section for how to derive them.
5. **Quantization parameters** are read from the HEF automatically. The constants in `config.py` are only a fallback.
6. **Verify embeddings, not shapes.** The test suite asserts dimensions, which will not catch a model producing correctly-shaped but wrong output. `./tests/golden.sh` compares against a stored reference; for any CLIP change also compare against a reference implementation before trusting search results.

---

## Verification status

**Measured on the production Hailo-8** (one device, HailoRT 4.24.0): all `hailortcli benchmark` latencies, throughputs and HEF sizes in [Measured device performance](#measured-device-performance); all in-pipeline stage p50s via `scripts/benchmark.sh`; the face batch-size sweep. Correctness at the shipped batch settings was verified independently — 19/19 assertions on both backends, `golden.sh check` at cosine 1.000000000000 for all three embeddings, and a 64-region OCR grid returning `exact=64 SHIFTED=0` with padding active.

**Derived from measurement:** the two-term batching cost model and its 26.3 ms / 1.11 ms coefficients (fitted, within 0.1% at batch 4 and 8); the OCR per-frame speedup of ~3.6×, backed out of one data point; the effective GB/s column, which is HEF size ÷ measured latency.

**Inference, not observation:** that large-model latency is bound by weight streaming across PCIe. It rests on the size/latency correlation, the tight ~0.9–1.05 GB/s convergence of the three largest models, and the SigLIP pair showing 3.06× compute difference at 0.03% latency difference. The mechanism has not been directly instrumented — no bus counters were read.

**Estimated:** face recognition at 2–3 ms per request with `arcface_mobilefacenet`, extrapolated from its 1.09 ms device latency plus the host-side share seen in other stages. Not measured in-pipeline.

**Assumed:** the distribution of faces per photo behind the weighted batching table. The per-N costs are measured; the weighting is a guess. The `faces=` field in the request log records the real distribution — a histogram over one library scan would replace the assumption with a fact.

**Verified externally, not here:** HEF availability and sizes via HTTP HEAD against Hailo's S3; SCRFD layer-name stability across three Model Zoo versions; Immich's model catalog and recall benchmarks from its published documentation; accuracy figures (mAP, LFW) from Hailo's reference material — these are **not** independently verified, and given how the FPS figures fared, they deserve the same scepticism.

**Not verified:** SigLIP2 embedding compatibility with Immich; the practical quality delta of landmark alignment; device latency for either OCR model; `retinaface_mobilenet_v1`'s published throughput; end-to-end in-pipeline throughput of any alternative model.
