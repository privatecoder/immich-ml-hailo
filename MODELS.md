# Model Selection

Why this project runs the models it does, which alternatives were evaluated, and what is worth changing.

Every model must exist as a compiled `.hef` for Hailo-8 in the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) — this project does not train or compile models. Within that constraint the choices below are deliberate, and the alternatives were evaluated against four criteria:

1. **Immich embedding compatibility** — whether a change forces users to re-index their library.
2. **Accuracy** on the task.
3. **Throughput on Hailo-8**, not on a GPU. All models share one 26 TOPS device.
4. **Whether it is the bottleneck.** Making a fast stage faster buys nothing.

---

## What ships today

| Role | Model | Source | Output |
|---|---|---|---|
| Face detection | `scrfd_2.5g` | Model Zoo v2.17.0 | 9 output streams, 3 strides |
| Face recognition | `arcface_r50` | v2.17.0 | 512-dim, L2-normalized |
| Smart search (default) | `tinyclip_vit_39m_16_text_19m_yfcc15m` | v2.17.0 | 512-dim |
| Smart search (quality) | `siglip_b_16` | v2.18.0 | 768-dim |
| OCR detection | `paddle_ocr_v5_mobile_detection` | v2.18.0 | 544×960 probability map |
| OCR recognition | `paddle_ocr_v5_mobile_recognition` | v2.18.0 | CTC over 18,383 characters |

## The CLIP backend decision

This is the only choice with consequences beyond this worker, because CLIP embeddings are written into Immich's search index.

**`siglip` — 768-dim, the quality option.** Embeddings are interchangeable with Immich's own `ViT-B-16-SigLIP__webli`. A user can switch between this worker and Immich's built-in ML worker without re-running Smart Search. That interoperability is the reason SigLIP exists here.

**`tinyclip` — 512-dim, the throughput option.** Hailo's device-only benchmarks put it at roughly 60 FPS image encoding against ~14 for SigLIP, and it has a simpler code path (it requests `FLOAT32` output, so no dequantization). Treat that ratio as the comparison and not as a per-request cost — measured, SigLIP image encoding is 224 ms per call, about a third of the vendor figure; see [Throughput](#throughput-the-measured-cost-model). It matches **no** Immich CLIP model, so a user on TinyCLIP is in a private embedding space: moving to or from Immich's own worker, or to SigLIP, means re-processing the entire library.

TinyCLIP remains the default for historical reasons and because it is the fastest path for an initial index on a large library. **SigLIP is the better recommendation for most users**, and the README's `docker run` examples lead with it.

Changing CLIP backend on an existing library always requires re-running Smart Search — the dimensions differ (512 vs 768), and even at equal dimensions the vector spaces are unrelated.

---

## Throughput: the measured cost model

Everything in this section is **measured on the production Hailo-8** (2360×2360 image, SigLIP backend, 20 iterations, p50 of the per-stage request timings) unless marked otherwise. It replaces the earlier open question about whether this worker was device-bound.

### Face recognition scales with bursts, not frames

`rec_infer_batch` for 5 faces, varying the configured device batch size:

| batch | frames sent | p50 | per frame | speedup |
|---|---|---|---|---|
| unset | 5 | 144.2 ms | 28.84 ms | — |
| 2 | 6 | 88.4 ms | 14.73 ms | 1.96× |
| 4 | 8 | 61.5 ms | 7.69 ms | 3.75× |
| 8 | 8 | 35.2 ms | 4.40 ms | **6.55×** |

Those numbers fit a two-term model to within 0.1% at batch 4 and 8, and 3.2% at batch 2:

> **cost = ceil(N / B) × (26.3 ms burst overhead + B × 1.11 ms per frame)**

The shape is the point. **Actual per-face compute is ~1.1 ms.** The unbatched path was paying the entire 26.3 ms burst overhead on *every single frame*. ArcFace R50 is a large multi-context model and each burst cycles its contexts once; batching amortises that cycle across the whole burst.

Two consequences that generalise beyond this model:

- **Padding is nearly free.** A burst costs about the same whether it carries one real frame or eight, so padding a short batch up to the burst size wastes little. That is why batching wins even for 2 faces.
- **The datasheet understates it.** Hailo publishes ArcFace R50 at 113 FPS batch-1 and 391 FPS batch-8, implying a 3.46× gain. The real gain here is 6.55×, because the vendor benchmark's batch-1 figure does not carry this deployment's per-burst cost.

### What that means per photo

Weighted over an **assumed** distribution of faces per photo (1: 40%, 2: 25%, 3: 15%, 5: 12%, 8: 6%, 20: 2%) — the costs are measured, the distribution is a guess, and the weighted totals inherit that uncertainty:

| batch | weighted mean | vs unbatched |
|---|---|---|
| unset | 81.6 ms | — |
| 2 | 49.9 ms | 1.64× |
| 4 | 38.7 ms | 2.11× |
| **8** | **36.6 ms** | **2.23×** |
| 16 | 44.9 ms | 1.82× |

**8 is the optimum**, and 16 is worse only because single-face photos pad so wastefully into a 16-frame burst. Batch 8 beats unbatched at every N ≥ 2; at N = 1 it costs 28.8 → 35.2 ms, about 3% of a ~205 ms request.

At the `max_faces` cap of 100 the difference is stark: 13 bursts totalling ~457 ms, against ~2884 ms unbatched — 6.3×.

### OCR recognition

Same mechanism, smaller model, so a smaller per-burst overhead to amortise. `ocr_rec_batch` for 12 regions: **160.1 ms → 58.8 ms** at batch 8. Backing out the 1.33× padding tax (16 frames sent for 12 real) gives a per-frame speedup of ~3.6×. One data point is not enough to solve the two-term model separately for this model, so its burst overhead is not quantified.

### Burst overhead scales with model size

`det_infer` — one SCRFD frame, one burst — is **8.8 ms**, well under ArcFace's 26.3 ms burst cost. SCRFD 2.5G is a much smaller network with fewer contexts to cycle. This is the number to reason from when considering a larger detector: see the `scrfd_10g` entry below.

### CLIP is now the dominant cost, and batching cannot help it

`clip_image_infer` is **224 ms** and `clip_text_infer` **217.9 ms** (SigLIP). Against a fully batched face pipeline at 35 ms, CLIP is now the largest single item in a scan.

**Batching cannot address it.** Immich sends one asset per request, so the visual encoder receives exactly one frame per call — there is no batch to form. A device batch size would only pad one real frame up to eight and make it worse, which is why it is not configured for CLIP.

Whether that 224 ms is mostly device work or mostly per-call overhead is **not yet measured**. Hailo's device-only figure of ~14 FPS for SigLIP B/16 implies roughly 71 ms of compute, which would leave a substantial remainder — but that is inference from a vendor benchmark, not a measurement, and the honest answer is that it is untested. If the remainder is per-call overhead, it is the same class of cost that batching just eliminated for faces, and the way to attack it is virtual-stream reuse rather than batching.

---

## Evaluated alternatives

### Worth doing

#### SigLIP2 B/32-256 — the strongest available upgrade

`siglip2_b_32_256_image_encoder.hef` and `siglip2_b_32_256_text_encoder.hef` exist for Hailo-8 in v2.18.0 and v2.19.0 (verified, HTTP 200; ~196 MB and ~189 MB).

Immich offers a matching model, **`ViT-B-32-SigLIP2-256__webli`** — same patch size, same resolution. If Hailo's HEF is built from the same weights, this preserves the Immich interoperability that makes SigLIP valuable, while improving on it:

| | English recall | Multilingual | Host memory (Immich's runtime) |
|---|---|---|---|
| `ViT-B-16-SigLIP__webli` *(current)* | 81.9% | baseline | 1,081 MiB |
| `ViT-B-32-SigLIP2-256__webli` | **82.28%** | **+1–3 pp** | 3,061 MiB |

The memory column is why Immich's own docs treat B/32-SigLIP2 as a trade-off — and **it does not apply here.** On Hailo the weights live in a compiled HEF on the accelerator, and the two HEFs are within 4% of each other in size. The penalty that makes this model unattractive in Immich's worker is simply absent on this one.

It should also be *faster*: patch-32 at 256px is 64 patches against patch-16 at 224px at 196 — roughly a third of the tokens.

**Unproven:** that Hailo's HEF matches Immich's weights, preprocessing, tokenizer, and pooling. Verify before shipping — extract from HuggingFace `google/siglip2-base-patch32-256`, run both on identical images, and compare cosine similarity. Near-1.0 confirms it; anything lower means a private embedding space and the upgrade is not worth a forced re-index.

#### `scrfd_10g` as an optional quality profile

| | mAP (Hailo-8) | FPS |
|---|---|---|
| `scrfd_2.5g` *(current)* | 76.4 | 1,057 |
| `scrfd_10g` | **82.1** | 440 |

+5.7 mAP for 2.4× slower detection — and detection is nowhere near the bottleneck. Measured, `det_infer` is 8.8 ms against `clip_image_infer` at 224 ms. Better on small, occluded, and profile faces.

**The measured cost model sharpens this.** Detection runs one frame per request, so it pays one full burst overhead and gets no benefit from batching. SCRFD 2.5G's burst is cheap (8.8 ms) precisely because it is a small network with few contexts to cycle; a 10g build is larger and would cycle more, so expect its cost to rise by more than the 2.4× the FPS ratio suggests — per-burst overhead grows with model size, and at one frame per request that overhead is the whole cost. Budget for it, but 8.8 ms has room to grow several times over before it rivals CLIP.

Requires setting `ScrfdConfig.output_layers` for the 10g build; see the Hailo-8L section of the README for the procedure, which applies identically here. More detections also means more ArcFace crops — but with batching that is now much cheaper than it was, since extra faces mostly fill bursts that were being paid for anyway, and the total is bounded by `ScrfdConfig.max_faces` (default 100, at most 13 bursts).

Does not affect CLIP compatibility. Face embeddings are unchanged, so existing clusters survive.

#### `arcface_mobilefacenet` as an optional fast profile

| | LFW accuracy | FPS |
|---|---|---|
| `arcface_r50` *(current)* | 99.7% | 113 (batch 1) / 391 (batch 8) |
| `arcface_mobilefacenet` | 99.4% | 5,191 |

13× throughput for 0.3 pp accuracy — on paper.

**Measurement has weakened this case rather than strengthened it.** When this entry was written, recognition cost 144 ms for 5 faces and the argument was that it still was not the bottleneck. With device batching it is **35 ms for 5 faces**, and per-face compute is ~1.1 ms of a 26.3 ms burst. Nearly all of what remains is per-burst overhead, which is a function of cycling the model's contexts — so a smaller model would reduce the 1.1 ms that is already negligible, and only partly the 26.3 ms that dominates. The 13× figure is a batch-saturated throughput ratio and will not translate.

Against that, **changing face embeddings disturbs existing face clusters** — Immich would need its face jobs re-run across the library. The cost is unchanged and the benefit is now smaller than it looked. Still a legitimate optional profile for a very large library being indexed from scratch; a worse default than before.

### Researched, deferred

**Face landmark alignment before ArcFace.** `face_landmarks_lite` was added in v2.18.0. Aligning crops by landmarks typically improves embeddings for rotated and profile faces. Cost is a third model in the cascade and another inference per detected face. Plausible for v2; unquantified.

### Not worth doing

| Candidate | Why not |
|---|---|
| `scrfd_500m` | 68.7 mAP — a large accuracy loss for a modest speed gain over 2.5g, which is already not the bottleneck |
| `retinaface_mobilenet_v1` | 81.2 mAP at 163 FPS. `scrfd_10g` is more accurate *and* 2.7× faster |
| TinyCLIP 8m / 40m / 61m | HEFs exist, but every one leaves the user in a private embedding space. A better TinyCLIP is still not an Immich-compatible one |
| `siglip_l_16_256`, `siglip2_l_16_256` | No Hailo-8 HEF exists (verified: 403) |
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

There is no evidence of face, CLIP, or OCR improvements in v2.19.0 relevant to these roles. Treat it as a source for new optional models, not a migration to make for its own sake.

---

## Evaluating a swap yourself

1. **Check the HEF exists** for `hailo8` (or `hailo8l`) at the version you want.
2. **Inspect it**: `python3 -m ml_target.hef_inspect /app/models/<model>.hef` prints stream names, shapes, and quantization parameters.
3. **Layer names** — only SCRFD hardcodes them, in `ScrfdConfig.output_layers`. Wrong names do not crash; face detection silently returns zero faces. See the README's Hailo-8L section for how to derive them.
4. **Quantization parameters** are read from the HEF automatically. The constants in `config.py` are only a fallback.
5. **Verify embeddings, not shapes.** The test suite asserts dimensions, which will not catch a model that produces correctly-shaped but wrong output. For any CLIP change, compare embeddings against a reference implementation before trusting search results.

## Verification status

**Measured on the production Hailo-8:** the batch-size sweep and the resulting cost model; per-stage p50 latencies for every pipeline stage (`scripts/benchmark.sh`); the OCR batch-8 improvement; `det_infer`, `clip_image_infer` and `clip_text_infer` costs. Correctness at batch 8 was verified independently — 19/19 assertions on both backends, `golden.sh check` at cosine 1.000000000000 for all three embeddings, and a 64-region OCR grid returning `exact=64 SHIFTED=0` with padding active on the final chunk.

**Derived from measurement:** the two-term cost model and its 26.3 ms / 1.11 ms coefficients, fitted to the batch sweep (within 0.1% at batch 4 and 8); the OCR per-frame speedup of ~3.6×, backed out of one data point after removing the padding tax.

**Assumed:** the distribution of faces per photo used for the weighted table. The per-N costs are measured; the weighting is a guess, and the weighted totals inherit that. The `faces=` field in the request log records the real distribution — a histogram over one library scan would replace the assumption with a fact.

**Verified externally, not here:** HEF availability and sizes via HTTP HEAD against Hailo's S3; SCRFD layer-name stability across three Model Zoo versions by extracting strings from the HEFs; Immich's model catalog and recall benchmarks from its published documentation; Hailo's accuracy and FPS figures from Hailo's own reference material. Note that Hailo's FPS figures are device-only benchmarks at saturation and do not predict this worker's per-request cost — the measured 224 ms for SigLIP image encoding is roughly a third of the ~14 FPS the datasheet implies.

**Not verified:** SigLIP2 embedding compatibility with Immich; the practical quality delta of landmark alignment; whether CLIP's 224 ms is dominated by device work or per-call overhead; end-to-end throughput of any alternative model in this worker.
