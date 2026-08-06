# Immich ML Worker for Hailo-8 PCIe Accelerators

An external ML inference worker for [Immich](https://immich.app/) that offloads **face detection/recognition**, **CLIP smart search**, and **OCR** to a **Hailo-8** PCIe accelerator. It replaces Immich's built-in ONNX-based ML worker with a FastAPI service that speaks the same `/predict` protocol — but runs inference on the Hailo-8 hardware at a fraction of the power.

## Immich Jobs Handled by This Worker

This worker accelerates the following Immich jobs on the Hailo-8:

| Immich Job | Hailo Model | Notes |
|------------|-------------|-------|
| **Smart Search** | TinyCLIP ViT-39M/16 **or** SigLIP B/16 | CLIP image embeddings for semantic search |
| **Duplicate Detection** | (uses Smart Search embeddings) | No separate inference — reuses CLIP embeddings |
| **Face Detection** | SCRFD 2.5G *or* SCRFD 10G | Detects faces in images — see [Face detector](#face-detector) |
| **Facial Recognition** | ArcFace R50 *or* MobileFaceNet | Generates face embeddings for grouping people — see [Face recognition model](#face-recognition-model) |
| **OCR** | PaddleOCR v5 mobile | Extracts text from images |

Other Immich jobs (Generate Thumbnails, Extract Metadata, Transcode Videos, Sidecar Metadata, External Libraries, Storage Template Migration) run on the Immich server itself and are not affected by this worker.

### CLIP Backend Choice

Two CLIP backends are available, selectable via the `CLIP_BACKEND` environment variable:

|  | TinyCLIP (default) | SigLIP |
|--|-------------------|--------|
| Image input | 224x224 (center-crop) | 224x224 (squash resize) |
| Embedding dim | 512 | 768 |
| Image encode — **measured device** | **46.7 ms** (18.9 FPS) | **205.7 ms** (4.46 FPS) |
| Image encode — in pipeline | — | 224.2 ms |
| Image FPS — *vendor, published* | ~60 | ~14 |
| Text FPS — *vendor, published* | ~18 | ~17 |
| Search quality | Good | Better |
| Immich model match | None | `ViT-B-16-SigLIP__webli` |

> **The vendor row is roughly 3× optimistic for both backends** — measured on a Hailo-8 with `hailortcli benchmark`, TinyCLIP is 18.9 FPS against a published ~60, and SigLIP is 4.46 against ~14.
>
> Their *ratio* survives: 4.3× published, 4.2× measured. So the published figures are usable for choosing **between** the two models and misleading for predicting **how long a library scan takes**. Size a scan from the measured milliseconds.
>
> This is not specific to CLIP. Across the models benchmarked for this project, three of five published figures were off by 1.9–3.2× and two matched exactly, with no way to tell which in advance. See [MODELS.md](MODELS.md) for the measurements and the reason (large models on this device are bound by streaming weights across PCIe, not by compute).

**TinyCLIP** is significantly faster — **4.4× measured** on the device (46.7 ms against 205.7 ms), which is the one place the vendor ratio held up — but produces embeddings incompatible with any Immich default model.

**SigLIP** produces the same embeddings as Immich's `ViT-B-16-SigLIP__webli` (same underlying Google model weights). This means you can switch between this Hailo worker and the official Immich ML worker **without re-running Smart Search** — the embeddings are compatible. The text encoder output is also simpler: already pooled to a single vector (no CPU-side projection needed).

The two backends come off the device differently. **TinyCLIP** outputs FLOAT32 directly — no dequantization step. **SigLIP** outputs UINT16 and is dequantized to float32 using per-model quantization parameters (`output_qp_scale` / `output_qp_zp` in `config.py`). Both are L2-normalized before being returned.

See [MODELS.md](MODELS.md) for why each model was chosen, which alternatives were evaluated (and rejected), and what a future upgrade path looks like.

## Prerequisites

- **Hailo-8** M.2 PCIe accelerator — the tested configuration. **Hailo-8L** is supported via [documented model substitution](#hailo-8l-model-substitution), but has not been verified on hardware.
- **Host Hailo drivers** installed and working. For Unraid, use the `Hailo RT Driver` app by ich777. See [hailort-drivers](https://github.com/hailo-ai/hailort-drivers) (v4 branch for Hailo-8/8L, v5 for Hailo-10H/15H).
- **Docker** on the host

### ⚠️ HailoRT version must match the host driver exactly

The HailoRT library inside the container and the `hailo_pci` kernel module on the host must be the **same version**. Any difference — even a patch bump — makes the container fail at startup with:

```
CHECK failed - Driver version (X) is different from library version (Y)
HAILO_INVALID_DRIVER_VERSION(76)
```

Find your host's version first, and use it everywhere below:

```bash
modinfo hailo_pci | grep '^version:'
# or
cat /sys/module/hailo_pci/version
```

Read the **kernel module** version. Do not use `hailortcli fw-control identify` — it reports board/firmware identity and the CLI's own HailoRT version, not the driver's.

> **Unraid users:** the ich777 `Hailo RT Driver` plugin updates the driver on its own schedule, and an Unraid OS update can move it too. A driver bump breaks a previously working container. After any Unraid or plugin update, re-check `modinfo hailo_pci` and rebuild the images if the version changed. There is no `apt-mark hold` equivalent on Unraid.

The examples in this README use `4.24.0`. **Substitute your host's actual version.**

## Download HailoRT Packages (required for both Quick and Manual Setup)

The HailoRT runtime packages require a free [Hailo Developer Zone](https://hailo.ai/developer-zone) account and cannot be downloaded automatically.

Go to [Software Downloads](https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l) and select:

| Filter | Value |
|--------|-------|
| Software Package | AI Software Suite |
| Software Sub-Package | HailoRT |
| Architecture | **x86** or **ARM64** (match your host) |
| OS | Linux |
| Python Version | 3.12 |

Pick the **version that matches your host driver** (see above), then download the two files for your platform and place them in `hailo-rt-4/`. Keep the filenames exactly as downloaded — the build looks them up by name.

**x86_64:**
- _HailoRT – Python package (whl) for Python 3.12, x86_64_ → `hailort-<version>-cp312-cp312-linux_x86_64.whl`
- _HailoRT – Ubuntu package (deb) for amd64_ → `hailort_<version>_amd64.deb`

**ARM64 (aarch64):**
- _HailoRT – Python package (whl) for Python 3.12, aarch64_ → `hailort-<version>-cp312-cp312-linux_aarch64.whl`
- _HailoRT – Ubuntu package (deb) for arm64_ → `hailort_<version>_arm64.deb`

For example, on an x86_64 host running driver 4.24.0: `hailort-4.24.0-cp312-cp312-linux_x86_64.whl` and `hailort_4.24.0_amd64.deb`.

> **Why Python 3.12?** The Docker base image uses Ubuntu 24.04 LTS, which ships Python 3.12 as the system default. HailoRT 4.x supports Python 3.10, 3.11, and 3.12 — using the system Python avoids managing a venv, and 3.12 has the best performance (~5% faster runtime than 3.11).

## Quick Setup

Once the HailoRT packages are in `hailo-rt-4/`, run:

```bash
HAILORT_VERSION=4.24.0 ./setup.sh
```

Set `HAILORT_VERSION` to your host driver's version. It is threaded through both Docker builds, both weight-extraction scripts, and the image tags, so a future driver bump is a one-variable change.

**`HAILORT_VERSION` is required — there is no default.** A bare `./setup.sh` aborts immediately and prints the `modinfo` command to find your host's version. Guessing would build and tag cleanly, then fail at runtime with `HAILO_INVALID_DRIVER_VERSION(76)`, which is precisely the failure this is meant to prevent.

This will check for required files, offer to download missing models, build both Docker images, extract the CLIP text weights for both backends, and run the test suite against both. See the [Manual Setup](#manual-setup) section below if you prefer to do each step yourself.

## Manual Setup

### Step 1: Download HEF Models (skip if you used Quick Setup)

Download the pre-compiled `.hef` model files from the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/tree/master/docs/public_models) and place them in `models/`.

The `curl` commands below are for **Hailo-8**. For **Hailo-8L**, see [Hailo-8L model substitution](#hailo-8l-model-substitution) at the end of this step — the URLs differ, and the SCRFD output layer names in `config.py` must be re-derived from your own HEF.

**Face Detection** ([model card](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_face_detection.rst)):
```bash
# Default detector
curl -fLo models/scrfd_2.5g.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/scrfd_2.5g.hef

# Higher-accuracy alternative, selectable with FACE_DETECTOR=scrfd_10g (6.9 MB)
curl -fLo models/scrfd_10g.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/scrfd_10g.hef
```

**Face Recognition** ([model card](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_face_recognition.rst)):
```bash
curl -Lo models/arcface_r50.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/arcface_r50.hef
```

**CLIP Image Encoder** ([model card](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_zero_shot_classification.rst)):
```bash
curl -Lo models/tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder.hef
```

**CLIP Text Encoder** ([model card](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_text_image_retrieval.rst)):
```bash
curl -Lo models/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder.hef
```

**OCR Text Detection** ([model card](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_text_detection.rst)):
```bash
curl -Lo models/paddle_ocr_v5_mobile_detection.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/paddle_ocr_v5_mobile_detection.hef
```

**OCR Text Recognition** ([model card](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_text_recognition.rst)):
```bash
curl -Lo models/paddle_ocr_v5_mobile_recognition.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/paddle_ocr_v5_mobile_recognition.hef
```

**SigLIP Image Encoder** ([model card](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_zero_shot_classification.rst)) — only needed for SigLIP backend:
```bash
curl -Lo models/siglip_b_16_image_encoder.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/siglip_b_16_image_encoder.hef
```

**SigLIP Text Encoder** ([model card](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_text_image_retrieval.rst)) — only needed for SigLIP backend:
```bash
curl -Lo models/siglip_b_16_text_encoder.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/siglip_b_16_text_encoder.hef
```

#### Hailo-8L model substitution

**Quick Setup does not work for Hailo-8L.** `setup.sh` hardcodes the Hailo-8 URLs in `HEF_BASE` and `HEF_V218` (`setup.sh:65-66`), so an 8L card requires this Manual Setup path.

All eight models are available for Hailo-8L under a **single** prefix:

```
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/
```

This is simpler than the Hailo-8 path above, which straddles two Model Zoo releases — v2.17.0 for SCRFD, ArcFace, and TinyCLIP; v2.18.0 for SigLIP and OCR. For 8L, take everything from **v2.18.0/hailo8l**; v2.17.0 does not carry TinyCLIP for that device.

```bash
HEF_8L="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l"

for m in scrfd_2.5g arcface_r50 \
         tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder \
         tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder \
         siglip_b_16_image_encoder siglip_b_16_text_encoder \
         paddle_ocr_v5_mobile_detection paddle_ocr_v5_mobile_recognition; do
  curl -Lo "models/$m.hef" "$HEF_8L/$m.hef"
done
```

The supporting files in Step 2 (BPE vocabulary, `spiece.model`, OCR dictionary) are device-independent — download those unchanged.

##### Quantization parameters: handled automatically

Quantization parameters are a property of **how a particular HEF was compiled**, not of the model architecture, so a Hailo-8L build will generally differ from the Hailo-8 values that used to be hardcoded. Getting them wrong is dangerous precisely because it is *silent*: `dequantize_uint16()` applies whatever scale and zero-point it is handed, so wrong constants produce no exception, no warning, and no obviously broken output — just **plausible numbers that are subtly wrong**. The embeddings come out correctly shaped and correctly normalized and are quietly degraded, and the only symptom is smart search returning worse matches than it should.

**You no longer have to do anything about this.** The pipeline reads the quantization parameters from whichever HEF it actually loads, at startup. The constants in `config.py` are now only a fallback, used when the HailoRT version does not expose quantization info. At startup the log prints both the value read from the HEF and the config constant for every affected stream, and warns loudly if they disagree — so a mismatch on an 8L card is visible rather than silent:

```
INFO  siglip_image.output quant: HEF scale=0.00041 zp=8800 | config scale=0.000325549... zp=9506 | MISMATCH
WARN  siglip_image.output QUANTIZATION MISMATCH — ... Using the HEF values ...
```

That warning is expected and correct on Hailo-8L.

On a **Hailo-8**, two outcomes are both healthy:

- `… | match` — the HEF agrees with the constants; identical behaviour to before.
- `not exposed by this HailoRT — using config fallback` — this runtime does not report quantization info, so the constants are used. Also identical behaviour to before, and not a problem.

`MISMATCH` is the only one that warrants attention on a Hailo-8. It means the HEF-reading logic is at fault rather than your hardware — set `CLIP_QUANT_SOURCE=config` to restore the previous behaviour and please report it.

##### Still manual: SCRFD output layer names

`ScrfdConfig.output_layers` hardcodes `scrfd_2_5g/conv42`, `conv43`, `conv49`, `conv50`, `conv55`, `conv56`. These are **not** inferred automatically.

If the 8L build names its layers differently, `decode_scrfd` finds no match, logs a warning, and returns an empty list — so **face detection quietly finds zero faces in every image**. It does not crash, and no request fails. Like the quantization case it is quiet, but unlike it there is an explicit log line naming the problem, and the symptom (no faces, ever) is unmistakable once you look:

```
WARNING SCRFD decode: no matching output layers found in ['scrfd_2_5g/conv42', ...]
```

Checking and fixing the names needs a built base image, so the procedure comes later: see [Step 3b (Hailo-8L only): derive SCRFD layer names](#step-3b-hailo-8l-only-derive-scrfd-layer-names). Finish downloading models first.

##### Status: URLs verified, hardware not

All eight Hailo-8L URLs above returned HTTP 200 when this was written. They are published artifacts on Hailo's S3 bucket and can be moved, renamed, or re-versioned without notice — if one 404s, check the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/tree/master/docs/public_models) for the current path.

The models have **not been run on Hailo-8L hardware by the maintainer** — this project is developed and tested on a Hailo-8. This path is documented on the strength of the artifacts existing, not on an end-to-end run. If you try it, reports of what worked and what needed changing are welcome.

### Step 2: Download Supporting Files

**CLIP BPE Tokenizer Vocabulary** (TinyCLIP only, from [OpenAI CLIP](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/bpe_simple_vocab_16e6.txt.gz)):
```bash
curl -Lo models/bpe_simple_vocab_16e6.txt.gz \
  https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
```

**SentencePiece Tokenizer Model** (SigLIP only, from [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224)):
```bash
curl -Lo models/spiece.model \
  https://huggingface.co/google/siglip-base-patch16-224/resolve/main/spiece.model
```

**OCR Character Dictionary** (from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/ppocrv5_dict.txt) — 18,383 characters covering CJK, Latin, Cyrillic, symbols, and emoji):
```bash
curl -Lo models/ppocrv5_dict.txt \
  https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv5_dict.txt
```

### Step 3: Build Docker Images

Both builds **require** `--build-arg HAILORT_VERSION`. Neither Dockerfile declares a default, so omitting it fails the build rather than quietly producing an image that carries one version in its tag and a different one inside.

```bash
# Your host driver's version — see Prerequisites
export HAILORT_VERSION=4.24.0

# Build base image (HailoRT + Python deps)
docker build --build-arg HAILORT_VERSION="$HAILORT_VERSION" \
  -t "hailo-base:v$HAILORT_VERSION" -f Dockerfile.hailo-base .

# Build application image (FROM hailo-base:v$HAILORT_VERSION)
docker build --build-arg HAILORT_VERSION="$HAILORT_VERSION" \
  -t "immich-ml-hailo:v$HAILORT_VERSION" -f Dockerfile.immich-ml-hailo .
```

> **Architecture.** `setup.sh` detects the host architecture and passes the matching `DEB_ARCH`/`WHL_ARCH` automatically, so building natively on the target machine — x86_64 or ARM64 — needs nothing extra.
>
> **Cross-building needs more than those two build args.** `Dockerfile.hailo-base` is `FROM ubuntu:24.04` with no platform pinning, so the image is built for whatever architecture the builder runs. On an x86 host, `--build-arg DEB_ARCH=arm64 --build-arg WHL_ARCH=aarch64` only changes *which HailoRT packages get installed* — the result is ARM64 packages inside an amd64 image, which fails at runtime. A genuine cross-build additionally requires `docker buildx` with `--platform linux/arm64` for **both** images, plus emulation or a native ARM builder. That path is not exercised by this project; building natively on the target host is the supported route.

If the base build fails at the `COPY` step, the requested version's `.deb`/`.whl` are not in `hailo-rt-4/` under the expected names — that check is deliberate, so you find out before installing the wrong runtime.

### Step 3b (Hailo-8L only): derive SCRFD layer names

Skip this on Hailo-8 — the shipped values are correct there.

`ScrfdConfig.output_layers` in `ml_target/config.py` names six layers from the Hailo-8 SCRFD build. If your 8L HEF names them differently, **face detection returns zero faces for every image** — nothing crashes and no request fails, so watch for the symptom described below rather than an error (see [Hailo-8L model substitution](#hailo-8l-model-substitution)).

**`ml_target/` is copied into the app image at build time, so editing `config.py` requires rebuilding the app image.** Do this now, before the app image matters — inspect first, edit, then build.

Inspect the HEF using the **base** image from Step 3, with the repo bind-mounted. No app image and no running container are needed:

```bash
docker run --rm \
  --device=/dev/hailo0:/dev/hailo0 \
  -v "$PWD/ml_target:/app/ml_target:ro" \
  -v "$PWD/models:/app/models:ro" \
  -w /app -e PYTHONPATH=/app \
  "hailo-base:v$HAILORT_VERSION" \
  python3 -m ml_target.hef_inspect /app/models/scrfd_2.5g.hef
```

The `=== OUTPUTS ===` block lists **nine** streams, not six: each of three strides has a class map, a bbox map, and a keypoint map. This pipeline uses the first two and ignores keypoints. Identify them by **shape** — that is authoritative, names vary between builds.

Channel count says what a stream is:

| Channels | Stream | Used |
|---|---|---|
| 2 | class/score map (2 anchors) | yes → `cls_layer_name` |
| 8 | bbox map (2 anchors × 4 coords) | yes → `box_layer_name` |
| 20 | keypoint/landmark map | no — ignore |

Spatial size says which stride it belongs to, for this project's 640×640 input:

| Shape | Stride |
|---|---|
| 80×80 | 8 |
| 40×40 | 16 |
| 20×20 | 32 |

So a stream printed as `shape=(80, 80, 8)` is the stride-8 bbox map. Fill in `ScrfdConfig.output_layers` in `ml_target/config.py` on the host as three `(stride, cls_layer_name, box_layer_name)` entries, for strides 8, 16 and 32 — six of the nine names, keypoints discarded.

As a cross-check: on the Hailo-8 build the nine outputs fall into consecutive triples per stride — `(conv42, conv43, conv44)`, `(conv49, conv50, conv51)`, `(conv55, conv56, conv57)` — each triple being `(cls, bbox, keypoints)`, in stride order 8, 16, 32. That is where the defaults `conv42`/`conv43`, `conv49`/`conv50`, `conv55`/`conv56` come from. **Use this only as a hint — the shapes are authoritative, and another build may name or order its layers differently.**

Then rebuild the app image so the edit is in it — the second `docker build` from Step 3:

```bash
docker build --build-arg HAILORT_VERSION="$HAILORT_VERSION" \
  -t "immich-ml-hailo:v$HAILORT_VERSION" -f Dockerfile.immich-ml-hailo .
```

**How to tell whether you got it right.** Wrong names do not raise — the request succeeds and simply reports no faces. Run the face-detection test from [Testing](#testing) and check the container log:

- `SCRFD decode: no matching output layers found in [...]` — none of your names matched; face detection returns zero faces for every image.
- Faces found at some scales but large or small ones consistently missed — only some entries matched; each unmatched stride is silently skipped.
- `ValueError: Unexpected bbox channels for SCRFD: ...` — a name matched but points at the wrong tensor, most likely `cls` and `box` swapped in an entry.
- No warnings and faces detected — correct.

The same command inspects any other model, and also prints the quantization parameters the pipeline will read at startup:

```bash
docker run --rm --device=/dev/hailo0:/dev/hailo0 \
  -v "$PWD/ml_target:/app/ml_target:ro" -v "$PWD/models:/app/models:ro" \
  -w /app -e PYTHONPATH=/app "hailo-base:v$HAILORT_VERSION" \
  python3 -m ml_target.hef_inspect /app/models/siglip_b_16_image_encoder.hef
```

Once the container is running, the same tool is available inside it — handy for later checks, though remember that any `config.py` edit still needs an image rebuild:

```bash
docker exec immich-ml-hailo python3 -m ml_target.hef_inspect /app/models/scrfd_2.5g.hef
```

### Step 4: Extract CLIP Text Weights

The CLIP text encoder needs CPU-side embedding weights extracted from the original model. Both scripts run the extraction inside the `hailo-base` image you built in Step 3, so they need the same `HAILORT_VERSION` — pass it inline rather than relying on an `export` from an earlier step, which does not survive a new shell. Both scripts require it and abort if it is missing.

**TinyCLIP:**
```bash
HAILORT_VERSION=4.24.0 ./scripts/extract_tinyclip_weights.sh
# Downloads TinyCLIP checkpoint (~330MB), saves models/tinyclip_text_weights.npz
```

**SigLIP:**
```bash
HAILORT_VERSION=4.24.0 ./scripts/extract_siglip_weights.sh
# Downloads SigLIP model (~813MB), saves models/siglip_text_weights.npz + models/spiece.model
```

Only needs to be done once per backend.

## Running

Start the container, passing through the Hailo device:

```bash
docker run -d \
  --device=/dev/hailo0:/dev/hailo0 \
  --group-add=0 \
  --publish 3003:3003 \
  -e CLIP_BACKEND=siglip \
  --name immich-ml-hailo \
  --restart unless-stopped \
  immich-ml-hailo:v4.24.0     # the tag you built — must match your host driver
```

Set `CLIP_BACKEND` to `siglip` or `tinyclip` (see [CLIP Backend Choice](#clip-backend-choice) for details). Both are included in the image — change the value and restart the container to switch, no rebuild needed.

> **Note on `--group-add=0`:** This grants the container process access to the root group (GID 0), which typically owns `/dev/hailo0`. It may not be required on all systems (e.g., Unraid works without it), but is safe to include.

## Immich Configuration

In the Immich **Admin Settings → Machine Learning**:

**Required:**
- Set **Machine Learning URL** to `http://<hailo-host-ip>:3003`

**Model names — leave as default:**

The model name dropdowns (CLIP model, Facial recognition model, OCR model) can be left at their defaults. This worker ignores the model names — it always uses the Hailo-accelerated models regardless of what's selected. The names are sent with each request but have no effect.

**Score thresholds — these work normally:**

All threshold settings (minimum detection score, maximum recognition distance, minimum recognized faces, OCR confidence scores, etc.) are sent with each request and respected by this worker. Adjust them as you normally would.

**CLIP backend (`CLIP_BACKEND` env var):**

Both CLIP backends are included in every Docker image. You switch between them by setting the `CLIP_BACKEND` environment variable at container startup — no rebuild needed:

Add one of these flags to the full `docker run` command in [Running](#running), then restart the container:

- `-e CLIP_BACKEND=siglip` — better quality, Immich-compatible embeddings
- `-e CLIP_BACKEND=tinyclip` — faster, and the default when the variable is unset

- **SigLIP** (`CLIP_BACKEND=siglip`): Embeddings are compatible with Immich's `ViT-B-16-SigLIP__webli`. You can switch between this Hailo worker and the official Immich ML worker (with the same CLIP model selected in Immich) **without re-running Smart Search**.
- **TinyCLIP** (`CLIP_BACKEND=tinyclip`): Embeddings are not compatible with any of the CLIP models Immich offered at the time of writing (`ViT-SO400M-16-SigLIP2-384__webli`, `ViT-B-16-SigLIP2__webli`, `ViT-B-16-SigLIP__webli`, `ViT-B-32__laion2b-s34b-b79k`). Immich's model list changes between releases — check yours before relying on this. Switching to/from the official ML worker requires re-running Smart Search.

> **Note:** Changing `CLIP_BACKEND` between TinyCLIP and SigLIP also requires re-running Smart Search, since the embedding dimensions differ (512 vs 768).

## Testing

Run the test suite inside the container:

```bash
# Copy test script and image into the container
docker cp tests/test.sh immich-ml-hailo:/tmp/test.sh
docker cp tests/test.jpg immich-ml-hailo:/tmp/test.jpg

# Run tests
docker exec immich-ml-hailo bash /tmp/test.sh /tmp/test.jpg
```

The test suite validates all endpoints and inference pipelines — 19 assertions when OCR is available, 18 when it is not (the two OCR checks collapse into a single skip).

It reads `CLIP_BACKEND` from the container's environment and asserts the exact embedding dimension that backend must produce — 512 for TinyCLIP, 768 for SigLIP — so a SigLIP container that silently fell back to TinyCLIP fails the suite instead of passing it. The resolved backend is printed in the test header.

The suite targets `http://localhost:3003` by default. Override with `BASE_URL` to test a remapped port or a service on another host — useful when running the script from your workstation rather than inside the container:

```bash
BASE_URL=http://192.168.1.50:3003 ./tests/test.sh tests/test.jpg
```

### Golden embedding test

`test.sh` asserts embedding *dimensions*. `tests/golden.sh` asserts embedding *values*, against a reference captured from a build you trust.

This exists because the dangerous failure mode is a **correctly-shaped, correctly-normalised, subtly-wrong vector** — wrong quantization parameters, a mis-shaped batch, a silently substituted model. Every one of `test.sh`'s assertions passes in that case. Only comparing against a known-good reference catches it. Run it before and after any change to how tensors reach the device.

Run it on the Docker host, against a running container:

```bash
# Capture references from a build you trust (do this once, deliberately)
./tests/golden.sh generate

# Verify nothing has drifted
./tests/golden.sh check
```

`generate` runs the same image ten times first, reports the observed run-to-run cosine spread, and sets the pass threshold at ten times that measured noise floor (never tighter than `1e-5`). The measurement is printed so the number is auditable:

```
  measured run-to-run cosine similarity:
    clip_visual    min=1.000000000000  dim=768  n=10
    clip_textual   min=1.000000000000  dim=768  n=10
    face           min=1.000000000000  dim=512  n=10

  Device is bit-exact across repeats (all similarities == 1.0).
  Threshold set to 0.999990000 (margin 1.000e-05 = 10x observed, floor 1e-5)
```

It covers CLIP visual, CLIP textual, and the ArcFace face embedding, plus the detected face count.

> **⚠️ References are not portable, and must be regenerated deliberately.**
>
> They pin the numeric output of one HEF build on one device. A Hailo-8 reference will not match a Hailo-8L, and a Model Zoo version bump recompiles the HEF and moves the values. **Regenerate whenever the model, the HEF version, or the device changes** — and never merely to make the test pass, which discards the only signal you have.
>
> A stale reference fails with a large similarity drop that looks exactly like a regression. The checker warns when the container image tag or the test image has changed since the reference was captured, but it cannot detect every case.

References live in `tests/golden/` (gitignored, one file per CLIP-backend / detector / recognizer combination — e.g. `siglip__scrfd_2.5g__arcface_r50.json`) and are **not** shipped in the repo — generate them on your own deployment. `check` skips cleanly with instructions when no reference exists, so it is safe to run on a fresh install. Run `generate` once per combination you use — changing `CLIP_BACKEND`, `FACE_DETECTOR` or `FACE_RECOGNIZER` needs its own reference, and an unseen combination skips rather than failing.

Environment: `BASE_URL`, `CONTAINER`, `GOLDEN_DIR`, and `SAMPLES` (repeats used to measure the noise floor, default 10).

### Benchmarking

`scripts/benchmark.sh` reports **per-stage** p50/p95 latency, so a change can be attributed to a specific pipeline stage rather than to an end-to-end total:

```bash
./scripts/benchmark.sh                      # tests/test.jpg, 20 iterations
./scripts/benchmark.sh /path/to/img.jpg 50  # custom image, 50 iterations
```

It fires requests for all four task shapes (CLIP visual, CLIP textual, facial recognition, OCR), then reads the timings back out of the worker's per-request summary lines via `docker logs`. That means it needs no code change and no rebuild — it measures the container you already have running. It is read-only: it sends inference requests and reads logs, and never restarts or reconfigures anything.

Run it on the Docker host, with `LOG_LEVEL` at `INFO` or `DEBUG` (it aborts otherwise, since a higher level suppresses the lines it parses).

**Pause Immich's ML jobs first.** Concurrent traffic both competes for the device and writes into the same log; the script warns when it sees more matching lines than it sent, which means the numbers are contaminated.

To compare two runs, keep the image, iteration count, CLIP backend, face detector and face recognizer identical — face and OCR timings scale with how many faces and text regions the image happens to contain.

**Record the batch settings and the detector with every run.** `HAILO_BATCH_SIZE_FACE` and `HAILO_BATCH_SIZE_OCR` change `rec_infer_batch` and `ocr_rec_batch` by a factor of several, so two runs are only comparable if both are known. The script prints the container's resolved image tag and backend but cannot see these, since they take effect at model-configure time — note them yourself alongside the results.

This is a tool you reach for deliberately. It is not part of `setup.sh`.

See **[BENCHMARKS.md](BENCHMARKS.md)** for every measurement taken on this hardware — device benchmarks, per-stage timings, the batch-size sweep, concurrency curves — and the exact commands to reproduce each one.

### Concurrency benchmark

`scripts/benchmark_concurrency.sh` answers a different question: **does firing requests in parallel improve throughput?**

```bash
./scripts/benchmark_concurrency.sh                      # clip, 20 requests, C=1,2,4,8
./scripts/benchmark_concurrency.sh face                 # facial-recognition
./scripts/benchmark_concurrency.sh ocr tests/test.jpg 40 "1 2 4 8 16"
```

It sweeps concurrency levels and reports throughput (requests/sec), wall clock, and **both** client-side and server-side latency. The gap between those two is the queue wait, which is what distinguishes "requests are queueing" from "requests are getting slower".

Pick the task deliberately — the three have very different CPU/device ratios, so the shape of the curve differs. CLIP visual is device-dominated (~224 ms device against ~49 ms JPEG decode); facial recognition is CPU-dominated (~17 ms device against the same ~49 ms decode); OCR sits between them.

Before measuring the worker it runs a **control**: the same requests with deliberately invalid `entries`, so the full image upload happens but the request is rejected before any inference. If control throughput does not scale with concurrency, the load generator is the bottleneck rather than the worker, and the script says so and marks the results void. Without that check a flat result is ambiguous.

Same discipline as the latency benchmark: read-only, pause Immich's ML jobs first, and hold the image, request count, backend, detector and recognizer constant between runs. A diagnostic tool, not part of `setup.sh`.

## Debugging

Run on the **host**:

```bash
# Follow the container's logs
docker logs -f immich-ml-hailo

# Open an interactive shell inside the container
docker exec -it immich-ml-hailo /bin/bash
```

### Reading the request log

At the default `INFO` level each `/predict` call emits one summary line:

```
2026-08-06 11:20:04 - INFO - /predict image=1920x1080 tasks=clip backend=siglip clip=visual status=200 decode_image=18.4ms clip_preprocess=2.9ms clip_image_infer=71.2ms total=94.1ms
2026-08-06 11:20:05 - INFO - /predict image=1920x1080 tasks=facial-recognition faces=3 status=200 decode_image=17.9ms letterbox=11.6ms det_infer=24.8ms det_decode=6.1ms crop_faces=1.4ms rec_infer_batch=19.7ms total=82.6ms
2026-08-06 11:20:07 - INFO - /predict image=1920x1080 tasks=ocr ocr=12/15 status=200 decode_image=18.1ms ocr_letterbox=14.2ms ocr_det_infer=38.5ms ocr_det_decode=9.3ms ocr_crop=3.7ms ocr_rec_batch=64.9ms total=149.8ms
```

Facts come first, then per-stage timings, then the end-to-end total. `faces=` is the detection count; `ocr=12/15` means 12 of 15 detected regions passed the recognition score threshold. Stage names map directly onto pipeline steps, so a slow stage is immediately attributable. Every request produces one such summary line, including failures — a request that fails outright carries `status=400`/`status=500` and an `error=` tag.

A request can produce **additional** `WARNING`/`ERROR` lines alongside its summary — a detection cap triggering, an unknown task, SCRFD layer names not matching, an unusable OCR output shape, or a malformed request. Those are deliberately not suppressed at `INFO`: they are the lines you need. Note also that per-task errors returned inside a `200` response (missing image or text, OCR models unavailable) are reported in the response body and do not add an `error=` tag to the summary.

Set `LOG_LEVEL=DEBUG` for the underlying per-call detail (letterbox geometry, SCRFD decode parameters, per-timer lines, the full `entries` payload). That is useful for one-off diagnosis and far too verbose for a library scan — a 50,000-asset scan at `INFO` writes roughly 50,000 lines rather than 500,000, which matters on Unraid where `docker.img` space is finite.

Run **inside the container** — either from that shell, or prefixed with `docker exec immich-ml-hailo`:

```bash
# Inspect one HEF's inputs/outputs (shapes, formats, quantization params)
python3 -m ml_target.hef_inspect /app/models/scrfd_2.5g.hef

# Inspect every HEF in models/
python3 -m ml_target.inspect_models
```

## Project Structure

The repository is 28 files and a few hundred kilobytes. **`models/` and `hailo-rt-4/` ship containing nothing but a `.gitkeep`** — every model, weight file, and runtime package is downloaded from its original source or generated on your machine at setup time. None of it is redistributed here, which is both why the repo is small and why `setup.sh` exists.

```
.dockerignore             # Shared build context exclusions for both Dockerfiles
.gitignore
Dockerfile.hailo-base     # Base image: Ubuntu 24.04 + HailoRT
Dockerfile.immich-ml-hailo # App image: FastAPI + models + inference code
LICENSE                   # MIT
MODELS.md                 # Model choices, evaluated alternatives, upgrade paths
BENCHMARKS.md             # Measured numbers and how to reproduce them
README.md
setup.sh                  # Full setup: check prereqs, download models, build, test
hailo-rt-4/
  .gitkeep                # Placeholder — you download the HailoRT .deb + .whl here
models/
  .gitkeep                # Placeholder — setup.sh downloads the HEFs and dictionaries,
                          #   the extract scripts generate the .npz weights
ml_target/                # Application code
  __init__.py
  app.py                  # FastAPI endpoints: GET /, GET /ping, POST /predict
  config.py               # All model-specific configuration (paths, layer names, quant params)
  pipeline.py             # Pipeline initialization and inference orchestration
  models.py               # Hailo model wrapper, activation, inference helpers
  preprocessing.py        # Image transforms, CLIP preprocessing, L2 normalize
  decoders.py             # SCRFD face detection post-processing + NMS
  ocr.py                  # PaddleOCR DBNet post-processing + CTC decode
  tokenizer.py            # CLIP BPE + SigLIP SentencePiece tokenizers
  hailo_backend.py        # Back-compat shim; re-exports init_pipeline / run_inference
  hef_inspect.py          # Utility: print one HEF's input/output stream info
  inspect_models.py       # Utility: print stream info for every HEF in models/
scripts/
  extract_tinyclip_weights.sh  # Generate tinyclip_text_weights.npz from checkpoint
  extract_siglip_weights.sh    # Generate siglip_text_weights.npz + spiece.model
  benchmark.sh                 # Per-stage p50/p95 latency against a running container
  benchmark_concurrency.sh     # Throughput vs concurrency sweep, with a load-generator control
tests/
  test.sh                 # End-to-end test suite (19 assertions / 18 without OCR)
  golden.sh               # Golden-embedding regression test (generate | check)
  test.jpg                # Sample test image
  ocr-align.jpg           # 64-cell position-encoding grid for OCR alignment checks
  ocr-align.svg           # Source the grid was rendered from
  verify_ocr_align.py     # Asserts text[i] names the cell box[i] sits in
  golden/                 # Generated references — gitignored, per device, backend, detector and recognizer
```

## Configuration

All model parameters are in `ml_target/config.py`. To swap models (e.g., SCRFD 2.5G → SCRFD 10G), update the config dataclass — no inference code changes needed. See the docstrings in `config.py` for available alternatives.

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `CLIP_BACKEND` | `tinyclip` | `tinyclip` or `siglip` — see [CLIP Backend Choice](#clip-backend-choice) |
| `MODELS_DIR` | `/app/models` | Where the pipeline looks for HEFs and supporting files |
| `CLIP_QUANT_SOURCE` | `hef` | `hef` reads CLIP quantization parameters from the loaded HEF, falling back to the `config.py` constants when the runtime does not expose them. `config` forces the constants. Only change this if the startup log reports a quantization mismatch on hardware you know was working. |
| `LOG_LEVEL` | `INFO` | Logging verbosity — `DEBUG`, `INFO`, `WARNING`, `ERROR`. At `INFO` each request emits one summary line; `DEBUG` adds the per-stage detail. Read at startup, so changing it needs a container restart but no rebuild. Case-insensitive and whitespace-tolerant. An unrecognized non-empty value falls back to `INFO` and logs a warning; setting it to an empty value is treated as unset and falls back silently. |
| `HAILO_BATCH_SIZE_FACE` | `8` | Device batch size for ArcFace face recognition. `default` disables device batching on this path. |
| `HAILO_BATCH_SIZE_OCR` | `8` | Device batch size for OCR text recognition. `default` disables device batching on this path. |
| `HAILO_BATCH_SIZE` | `8` | Fallback for both of the above. A per-path variable always wins — including when it is set to `default`. |
| `FACE_DETECTOR` | `scrfd_2.5g` | Face detection model — `scrfd_2.5g` or `scrfd_10g`. See [Face detector](#face-detector). An unrecognised value fails at startup rather than falling back. |
| `REQUEST_MODE` | `serial` | `serial` runs one request at a time; `threadpool` overlaps host CPU work with device time. See [Request handling](#request-handling). Opt-in — reverting is a restart. |
| `REQUEST_THREADS` | `4` | Worker threads when `REQUEST_MODE=threadpool`. Measured optimum; 8 buys +3% for double the peak memory. |
| `FACE_RECOGNIZER` | `arcface_r50` | Face recognition model — `arcface_r50` or `arcface_mobilefacenet`. **Changing this forces Immich to re-run its face jobs.** See [Face recognition model](#face-recognition-model). An unrecognised value fails at startup. |

All three are read at startup, so **changing them is a container restart, not a rebuild.**

### Device batching

`8` is the measured optimum on a Hailo-8, not a guess — see [MODELS.md](MODELS.md) for the measurements and the cost model behind them. In short: batching face recognition at 8 is **2.2× faster** across a typical mix of photos, because the dominant cost is a fixed per-burst overhead that batching amortises, not per-frame compute.

It applies only to the two paths that ever receive more than one frame — face recognition and OCR recognition. CLIP and both detection models are sent exactly one frame per request and are left at HailoRT's default.

Frames are padded up to a whole multiple of the batch size, because this device runs multi-context HEFs without the model scheduler and HailoRT rejects any other frame count:

```
CHECK failed - On the case of multi-context without the model scheduler,
frames count must be a multiplier of the batch size! (5 % 8 != 0)
```

Padding is cheap here precisely because the overhead is per burst — a burst costs about the same whether it carries one real frame or eight. Padded rows are discarded before any result is assembled.

The one case where batching loses is a photo with exactly **one** face: 28.8 ms → 35.2 ms. That is roughly 3% of a request whose total is ~205 ms, and it is repaid from two faces upward. If your library is overwhelmingly single-face portraits, `HAILO_BATCH_SIZE_FACE=default` reverts that path — but measure before assuming it helps.

`ScrfdConfig.max_faces` (default 100) bounds the recognition batch: at batch 8 that is at most 13 bursts for a single image.

### Request handling

By default the worker handles **one request at a time**. `REQUEST_MODE=threadpool` lets it overlap the host-side work of one request with the device time of another.

**What can and cannot overlap.** There is one accelerator, so device time can never overlap device time — a single global lock serialises every inference call. What overlaps is everything else: JPEG decode (which dominates, ~49 ms for a 2360×2360 image), letterboxing, SCRFD decode and NMS, crop extraction, and embedding serialisation. That is the entire win, and it is bounded by how much host work a request does relative to its device work.

Measured on a Hailo-8 (HailoRT 4.24.0, SigLIP, 2360×2360 image, 20 requests per level):

| Configuration | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|
| face, `serial` | 9.03 | — | — | 9.73 RPS *(1.08×)* |
| face, `threadpool`, `arcface_r50` | 9.88 | — | — | 15.59 RPS *(1.58×)* |
| face, `threadpool`, `arcface_mobilefacenet` | 12.67 | 20.86 | 26.53 | **28.37 RPS** *(2.24×)* |
| clip, `threadpool` | 3.53 | — | — | 4.12 RPS *(1.17×)* |

CLIP gains little because it is device-bound — 224 ms of its 275 ms is the accelerator, so there is almost nothing to hide. Face gains most, and gains more with a faster recognition model, because shrinking device time raises the ceiling on what pipelining can achieve.

`REQUEST_THREADS=4` is the measured optimum: 8 threads yields 29.28 RPS against 4 threads' 28.37 — **+3% for double the peak memory**. The limit is host-side serialisation rather than thread count; the worker reaches roughly half the device's theoretical ceiling and that is where this design lands.

> **`threadpool` is opt-in for now, on purpose.** Correctness under load has been verified — three separate 64-region OCR alignment checks returned `exact=64 SHIFTED=0` while 60 requests ran at C=8 against a genuinely saturated worker, plus a bit-exact `golden.sh check` and 19/19 on the test suite. But that is a saturated 60-request run, **not a full library scan**, and a rare race would need sustained real-world load to surface. Enable it deliberately, watch the first scan, and remember that reverting is `REQUEST_MODE=serial` and a restart — no rebuild.

#### Immich's side — none of this works without it

**Immich decides how many requests to send.** Smart Search, Face Detection and OCR each have their own **job concurrency** setting under *Administration → Settings → Job Settings*. At the default of **1**, Immich issues one request at a time and waits for the response, so the worker never sees overlapping requests and `REQUEST_MODE=threadpool` changes nothing whatsoever.

**Recommended: 3.**

The reasoning, from the table above: face throughput is 20.86 at C=2, 26.53 at C=4 and 28.37 at C=8 — so it is nearly saturated by C=4, and the last doubling buys only **+7%**. CLIP saturates earlier still. 3 captures the large majority of the available gain.

The tie-breaker for 3 over 4 is that these are **three independent settings**. If more than one job type is draining its queue at once, the worker sees their *sum*, not the largest — three jobs at 4 each could offer 12 concurrent requests against 4 worker threads. Size them together, and keep the total near `REQUEST_THREADS`.

> **Do not set these high.** Past saturation, extra concurrency adds no throughput at all and only deepens the queue. In the measured sweep, client p95 latency rose from **70 ms at C=1 to 289 ms at C=8** while throughput moved 26.53 → 28.37. A value like 8 or 16 buys nothing and makes every request slower.

### Face detector

Two SCRFD variants ship, selected with `FACE_DETECTOR`. Both HEFs are in the image, so switching is a container restart.

| | mAP | Device latency | Share of a 96.8 ms face request |
|---|---|---|---|
| **`scrfd_2.5g`** *(default)* | 76.4 | **2.53 ms** | 2.6% |
| `scrfd_10g` | **82.1** | **4.40 ms** | 4.5% |

Measured with `hailortcli benchmark` on a Hailo-8, HailoRT 4.24.0.

`scrfd_10g` costs **+1.87 ms** — under 2% of a face request — for **+5.7 mAP**, and is materially better on small, occluded and profile faces. `scrfd_2.5g` remains the default because it is the tested configuration and the one existing golden references were generated against.

```bash
docker run -d ... -e FACE_DETECTOR=scrfd_10g ...
```

**It does not change face embeddings by itself.** The recognition model is unchanged, so your existing face clusters are not invalidated and Immich needs no re-index.

**But it will detect more faces**, which is the point of using it — and that is user-visible: Immich will find new faces in already-scanned photos and cluster them, so expect new people to appear and some existing people to gain photos. Re-running Immich's face detection job over the library is what surfaces that.

> **Switching detectors invalidates the golden face reference.** A different detector finds different faces at slightly different boxes, so the crops differ and so do the ArcFace embeddings — the face count may change too. References are keyed by CLIP backend *and* detector, so an unseen combination **skips** with an explicit message rather than failing. Run `./tests/golden.sh generate` after switching. See [Golden embedding test](#golden-embedding-test).

If the selected HEF is missing, the worker **refuses to start** and names the file and its download URL. Face detection is a core task; a worker running without it would answer every request with zero faces and no error.

### Face recognition model

Two recognition models ship, selected with `FACE_RECOGNIZER`. Both HEFs are in the image.

**Read the trade before the speed.** This is face *identity*: the embedding is what Immich clusters people by.

| | LFW accuracy | Device latency |
|---|---|---|
| **`arcface_r50`** *(default)* | **99.7%** | 20.36 ms |
| `arcface_mobilefacenet` | 99.4% | **1.09 ms** |

0.3 percentage points sounds negligible and is not. The failure mode is visible and irritating: two people merged into one cluster, or one person split across two, needing manual correction in the Immich UI. A slower first scan is forgotten in a week; a mis-clustered family album is not. **`arcface_r50` is the default and should stay it for most people.**

Against that, the speed difference is genuinely large — 19× lower latency, measured with `hailortcli` on a Hailo-8. In the pipeline, `rec_infer_batch` is 35.2 ms for 5 faces; mobilefacenet should bring that to a few milliseconds, and combined with `FACE_DETECTOR=scrfd_10g` a face request could fall from ~101 ms toward ~65 ms.

That is worth considering if you are indexing a very large library from scratch and accept the accuracy trade knowingly.

```bash
docker run -d ... -e FACE_RECOGNIZER=arcface_mobilefacenet ...
```

> **⚠️ Switching recognition models forces a re-index of faces.**
>
> Unlike the detector, this changes the face **embeddings themselves**. Every vector already stored in Immich becomes incomparable with newly produced ones, so **Immich must re-run its face jobs across the whole library**: clusters are rebuilt from scratch and every named person has to be reconfirmed by hand. On a large library that is hours of processing plus real manual work.
>
> This is why it is an option and not a default — not because the speed is unwelcome, but because no one should have their people-tagging reset by pulling an update.

Golden references are keyed by recognition model too, so switching makes `./tests/golden.sh check` skip rather than fail. Regenerate after switching.

If the selected HEF is missing, the worker refuses to start and names the file — the same rule as the detector.

### Detection caps

`ScrfdConfig` in `config.py` bounds how much work one image can create:

| Setting | Default | Purpose |
|---|---|---|
| `max_pre_nms` | `1000` | Highest-scoring candidate boxes kept before NMS |
| `max_faces` | `100` | Faces returned after NMS |

SCRFD emits 16,800 candidate boxes for a 640×640 input. Immich's **minimum detection score** is an admin-facing slider: at its default of 0.7 almost none survive, but at a low value thousands can — and every survivor costs an NMS iteration and then a 112×112 crop in a single stacked recognition batch. These caps keep that bounded.

Neither cap has any effect at normal thresholds. When one does truncate it logs at `WARNING` with the counts, so it is never silent:

```
WARNING SCRFD decode: 16626 candidates above score_thr=0.100 exceeds max_pre_nms=1000 —
        keeping the 1000 highest-scoring. ...
WARNING SCRFD decode: 625 faces after NMS exceeds max_faces=100 —
        returning the 100 highest-scoring. ...
```

If you see these, the usual cause is a minimum detection score set too low in Immich. Raise it, or raise the cap in `config.py` if the image genuinely contains that many faces.

`OcrRecognitionConfig.rec_batch_size` (default `32`) controls how many detected text regions are recognized per device round-trip.

## License

This project is licensed under the [MIT License](LICENSE).
