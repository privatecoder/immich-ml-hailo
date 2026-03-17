# Immich ML Worker for Hailo-8 PCIe Accelerators

An external ML inference worker for [Immich](https://immich.app/) that offloads **face detection/recognition**, **CLIP smart search**, and **OCR** to a **Hailo-8** PCIe accelerator. It replaces Immich's built-in ONNX-based ML worker with a FastAPI service that speaks the same `/predict` protocol — but runs inference on the Hailo-8 hardware at a fraction of the power.

## Immich Jobs Handled by This Worker

This worker accelerates the following Immich jobs on the Hailo-8:

| Immich Job | Hailo Model | Notes |
|------------|-------------|-------|
| **Smart Search** | TinyCLIP ViT-39M/16 **or** SigLIP B/16 | CLIP image embeddings for semantic search |
| **Duplicate Detection** | (uses Smart Search embeddings) | No separate inference — reuses CLIP embeddings |
| **Face Detection** | SCRFD 2.5G | Detects faces in images |
| **Facial Recognition** | ArcFace R50 | Generates face embeddings for grouping people |
| **OCR** | PaddleOCR v5 mobile | Extracts text from images |

Other Immich jobs (Generate Thumbnails, Extract Metadata, Transcode Videos, Sidecar Metadata, External Libraries, Storage Template Migration) run on the Immich server itself and are not affected by this worker.

### CLIP Backend Choice

Two CLIP backends are available, selectable via the `CLIP_BACKEND` environment variable:

|  | TinyCLIP (default) | SigLIP |
|--|-------------------|--------|
| Image input | 224x224 (center-crop) | 224x224 (squash resize) |
| Embedding dim | 512 | 768 |
| Image FPS | ~60 | ~14 |
| Text FPS | ~18 | ~17 |
| Search quality | Good | Better |
| Immich model match | None | `ViT-B-16-SigLIP__webli` |

**TinyCLIP** is significantly faster (~4x for images) but produces embeddings incompatible with any Immich default model.

**SigLIP** produces the same embeddings as Immich's `ViT-B-16-SigLIP__webli` (same underlying Google model weights). This means you can switch between this Hailo worker and the official Immich ML worker **without re-running Smart Search** — the embeddings are compatible. The text encoder output is also simpler: already pooled to a single vector (no CPU-side projection needed).

Both backends output UINT16 from the Hailo device and are dequantized to float32 before L2 normalization.

## Prerequisites

- **Hailo-8 or Hailo-8L** M.2 PCIe accelerator
- **Host Hailo drivers** installed and working (HailoRT v4.23.0). For Unraid, use the `Hailo RT Driver` app by ich777. See [hailort-drivers](https://github.com/hailo-ai/hailort-drivers) (v4 branch for Hailo-8/8L, v5 for Hailo-10H/15H).
- **Docker** on the host

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

Download the two files for your platform and place them in `hailo-rt-4/`:

**x86_64:**
- _HailoRT – Python package (whl) for Python 3.12, x86_64_ → `hailort-4.23.0-cp312-cp312-linux_x86_64.whl`
- _HailoRT – Ubuntu package (deb) for amd64_ → `hailort_4.23.0_amd64.deb`

**ARM64 (aarch64):**
- _HailoRT – Python package (whl) for Python 3.12, aarch64_ → `hailort-4.23.0-cp312-cp312-linux_aarch64.whl`
- _HailoRT – Ubuntu package (deb) for arm64_ → `hailort_4.23.0_arm64.deb`

> **Why Python 3.12?** The Docker base image uses Ubuntu 24.04 LTS, which ships Python 3.12 as the system default. HailoRT 4.23.0 supports Python 3.10, 3.11, and 3.12 — using the system Python avoids managing a venv, and 3.12 has the best performance (~5% faster runtime than 3.11).

## Quick Setup

Once the HailoRT packages are in `hailo-rt-4/`, run:

```bash
./setup.sh
```

This will check for required files, offer to download missing models, build both Docker images, extract TinyCLIP weights, and run the test suite. See the [Manual Setup](#manual-setup) section below if you prefer to do each step yourself.

## Manual Setup

### Step 1: Download HEF Models (skip if you used Quick Setup)

Download the pre-compiled `.hef` model files from the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/tree/master/docs/public_models) and place them in `models/`.

These links are for **Hailo-8** accelerators and may not work on **Hailo-8L** cards.

**Face Detection** ([model card](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_face_detection.rst)):
```bash
curl -Lo models/scrfd_2.5g.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/scrfd_2.5g.hef
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

```bash
# Build base image (HailoRT + Python deps)
docker build -t hailo-base:v4.23.0 -f Dockerfile.hailo-base .

# Build application image
docker build -t immich-ml-hailo:v4.23.0 -f Dockerfile.immich-ml-hailo .
```

> On ARM64, the base image build automatically detects the platform. If you need to cross-build, pass `--build-arg DEB_ARCH=arm64 --build-arg WHL_ARCH=aarch64`.

### Step 4: Extract CLIP Text Weights

The CLIP text encoder needs CPU-side embedding weights extracted from the original model. Run the script for your chosen backend:

**TinyCLIP:**
```bash
./scripts/extract_tinyclip_weights.sh
# Downloads TinyCLIP checkpoint (~330MB), saves models/tinyclip_text_weights.npz
```

**SigLIP:**
```bash
./scripts/extract_siglip_weights.sh
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
  immich-ml-hailo:v4.23.0
```

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

```bash
# Use SigLIP (better quality, Immich-compatible)
docker run -e CLIP_BACKEND=siglip ...

# Use TinyCLIP (faster, default)
docker run -e CLIP_BACKEND=tinyclip ...
```

- **SigLIP** (`CLIP_BACKEND=siglip`): Embeddings are compatible with Immich's `ViT-B-16-SigLIP__webli`. You can switch between this Hailo worker and the official Immich ML worker (with the same CLIP model selected in Immich) **without re-running Smart Search**.
- **TinyCLIP** (`CLIP_BACKEND=tinyclip`): Embeddings are not compatible with any of Immich's available CLIP models (`ViT-SO400M-16-SigLIP2-384__webli`, `ViT-B-16-SigLIP2__webli`, `ViT-B-16-SigLIP__webli`, `ViT-B-32__laion2b-s34b-b79k`). Switching to/from the official ML worker requires re-running Smart Search.

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

The test suite validates all endpoints and inference pipelines (19 assertions).

## Debugging

```bash
# Interactive shell in a running container
docker exec -it immich-ml-hailo /bin/bash

# Inspect HEF model inputs/outputs
python3 -m ml_target.hef_inspect /app/models/scrfd_2.5g.hef

# Inspect all models
python3 -m ml_target.inspect_models

# View container logs
docker logs -f immich-ml-hailo
```

## Project Structure

```
hailo-rt-4/               # HailoRT .deb and .whl packages (not in repo, download manually)
models/                   # HEF models, weights, and dictionaries
ml_target/                # Application code
  app.py                  # FastAPI endpoints: GET /, GET /ping, POST /predict
  config.py               # All model-specific configuration (paths, layer names, quant params)
  pipeline.py             # Pipeline initialization and inference orchestration
  models.py               # Hailo model wrapper, activation, inference helpers
  preprocessing.py        # Image transforms, CLIP preprocessing, L2 normalize
  decoders.py             # SCRFD face detection post-processing + NMS
  ocr.py                  # PaddleOCR DBNet post-processing + CTC decode
  tokenizer.py            # CLIP BPE tokenizer (stdlib re, no regex dependency)
scripts/
  extract_tinyclip_weights.sh  # Generate tinyclip_text_weights.npz from checkpoint
tests/
  test.sh                 # End-to-end test suite (19 assertions)
  test.jpg                # Sample test image
setup.sh                  # Full setup: check prereqs, download models, build, test
Dockerfile.hailo-base     # Base image: Ubuntu 24.04 + HailoRT
Dockerfile.immich-ml-hailo # App image: FastAPI + models + inference code
```

## Configuration

All model parameters are in `ml_target/config.py`. To swap models (e.g., SCRFD 2.5G → SCRFD 10G), update the config dataclass — no inference code changes needed. See the docstrings in `config.py` for available alternatives.

The models directory can be overridden with the `MODELS_DIR` environment variable.

## License

This project is licensed under the [MIT License](LICENSE).
