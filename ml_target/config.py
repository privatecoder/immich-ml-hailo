"""Centralized model configuration.

All model-specific parameters (paths, layer names, quantization params) live here.
To swap a model, change the config — not the inference code.

CLIP backend is selectable via the CLIP_BACKEND environment variable:
  CLIP_BACKEND=tinyclip  (default) — faster inference, 512-dim embeddings
  CLIP_BACKEND=siglip    — better search quality, 768-dim, Immich-compatible embeddings
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

class ConfigError(RuntimeError):
    """A configuration value is unusable.

    Raised at startup so the container fails visibly instead of running with
    something other than what was asked for.
    """


MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
CLIP_BACKEND = os.environ.get("CLIP_BACKEND", "tinyclip").lower()

# Where CLIP quantization parameters come from.
#   "hef"    (default) — read from the loaded HEF, falling back to the constants
#                        below when the HailoRT version does not expose them
#   "config"           — always use the constants below
# The constants were read from the Hailo-8 builds and are correct only for those.
# Set CLIP_QUANT_SOURCE=config to pin the pre-existing behaviour if HEF reading
# ever misbehaves; see the startup log, which prints both values.
CLIP_QUANT_SOURCE = os.environ.get("CLIP_QUANT_SOURCE", "hef").lower()

# Device-side batch size for the two paths that submit more than one frame:
# ArcFace face recognition and OCR text recognition.
#
# Configuring this makes frame count a HARD CONSTRAINT: this device runs
# multi-context HEFs without the model scheduler, where HailoRT requires the
# frame count to be an exact multiple of batch_size. Short batches are padded
# up in pipeline._pad_to_batch() and the pad rows discarded.
#
# Measured on a Hailo-8 (5 faces, p50 rec_infer_batch):
#
#     unset  144.2 ms   28.84 ms/frame
#     B=2     88.4 ms   14.73 ms/frame    r = 1.96x
#     B=4     61.5 ms    7.69 ms/frame    r = 3.75x
#     B=8     35.2 ms    4.40 ms/frame    r = 6.55x
#
# which fits, to within 0.1% at B=4 and B=8:
#
#     cost = ceil(N/B) x (26.3 ms burst overhead + B x 1.11 ms per frame)
#
# The interesting part is the shape: actual per-face compute is ~1.1 ms, and
# the unbatched path was paying the whole 26.3 ms burst overhead on *every
# frame*. ArcFace R50 is a large multi-context model and each burst cycles its
# contexts once; batching amortises that. This is why padding is cheap in
# practice — a burst costs almost the same whether it carries 1 real frame or 8.
#
# 8 is the measured optimum. B=16 is worse (44.9 vs 36.6 ms weighted) purely
# because single-face photos pad so wastefully. The only case where 8 loses to
# unbatched is N=1 (28.8 -> 35.2 ms), roughly 3% of a request whose total is
# ~205 ms — paid back many times over from N=2 upward.
#
# "default"/"auto"/"0"/"none"/"off" means do not configure it at all: no
# padding, no multiple-of constraint, and the pre-batching behaviour exactly.
# That is NOT the same as 1 — unset leaves HAILO_DEFAULT_BATCH_SIZE
# ("determined by HailoRT automatically"), which measures ~5% slower than the
# model predicts for a true batch of 1.
_BATCH_OFF = ("default", "auto", "", "0", "none", "off")


def _parse_batch_size(raw):
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw in _BATCH_OFF:
        return None
    try:
        v = int(raw)
    except ValueError:
        return None
    return v if v >= 1 else None


def _batch_for(name: str, fallback):
    """Per-path override. Explicitly setting it to an 'off' value wins over the
    shared HAILO_BATCH_SIZE — absence, not falsiness, is what falls through."""
    raw = os.environ.get(name)
    if raw is None:
        return fallback
    return _parse_batch_size(raw)


HAILO_BATCH_SIZE = _parse_batch_size(os.environ.get("HAILO_BATCH_SIZE", "8"))
HAILO_BATCH_SIZE_FACE = _batch_for("HAILO_BATCH_SIZE_FACE", HAILO_BATCH_SIZE)
HAILO_BATCH_SIZE_OCR = _batch_for("HAILO_BATCH_SIZE_OCR", HAILO_BATCH_SIZE)


@dataclass
class ScrfdConfig:
    """Face detection model config for one SCRFD variant.

    Built by scrfd_config_for(); do not instantiate directly unless you are
    adding a variant. See SCRFD_VARIANTS below.
    """
    hef: str = "scrfd_2.5g.hef"
    input_size: int = 640
    # (stride, cls_layer_name, box_layer_name) — must match the compiled HEF
    output_layers: List[Tuple[int, str, str]] = field(default_factory=lambda: [
        (8, "scrfd_2_5g/conv42", "scrfd_2_5g/conv43"),
        (16, "scrfd_2_5g/conv49", "scrfd_2_5g/conv50"),
        (32, "scrfd_2_5g/conv55", "scrfd_2_5g/conv56"),
    ])

    # ── Safety caps ───────────────────────────────────────────────────
    # SCRFD emits 16,800 candidate boxes at 640x640 (80²+40²+20² cells, two
    # anchors each). Immich's `minScore` is an admin-facing slider: at 0.7
    # almost nothing survives, but at 0.1 thousands can, and every survivor
    # costs an NMS iteration and then a 112x112 ArcFace crop in a single batch.
    # These caps bound that. Both truncations log at WARNING with the counts —
    # a silent cap would just move the surprise somewhere harder to find.
    max_pre_nms: int = 1000   # highest-scoring candidates kept before NMS
    max_faces: int = 100      # faces returned after NMS


# ── SCRFD variants ────────────────────────────────────────────────────
#
# Selected by the FACE_DETECTOR env var. Both HEFs ship, so switching is a
# container restart.
#
# **The output layer names are per-variant and are NOT interchangeable.** They
# were read from each HEF with `python3 -m ml_target.hef_inspect`, never
# inferred. Compare the two below: stride 16 happens to be identical
# (conv49/conv50), but strides 8 and 32 are shifted by one. Copying 2.5g's
# names onto 10g would silently match only the stride-16 pair, so the detector
# would find mid-size faces and quietly miss everything large or small —
# decode_scrfd logs a warning only when *nothing* matches, not when some do.
# That is why these live per model rather than in one shared list.
#
# Identifying them yourself: channel count says what a stream is (2 = class,
# 8 = bbox, 20 = keypoints, unused), spatial size says the stride (80x80 = 8,
# 40x40 = 16, 20x20 = 32 at this 640x640 input). See the README's Hailo-8L
# section, which walks through the same procedure.
#
# Both variants' class outputs carry qp_scale 1/255, so the dequantized values
# are already sigmoid probabilities and decode_scrfd needs no per-variant score
# handling. The HEFs are natively UINT8; we request FLOAT32 and let HailoRT
# dequantize.
#
# Latencies are `hailortcli benchmark` on a production Hailo-8, HailoRT 4.24.0.

SCRFD_VARIANTS = {
    "scrfd_2.5g": {
        "hef": "scrfd_2.5g.hef",
        "map": 76.4,
        "latency_ms": 2.53,
        "layers": [
            (8, "scrfd_2_5g/conv42", "scrfd_2_5g/conv43"),
            (16, "scrfd_2_5g/conv49", "scrfd_2_5g/conv50"),
            (32, "scrfd_2_5g/conv55", "scrfd_2_5g/conv56"),
        ],
    },
    "scrfd_10g": {
        "hef": "scrfd_10g.hef",
        "map": 82.1,
        "latency_ms": 4.40,
        "layers": [
            (8, "scrfd_10g/conv41", "scrfd_10g/conv42"),
            (16, "scrfd_10g/conv49", "scrfd_10g/conv50"),
            (32, "scrfd_10g/conv56", "scrfd_10g/conv57"),
        ],
    },
}

# Which detector to load. Named for the task rather than the model family, so a
# future non-SCRFD detector does not need a new variable — the same reasoning
# that makes CLIP_BACKEND the right name for tinyclip/siglip.
FACE_DETECTOR = os.environ.get("FACE_DETECTOR", "scrfd_2.5g").strip().lower()

MODEL_ZOO_BASE = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8"


def scrfd_config_for(name: str) -> ScrfdConfig:
    """Build the ScrfdConfig for a named variant.

    Raises rather than falling back. Silently loading a detector other than the
    one asked for is exactly the class of surprise this project keeps finding —
    and here it would degrade detection quality with nothing in the logs.
    """
    variant = SCRFD_VARIANTS.get(name)
    if variant is None:
        raise ConfigError(
            f"FACE_DETECTOR={name!r} is not a known face detector.\n"
            f"  Valid values: {', '.join(sorted(SCRFD_VARIANTS))}\n"
            f"  Refusing to fall back to a different detector than requested."
        )
    # Fresh list per instance — the registry entry must not be shared or mutated.
    return ScrfdConfig(hef=variant["hef"], output_layers=list(variant["layers"]))


@dataclass
class ArcfaceConfig:
    hef: str = "arcface_r50.hef"
    crop_size: int = 112
    # Host-side cap on frames per infer() call. Only takes effect when a device
    # batch size is configured; without one, all crops go in a single call as
    # they always have. The step actually used is the largest multiple of the
    # device batch that fits under this.
    rec_chunk_size: int = 32


# ── TinyCLIP CLIP backend ────────────────────────────────────────────

@dataclass
class TinyClipImageConfig:
    """TinyCLIP ViT-39M/16 image encoder.

    Fast (~60 FPS), 512-dim embeddings. Uses center-crop preprocessing.
    Not compatible with any Immich default CLIP model embeddings.
    """
    hef: str = "tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder.hef"
    crop_size: int = 224
    embed_dim: int = 512
    # Output is FLOAT32, no dequantization needed
    output_qp_scale: float = 0.0
    output_qp_zp: float = 0.0


@dataclass
class TinyClipTextConfig:
    """TinyCLIP ViT-39M/16 text encoder (BPE tokenizer, UINT16 quantized).

    Output is full sequence (1,1,77,512) — requires EOT-position pooling
    and text projection on CPU.
    """
    hef: str = "tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder.hef"
    weights_npz: str = "tinyclip_text_weights.npz"
    bpe_gz: str = "bpe_simple_vocab_16e6.txt.gz"
    context_length: int = 77
    embed_dim: int = 512
    # FALLBACK ONLY — read from the Hailo-8 HEF. The pipeline prefers the values
    # it reads from whatever HEF is actually loaded; these are used only when the
    # HailoRT version does not expose quant info, or CLIP_QUANT_SOURCE=config.
    input_qp_scale: float = 3.146522067254409e-05
    input_qp_zp: float = 15216.0


# ── SigLIP CLIP backend ──────────────────────────────────────────────

@dataclass
class SiglipImageConfig:
    """SigLIP v1 B/16 image encoder (google/siglip-base-patch16-224).

    Better quality (~13.6 FPS), 768-dim embeddings. Uses squash resize
    (not center-crop). Normalization is baked into the HEF.

    Embedding-compatible with Immich's ViT-B-16-SigLIP__webli (same weights).
    Output is UINT16 — requires dequantization.
    """
    hef: str = "siglip_b_16_image_encoder.hef"
    crop_size: int = 224
    embed_dim: int = 768
    # Output dequantization: float32 = (uint16 - zp) * scale
    # FALLBACK ONLY — see TinyClipTextConfig; read from the loaded HEF by default.
    output_qp_scale: float = 0.00032554997596889734
    output_qp_zp: float = 9506.0


@dataclass
class SiglipTextConfig:
    """SigLIP v1 B/16 text encoder (google/siglip-base-patch16-224).

    Uses SentencePiece tokenizer, 64 max tokens, 768-dim embeddings.
    Output is already pooled (1x1x768) — no CPU-side pooling or projection needed.

    Output is UINT16 — requires dequantization.
    """
    hef: str = "siglip_b_16_text_encoder.hef"
    weights_npz: str = "siglip_text_weights.npz"
    spiece_model: str = "spiece.model"
    context_length: int = 64
    embed_dim: int = 768
    pad_token_id: int = 1  # </s> token
    # FALLBACK ONLY — see TinyClipTextConfig; read from the loaded HEF by default.
    # Input quantization: uint16 = round(float32 / scale + zp)
    input_qp_scale: float = 0.000125956823467277
    input_qp_zp: float = 19290.0
    # Output dequantization: float32 = (uint16 - zp) * scale
    output_qp_scale: float = 0.0006467350176535547
    output_qp_zp: float = 17980.0


# ── OCR ───────────────────────────────────────────────────────────────

@dataclass
class OcrDetectionConfig:
    """PaddleOCR v5 text detection (DBNet with PPLCNetV3 backbone).

    Input: 544x960 UINT8 RGB (normalization baked into HEF).
    Output: 544x960x1 probability map (sigmoid, each pixel = text probability).
    """
    hef: str = "paddle_ocr_v5_mobile_detection.hef"
    input_h: int = 544
    input_w: int = 960
    binary_thresh: float = 0.3
    box_thresh: float = 0.6
    unclip_ratio: float = 1.5
    min_size: int = 3
    max_candidates: int = 1000


@dataclass
class OcrRecognitionConfig:
    """PaddleOCR v5 text recognition (SVTR_LCNet with CTC head).

    Input: 48x320 UINT8 RGB (normalization baked into HEF).
    Output: 1x40x18385 CTC logits (40 time steps, 18385 classes).
    """
    hef: str = "paddle_ocr_v5_mobile_recognition.hef"
    input_h: int = 48
    input_w: int = 320
    char_dict: str = "ppocrv5_dict.txt"
    blank_index: int = 0
    # Text crops are recognized in batches of this many per device round-trip.
    # Detection can emit up to OcrDetectionConfig.max_candidates regions; one
    # inference each would be that many sequential round-trips. Chunking bounds
    # peak memory and stays clear of any per-batch device limit.
    rec_batch_size: int = 32


# ── Pipeline config ───────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    models_dir: str = MODELS_DIR
    clip_backend: str = CLIP_BACKEND  # "tinyclip" or "siglip"
    # Both default to 8, the measured optimum. None disables device batching for
    # that path (no padding, pre-batching behaviour). Separate knobs because the
    # two models have different burst overheads — see HAILO_BATCH_SIZE above.
    face_batch_size: Optional[int] = HAILO_BATCH_SIZE_FACE
    ocr_batch_size: Optional[int] = HAILO_BATCH_SIZE_OCR
    # True: prefer quant params read from the HEF. False: always use the
    # constants in the CLIP configs below.
    quant_from_hef: bool = CLIP_QUANT_SOURCE != "config"

    # Face detection / recognition
    face_detector: str = FACE_DETECTOR
    scrfd: ScrfdConfig = field(default_factory=lambda: scrfd_config_for(FACE_DETECTOR))
    arcface: ArcfaceConfig = field(default_factory=ArcfaceConfig)

    # CLIP — TinyCLIP config (used when clip_backend == "tinyclip")
    tinyclip_image: TinyClipImageConfig = field(default_factory=TinyClipImageConfig)
    tinyclip_text: TinyClipTextConfig = field(default_factory=TinyClipTextConfig)

    # CLIP — SigLIP config (used when clip_backend == "siglip")
    siglip_image: SiglipImageConfig = field(default_factory=SiglipImageConfig)
    siglip_text: SiglipTextConfig = field(default_factory=SiglipTextConfig)

    # OCR
    ocr_detection: OcrDetectionConfig = field(default_factory=OcrDetectionConfig)
    ocr_recognition: OcrRecognitionConfig = field(default_factory=OcrRecognitionConfig)

    def hef_path(self, filename: str) -> str:
        return os.path.join(self.models_dir, filename)
