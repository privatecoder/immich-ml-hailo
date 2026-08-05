"""Centralized model configuration.

All model-specific parameters (paths, layer names, quantization params) live here.
To swap a model, change the config — not the inference code.

CLIP backend is selectable via the CLIP_BACKEND environment variable:
  CLIP_BACKEND=tinyclip  (default) — faster inference, 512-dim embeddings
  CLIP_BACKEND=siglip    — better search quality, 768-dim, Immich-compatible embeddings
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

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


@dataclass
class ScrfdConfig:
    """Face detection model config.

    Default: scrfd_2.5g (76.4 mAP, 1058 FPS on Hailo-8)
    Alternative: scrfd_10g (82.1 mAP, 440 FPS) — higher accuracy, lower throughput.
      To use: download scrfd_10g.hef, inspect with hef_inspect.py, update output_layers.
    """
    hef: str = "scrfd_2.5g.hef"
    input_size: int = 640
    # (stride, cls_layer_name, box_layer_name) — must match the compiled HEF
    output_layers: List[Tuple[int, str, str]] = field(default_factory=lambda: [
        (8, "scrfd_2_5g/conv42", "scrfd_2_5g/conv43"),
        (16, "scrfd_2_5g/conv49", "scrfd_2_5g/conv50"),
        (32, "scrfd_2_5g/conv55", "scrfd_2_5g/conv56"),
    ])


@dataclass
class ArcfaceConfig:
    hef: str = "arcface_r50.hef"
    crop_size: int = 112


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


# ── Pipeline config ───────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    models_dir: str = MODELS_DIR
    clip_backend: str = CLIP_BACKEND  # "tinyclip" or "siglip"
    # True: prefer quant params read from the HEF. False: always use the
    # constants in the CLIP configs below.
    quant_from_hef: bool = CLIP_QUANT_SOURCE != "config"

    # Face detection / recognition
    scrfd: ScrfdConfig = field(default_factory=ScrfdConfig)
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
