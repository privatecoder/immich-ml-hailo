"""Centralized model configuration.

All model-specific parameters (paths, layer names, quantization params) live here.
To swap a model, change the config — not the inference code.

Alternative model configs are provided as commented examples. To use them:
1. Download the HEF from the Hailo model zoo
2. Inspect it with: python3 -m ml_target.hef_inspect /app/models/<model>.hef
3. Update the config dataclass with the correct layer names and quant params
4. Rebuild the Docker image
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


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


@dataclass
class ClipImageConfig:
    """CLIP image encoder config.

    Default: TinyCLIP ViT-39M/16 (512-dim, fast, good quality)
    Alternative: SigLIP2 B/16 (siglip2_b_16_256.hef) — newer architecture,
      better zero-shot performance. Requires different tokenizer (SentencePiece
      instead of BPE) and new text_weights.npz. Input may be 256x256.
    """
    hef: str = "tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder.hef"
    crop_size: int = 224
    embed_dim: int = 512


@dataclass
class ClipTextConfig:
    """CLIP text encoder config.

    Default: TinyCLIP ViT-39M/16 text encoder (BPE tokenizer, UINT16 quantized)
    Alternative: SigLIP2 B/16 text (siglip2_b_16_256_text.hef) — requires
      SentencePiece tokenizer and different weights/quant params.
    """
    hef: str = "tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder.hef"
    weights_npz: str = "tinyclip_text_weights.npz"
    bpe_gz: str = "bpe_simple_vocab_16e6.txt.gz"
    context_length: int = 77
    embed_dim: int = 512
    # Quantization params extracted from the specific HEF via hef_inspect.py
    qp_scale: float = 3.146522067254409e-05
    qp_zp: float = 15216.0


@dataclass
class OcrDetectionConfig:
    """PaddleOCR v5 text detection (DBNet with PPLCNetV3 backbone).

    Input: 544x960 UINT8 RGB (normalization baked into HEF).
    Output: 544x960x1 probability map (sigmoid, each pixel = text probability).
    """
    hef: str = "paddle_ocr_v5_mobile_detection.hef"
    input_h: int = 544
    input_w: int = 960
    # DBNet post-processing thresholds
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
    Character dictionary: ppocrv5_dict.txt (18383 chars + blank + space).
    """
    hef: str = "paddle_ocr_v5_mobile_recognition.hef"
    input_h: int = 48
    input_w: int = 320
    char_dict: str = "ppocrv5_dict.txt"
    # CTC decode: index 0 = blank token
    blank_index: int = 0


@dataclass
class PipelineConfig:
    models_dir: str = MODELS_DIR
    scrfd: ScrfdConfig = field(default_factory=ScrfdConfig)
    arcface: ArcfaceConfig = field(default_factory=ArcfaceConfig)
    clip_image: ClipImageConfig = field(default_factory=ClipImageConfig)
    clip_text: ClipTextConfig = field(default_factory=ClipTextConfig)
    ocr_detection: OcrDetectionConfig = field(default_factory=OcrDetectionConfig)
    ocr_recognition: OcrRecognitionConfig = field(default_factory=OcrRecognitionConfig)

    def hef_path(self, filename: str) -> str:
        return os.path.join(self.models_dir, filename)
