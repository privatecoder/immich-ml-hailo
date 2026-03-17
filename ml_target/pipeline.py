"""Inference pipeline: initializes models and dispatches /predict requests."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import hailo_platform as hpf

from ml_target.config import OcrDetectionConfig, PipelineConfig
from ml_target.decoders import decode_scrfd
from ml_target.models import (
    HailoModel,
    activate_model,
    configure_model,
    infer_single,
    pick_output,
)
from ml_target.preprocessing import (
    center_crop_square,
    crop_and_resize_rgb,
    l2_normalize,
    letterbox_rgb,
    prep_clip_image,
    prep_clip_text_input,
)
from ml_target.ocr import CTCDecoder, crop_text_region, decode_db_detection
from ml_target.tokenizer import SimpleTokenizer

LOG = logging.getLogger("ml_target.pipeline")

_PIPE = None


class _Timer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        LOG.debug("[TIMER START] %s", self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = (time.time() - self.t0) * 1000.0
        LOG.debug("[TIMER END] %s: %.2f ms", self.name, dt)


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.vdevice = hpf.VDevice()

        # Face detection
        self.det = configure_model(
            self.vdevice,
            cfg.hef_path(cfg.scrfd.hef),
            input_format=hpf.FormatType.UINT8,
            output_format=hpf.FormatType.FLOAT32,
        )

        # Face recognition
        self.rec = configure_model(
            self.vdevice,
            cfg.hef_path(cfg.arcface.hef),
            input_format=hpf.FormatType.UINT8,
            output_format=hpf.FormatType.FLOAT32,
        )

        # CLIP image encoder
        self.clip_img = configure_model(
            self.vdevice,
            cfg.hef_path(cfg.clip_image.hef),
            input_format=hpf.FormatType.UINT8,
            output_format=hpf.FormatType.FLOAT32,
        )

        # CLIP text encoder
        self.clip_txt = configure_model(
            self.vdevice,
            cfg.hef_path(cfg.clip_text.hef),
            input_format=hpf.FormatType.UINT16,
            output_format=hpf.FormatType.FLOAT32,
        )

        # CLIP text weights
        w = np.load(cfg.hef_path(cfg.clip_text.weights_npz))
        self.token_embedding = np.asarray(w["token_embedding"], dtype=np.float32)
        self.positional_embedding = np.asarray(w["positional_embedding"], dtype=np.float32)
        self.text_projection = np.asarray(w["text_projection"], dtype=np.float32)
        self.eot_token_id = int(np.asarray(w["eot_token_id"]).reshape(()))

        self.tokenizer = SimpleTokenizer(cfg.hef_path(cfg.clip_text.bpe_gz))

        LOG.info("CLIP text assets loaded:")
        LOG.info("  token_embedding=%s", self.token_embedding.shape)
        LOG.info("  positional_embedding=%s", self.positional_embedding.shape)
        LOG.info("  text_projection=%s", self.text_projection.shape)
        LOG.info("  eot_token_id=%d", self.eot_token_id)

        # OCR models (optional — loaded only if HEFs and char dict exist)
        self.ocr_det: Optional[HailoModel] = None
        self.ocr_rec: Optional[HailoModel] = None
        self.ctc_decoder: Optional[CTCDecoder] = None
        ocr_det_path = cfg.hef_path(cfg.ocr_detection.hef)
        ocr_rec_path = cfg.hef_path(cfg.ocr_recognition.hef)
        char_dict_path = cfg.hef_path(cfg.ocr_recognition.char_dict)
        if (os.path.exists(ocr_det_path)
                and os.path.exists(ocr_rec_path)
                and os.path.exists(char_dict_path)):
            self.ocr_det = configure_model(
                self.vdevice,
                ocr_det_path,
                input_format=hpf.FormatType.UINT8,
                output_format=hpf.FormatType.FLOAT32,
            )
            self.ocr_rec = configure_model(
                self.vdevice,
                ocr_rec_path,
                input_format=hpf.FormatType.UINT8,
                output_format=hpf.FormatType.FLOAT32,
            )
            self.ctc_decoder = CTCDecoder(
                char_dict_path,
                blank_index=cfg.ocr_recognition.blank_index,
            )
            LOG.info("OCR models loaded: det=%s rec=%s dict=%s",
                     ocr_det_path, ocr_rec_path, char_dict_path)
        else:
            missing = [p for p in [ocr_det_path, ocr_rec_path, char_dict_path]
                       if not os.path.exists(p)]
            LOG.info("OCR disabled, missing files: %s", missing)


def init_pipeline(cfg: Optional[PipelineConfig] = None) -> None:
    global _PIPE
    if _PIPE is not None:
        return

    if cfg is None:
        cfg = PipelineConfig()

    LOG.info("init_pipeline: models_dir=%s", cfg.models_dir)
    _PIPE = Pipeline(cfg)
    LOG.info("init_pipeline: OK")


def run_inference(
    entries: Dict[str, Any],
    image_rgb: Optional[np.ndarray],
    text: Optional[str] = None,
) -> Dict[str, Any]:
    if _PIPE is None:
        raise RuntimeError("Pipeline not initialized. Call init_pipeline().")

    resp: Dict[str, Any] = {}
    cfg = _PIPE.cfg

    # Validate image if present
    if image_rgb is not None:
        if not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8:
            raise ValueError("image_rgb must be np.uint8 ndarray")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"image_rgb must be HWC RGB, got {image_rgb.shape}")
        H0, W0 = image_rgb.shape[:2]
    else:
        H0, W0 = 0, 0

    for task_name, task_cfg in entries.items():

        # ── FACE DETECTION + RECOGNITION ──────────────────────────
        if task_name == "facial-recognition":
            resp.update(_run_facial_recognition(
                task_cfg, image_rgb, H0, W0, cfg,
            ))
            continue

        # ── CLIP (Smart Search) ───────────────────────────────────
        if task_name == "clip":
            resp.update(_run_clip(
                task_cfg, image_rgb, text, H0, W0, cfg,
            ))
            continue

        # ── OCR ───────────────────────────────────────────────────
        if task_name == "ocr":
            resp.update(_run_ocr(
                task_cfg, image_rgb, H0, W0, cfg,
            ))
            continue

        # ── Unknown task ──────────────────────────────────────────
        resp[task_name] = {"error": f"unsupported task: {task_name}"}

    return resp


# ── Task implementations ─────────────────────────────────────────────

def _run_facial_recognition(
    task_cfg: Any,
    image_rgb: Optional[np.ndarray],
    H0: int, W0: int,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    if image_rgb is None:
        return {"facial-recognition": {"error": "missing image"}}

    det_opts = task_cfg.get("detection", {}).get("options", {}) if isinstance(task_cfg, dict) else {}
    min_score = float(det_opts.get("minScore", 0.7))
    iou_thr = float(det_opts.get("iouThreshold", 0.4))

    LOG.info("FACEREC: H=%d W=%d min_score=%.3f iou_thr=%.3f", H0, W0, min_score, iou_thr)

    # Detection
    input_size = cfg.scrfd.input_size
    with _Timer("letterbox"):
        det_rgb, scale, pad_x, pad_y = letterbox_rgb(image_rgb, input_size, input_size)
    xb = np.ascontiguousarray(det_rgb[None, ...], dtype=np.uint8)

    with _Timer("det_infer"):
        det_out = infer_single(_PIPE.det, xb)

    dets = decode_scrfd(
        det_out,
        score_thr=min_score,
        iou_thr=iou_thr,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        orig_w=W0,
        orig_h=H0,
        scrfd_cfg=cfg.scrfd,
    )

    LOG.info("FACEREC: detections=%d", len(dets))

    faces: List[Dict[str, Any]] = []

    if dets:
        crop_size = cfg.arcface.crop_size

        # Crop all faces
        with _Timer("crop_faces"):
            patches = [
                crop_and_resize_rgb(image_rgb, tuple(d["box"]), out_size=crop_size)
                for d in dets
            ]

        # Batch recognition in a single activation
        with _Timer("rec_infer_batch"):
            with activate_model(_PIPE.rec) as rec_infer:
                if _PIPE.rec.input_format == hpf.FormatType.UINT8:
                    batch = np.stack(patches, axis=0).astype(np.uint8)
                else:
                    batch = np.stack([
                        ((p.astype(np.float32) / 255.0) - 0.5) / 0.5 for p in patches
                    ], axis=0).astype(np.float32)
                batch = np.ascontiguousarray(batch)
                rec_out = rec_infer(batch)

        emb_all = np.asarray(pick_output(rec_out, hint="fc1"), dtype=np.float32)

        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d["box"]
            emb = l2_normalize(emb_all[i])
            emb_str = json.dumps(emb.tolist(), separators=(",", ":"))

            faces.append({
                "boundingBox": {
                    "x1": int(round(x1)),
                    "y1": int(round(y1)),
                    "x2": int(round(x2)),
                    "y2": int(round(y2)),
                },
                "score": float(d["score"]),
                "embedding": emb_str,
            })

    return {
        "facial-recognition": faces,
        "imageHeight": int(H0),
        "imageWidth": int(W0),
    }


def _run_clip(
    task_cfg: Any,
    image_rgb: Optional[np.ndarray],
    text: Optional[str],
    H0: int, W0: int,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    clip_cfg = task_cfg if isinstance(task_cfg, dict) else {}
    wants_text = "text" in clip_cfg or "textual" in clip_cfg

    # ── TEXT ──
    if wants_text:
        if not text:
            return {"clip": {"error": "missing text"}}

        LOG.info("CLIP TEXT: len=%d", len(text))

        tc = cfg.clip_text
        token_ids = _PIPE.tokenizer.tokenize(text, context_length=tc.context_length).astype(np.int32, copy=False)

        eot_positions = np.where(token_ids == _PIPE.eot_token_id)[0]
        eot_pos = int(eot_positions[0]) if eot_positions.size > 0 else tc.context_length - 1

        xb = prep_clip_text_input(
            token_ids,
            _PIPE.token_embedding,
            _PIPE.positional_embedding,
            qp_scale=tc.qp_scale,
            qp_zp=tc.qp_zp,
        )

        with _Timer("clip_text_infer"):
            out = infer_single(_PIPE.clip_txt, xb)

        y = np.asarray(pick_output(out), dtype=np.float32).squeeze()
        if y.shape != (tc.context_length, tc.embed_dim):
            raise ValueError(f"Unexpected text encoder output shape: {y.shape}")

        eot_vec = y[eot_pos].reshape(1, tc.embed_dim)
        proj = (eot_vec @ _PIPE.text_projection).reshape(-1)
        emb = l2_normalize(proj)

        return {"clip": json.dumps(emb.tolist(), separators=(",", ":"))}

    # ── VISUAL ──
    if image_rgb is None:
        return {"clip": {"error": "missing image"}}

    LOG.info("CLIP IMAGE: H=%d W=%d", H0, W0)

    xb = prep_clip_image(image_rgb, cfg.clip_image.crop_size, _PIPE.clip_img.input_format)

    with _Timer("clip_image_infer"):
        clip_out = infer_single(_PIPE.clip_img, xb)

    emb = l2_normalize(np.asarray(pick_output(clip_out), dtype=np.float32).squeeze())

    return {
        "clip": json.dumps(emb.tolist(), separators=(",", ":")),
        "imageHeight": int(H0),
        "imageWidth": int(W0),
    }


def _run_ocr(
    task_cfg: Any,
    image_rgb: Optional[np.ndarray],
    H0: int, W0: int,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    """OCR pipeline: DBNet text detection -> CTC text recognition.

    Uses PaddleOCR v5 mobile models on Hailo-8.
    """
    if _PIPE.ocr_det is None or _PIPE.ocr_rec is None or _PIPE.ctc_decoder is None:
        return {"ocr": {"error": "OCR models not available. Add PaddleOCR v5 HEF files and ppocrv5_dict.txt to models directory."}}

    if image_rgb is None:
        return {"ocr": {"error": "missing image"}}

    det_opts = task_cfg.get("detection", {}).get("options", {}) if isinstance(task_cfg, dict) else {}
    rec_opts = task_cfg.get("recognition", {}).get("options", {}) if isinstance(task_cfg, dict) else {}
    min_det_score = float(det_opts.get("minScore", cfg.ocr_detection.box_thresh))
    min_rec_score = float(rec_opts.get("minScore", 0.9))

    LOG.info("OCR: H=%d W=%d min_det_score=%.3f min_rec_score=%.3f", H0, W0, min_det_score, min_rec_score)

    det_cfg = cfg.ocr_detection
    rec_cfg = cfg.ocr_recognition

    # ── Step 1: Detection — letterbox to model input size ──
    with _Timer("ocr_letterbox"):
        det_input, scale, pad_x, pad_y = letterbox_rgb(
            image_rgb, det_cfg.input_w, det_cfg.input_h, pad_value=0,
        )
    xb = np.ascontiguousarray(det_input[None, ...], dtype=np.uint8)

    with _Timer("ocr_det_infer"):
        det_out = infer_single(_PIPE.ocr_det, xb)

    # Get probability map from detection output
    prob_map = np.asarray(pick_output(det_out), dtype=np.float32).squeeze()
    if prob_map.ndim != 2:
        LOG.warning("OCR det output unexpected shape: %s", prob_map.shape)
        return {"ocr": {"text": [], "box": [], "boxScore": [], "textScore": []},
                "imageHeight": int(H0), "imageWidth": int(W0)}

    # Override box_thresh with request-level minScore for detection
    det_cfg_override = OcrDetectionConfig(
        hef=det_cfg.hef,
        input_h=det_cfg.input_h,
        input_w=det_cfg.input_w,
        binary_thresh=det_cfg.binary_thresh,
        box_thresh=min_det_score,
        unclip_ratio=det_cfg.unclip_ratio,
        min_size=det_cfg.min_size,
        max_candidates=det_cfg.max_candidates,
    )

    # letterbox_rgb uses uniform scaling, so scale_x == scale_y == scale
    text_regions = decode_db_detection(
        prob_map,
        cfg=det_cfg_override,
        scale_x=scale,
        scale_y=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        orig_w=W0,
        orig_h=H0,
    )

    LOG.info("OCR: %d text regions detected", len(text_regions))

    if not text_regions:
        return {"ocr": {"text": [], "box": [], "boxScore": [], "textScore": []},
                "imageHeight": int(H0), "imageWidth": int(W0)}

    # ── Step 2: Recognition — crop each text region and run through recognizer ──
    texts: List[str] = []
    boxes: List[float] = []
    box_scores: List[float] = []
    text_scores: List[float] = []

    with _Timer("ocr_rec_batch"):
        with activate_model(_PIPE.ocr_rec) as rec_infer:
            for region in text_regions:
                crop = crop_text_region(
                    image_rgb,
                    region["box"],
                    target_h=rec_cfg.input_h,
                    target_w=rec_cfg.input_w,
                )
                crop_batch = np.ascontiguousarray(crop[None, ...], dtype=np.uint8)

                rec_out = rec_infer(crop_batch)
                logits = np.asarray(pick_output(rec_out), dtype=np.float32).squeeze()

                # CTC decode
                if logits.ndim == 1:
                    continue
                decoded = _PIPE.ctc_decoder.decode(logits[None, ...] if logits.ndim == 2 else logits)
                if not decoded:
                    continue

                text, confidence = decoded[0]
                if not text or confidence < min_rec_score:
                    continue

                texts.append(text)
                boxes.extend(region["box"])
                box_scores.append(region["score"])
                text_scores.append(confidence)

    LOG.info("OCR: %d text regions recognized (of %d detected)", len(texts), len(text_regions))

    return {
        "ocr": {
            "text": texts,
            "box": boxes,
            "boxScore": box_scores,
            "textScore": text_scores,
        },
        "imageHeight": int(H0),
        "imageWidth": int(W0),
    }
