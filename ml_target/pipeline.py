"""Inference pipeline: initializes models and dispatches /predict requests."""

import json
import logging
import math
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import hailo_platform as hpf

from ml_target.config import (
    MODEL_ZOO_BASE,
    ConfigError,
    OcrDetectionConfig,
    PipelineConfig,
)
from ml_target.decoders import decode_scrfd
from ml_target.models import (
    HailoModel,
    QuantParams,
    activate_model,
    configure_model,
    infer_single,
    pick_output,
)
from ml_target.preprocessing import (
    crop_and_resize_rgb,
    dequantize_uint16,
    l2_normalize,
    letterbox_rgb,
    prep_clip_text_input,
    prep_siglip_image,
    prep_tinyclip_image,
)
from ml_target.ocr import CTCDecoder, crop_text_region, decode_db_detection
from ml_target.tokenizer import SiglipTokenizer, SimpleTokenizer

LOG = logging.getLogger("ml_target.pipeline")

_PIPE = None


class BadRequest(ValueError):
    """The caller sent something malformed — a 400, not a worker fault.

    Distinct from the ValueErrors raised deeper in the pipeline for unexpected
    model output, which are genuine 500s and must not be reported as the
    client's fault.
    """

# Relative tolerance for treating a HEF-read quant param as equal to the
# config constant. The constants are stored at full float repr, so a genuine
# match is exact; anything looser than this is a real difference.
_QUANT_REL_TOL = 1e-9


def _resolve_quant(
    label: str,
    hef_quant: Optional[QuantParams],
    cfg_scale: float,
    cfg_zp: float,
    prefer_hef: bool,
) -> Tuple[float, float]:
    """Pick the quantization params to use, logging both candidate sources.

    Quantization params belong to a specific HEF compilation, so the values read
    from the loaded HEF are authoritative and the constants in config.py are a
    fallback. Applying the wrong ones is silent — no exception, no warning from
    HailoRT, just subtly wrong embeddings — so this logs what it chose and warns
    loudly when the two sources disagree.
    """
    if hef_quant is None:
        LOG.info(
            "%s quant: not exposed by this HailoRT — using config fallback "
            "scale=%.17g zp=%.17g", label, cfg_scale, cfg_zp,
        )
        return cfg_scale, cfg_zp

    matches = (
        math.isclose(hef_quant.qp_scale, cfg_scale, rel_tol=_QUANT_REL_TOL, abs_tol=1e-15)
        and math.isclose(hef_quant.qp_zp, cfg_zp, rel_tol=_QUANT_REL_TOL, abs_tol=1e-15)
    )

    LOG.info(
        "%s quant: HEF scale=%.17g zp=%.17g | config scale=%.17g zp=%.17g | %s",
        label, hef_quant.qp_scale, hef_quant.qp_zp, cfg_scale, cfg_zp,
        "match" if matches else "MISMATCH",
    )

    if not prefer_hef:
        LOG.info("%s quant: CLIP_QUANT_SOURCE=config — using the config constants", label)
        return cfg_scale, cfg_zp

    if not matches:
        LOG.warning(
            "%s QUANTIZATION MISMATCH — HEF says scale=%.17g zp=%.17g, config.py has "
            "scale=%.17g zp=%.17g. Using the HEF values, which is correct for a HEF "
            "compiled for a different device (e.g. Hailo-8L). But if this device was "
            "previously producing good search results with the config values, then the "
            "HEF-reading logic is wrong, not your hardware: set CLIP_QUANT_SOURCE=config "
            "to restore the previous behaviour, and please report it.",
            label, hef_quant.qp_scale, hef_quant.qp_zp, cfg_scale, cfg_zp,
        )

    return hef_quant.qp_scale, hef_quant.qp_zp


# ── Per-request tracing ──────────────────────────────────────────────
#
# Every stage timer feeds one structured INFO line per request instead of the
# ten-odd scattered INFO lines this used to emit. Detail stays available at
# DEBUG via LOG_LEVEL.
#
# A ContextVar rather than a global: today `predict` is `async def` doing
# synchronous work, so requests are serialised on the event loop and a plain
# global would be fine. ContextVar keeps this correct if that ever changes to a
# threadpool handler, which is the natural next step.

REQ_LOG = logging.getLogger("ml_target.request")

_TRACE: "ContextVar[Optional[_RequestTrace]]" = ContextVar("ml_target_trace", default=None)


class _RequestTrace:
    """Timings and facts accumulated over one /predict call."""

    def __init__(self) -> None:
        self.ms: Dict[str, float] = {}
        self.facts: Dict[str, Any] = {}
        self.total_ms: float = 0.0

    def add_ms(self, stage: str, dt: float) -> None:
        # Accumulate: a stage entered repeatedly (per-chunk OCR recognition,
        # for instance) reports its total, not just the last occurrence.
        self.ms[stage] = self.ms.get(stage, 0.0) + dt

    def note(self, **facts: Any) -> None:
        self.facts.update(facts)

    def format(self) -> str:
        parts = [f"{k}={v}" for k, v in self.facts.items()]
        parts += [f"{k}={v:.1f}ms" for k, v in self.ms.items()]
        parts.append(f"total={self.total_ms:.1f}ms")
        return " ".join(parts)


def note(**facts: Any) -> None:
    """Record facts on the current request trace, if one is active."""
    tr = _TRACE.get()
    if tr is not None:
        tr.note(**facts)


@contextmanager
def request_trace() -> Generator["_RequestTrace", None, None]:
    """Collect timings for one request and emit a single summary line.

    Logs on every exit path — including early returns and exceptions — so a
    request always leaves exactly one trace line behind.
    """
    tr = _RequestTrace()
    token = _TRACE.set(tr)
    t0 = time.perf_counter()
    try:
        yield tr
    finally:
        tr.total_ms = (time.perf_counter() - t0) * 1000.0
        _TRACE.reset(token)
        REQ_LOG.info("/predict %s", tr.format())


class _Timer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = (time.perf_counter() - self.t0) * 1000.0
        LOG.debug("[TIMER] %s: %.2f ms", self.name, dt)
        tr = _TRACE.get()
        if tr is not None:
            tr.add_ms(self.name, dt)


# Public alias — app.py times image decoding with it.
stage = _Timer


# ── Device serialisation ──────────────────────────────────────────────
#
# There is one Hailo accelerator, so one global lock is the right granularity.
# Per-model locks would buy nothing (the device is the contended resource, not
# any individual network group) and would invite lock-ordering deadlocks the
# moment a request touched two models — which the face and OCR paths both do.
#
# RLock rather than Lock: nothing nests today, but a future caller wrapping an
# already-locked helper would deadlock the whole worker with a plain Lock,
# which is a far worse outcome than the reentrancy RLock quietly allows. This
# is the one place where failing safe beats failing loud.
#
# THE POINT IS WHAT STAYS OUTSIDE. Holding this across a whole request would
# reproduce the serialisation it exists to remove, with extra machinery and
# more risk. Only the inference calls themselves are inside.
_DEVICE_LOCK = threading.RLock()

# Guards the one-time pipeline construction. Startup is single-threaded today
# (Starlette runs the startup event before accepting connections), so this is
# belt-and-braces against a future lazy-init path rather than a live race.
_INIT_LOCK = threading.Lock()


@contextmanager
def device_lock() -> Generator[None, None, None]:
    """Serialise access to the accelerator.

    Records the time spent waiting as `lock_wait` on the request trace, so
    contention shows up directly in the per-request summary line and in
    scripts/benchmark.sh rather than having to be inferred.
    """
    t0 = time.perf_counter()
    _DEVICE_LOCK.acquire()
    waited = (time.perf_counter() - t0) * 1000.0
    tr = _TRACE.get()
    if tr is not None:
        tr.add_ms("lock_wait", waited)
    try:
        yield
    finally:
        _DEVICE_LOCK.release()


def _aligned_chunk(host_cap: int, batch_size: int) -> int:
    """Frames per infer() call: the largest multiple of the device batch that
    fits under the host cap.

    So every full chunk is a whole number of device batches. The last chunk of a
    run is whatever is left over and may be shorter — unavoidable, since the
    number of faces or text regions in a photo is not ours to choose.

    Because the step is a whole number of batches, every chunk except the last
    needs no padding at all. Only the final chunk of a request can be short, so
    the padding overhead is bounded by (batch_size - 1) frames per request
    rather than per chunk.
    """
    step = (host_cap // batch_size) * batch_size
    return step if step >= batch_size else batch_size


def _pad_to_batch(batch: np.ndarray, batch_size: Optional[int]) -> Tuple[np.ndarray, int]:
    """Pad a batch up to a whole multiple of batch_size.

    Required, not optional: this device runs multi-context HEFs *without* the
    model scheduler, and HailoRT rejects anything else outright —

        CHECK failed - On the case of multi-context without the model scheduler,
        frames count must be a multiplier of the batch size! (5 % 8 != 0)

    Returns (padded_batch, n_real). **The caller must slice results to
    [:n_real] before using them.** Padded rows carry no meaning and must never
    reach an output list.

    Padding repeats the last real frame rather than using zeros. Both cost the
    same device time, but a repeated real frame is guaranteed to be in the same
    numeric range as the genuine input, so the filler cannot stray into an
    unusual code path. This assumes rows in a batch are independent — true for
    these models, whose inference graphs contain no operation that reduces
    across the batch axis (batch-norm uses fixed statistics at inference). If
    that assumption were ever wrong, the pad content would perturb real rows and
    the golden test would catch it.
    """
    n = int(batch.shape[0])
    if not batch_size or batch_size < 1 or n == 0:
        return batch, n
    remainder = n % batch_size
    if remainder == 0:
        return batch, n
    pad = batch_size - remainder
    filler = np.repeat(batch[-1:], pad, axis=0)
    LOG.debug("padding batch %d -> %d for device batch_size=%d", n, n + pad, batch_size)
    return np.ascontiguousarray(np.concatenate([batch, filler], axis=0)), n


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.vdevice = hpf.VDevice()

        # Face detection.
        #
        # Unlike OCR, this does not degrade gracefully. Face detection is a core
        # task: a worker that starts without it would answer every
        # facial-recognition request with zero faces and no error, which reads
        # to the user as "Immich found no faces in my library".
        det_hef = cfg.hef_path(cfg.scrfd.hef)
        if not os.path.exists(det_hef):
            raise ConfigError(
                f"Face detection model not found: {det_hef}\n"
                f"  FACE_DETECTOR={cfg.face_detector} requires {cfg.scrfd.hef}.\n"
                f"  Download it:\n"
                f"    curl -fLo models/{cfg.scrfd.hef} {MODEL_ZOO_BASE}/{cfg.scrfd.hef}\n"
                f"  or re-run ./setup.sh, which fetches every detector variant.\n"
                f"  Refusing to start: face detection is a core task, and starting\n"
                f"  without it would silently return zero faces for every image."
            )

        LOG.info("Face detector: %s (%s)", cfg.face_detector, cfg.scrfd.hef)
        self.det = configure_model(
            self.vdevice,
            det_hef,
            input_format=hpf.FormatType.UINT8,
            output_format=hpf.FormatType.FLOAT32,
        )

        # Face recognition — one of only two models that ever receives more
        # than one frame, so one of only two worth giving a device batch size.
        #
        # Same loud-failure rule as the detector: a core task, and a worker that
        # started without it would return detections with no embeddings.
        rec_hef = cfg.hef_path(cfg.arcface.hef)
        if not os.path.exists(rec_hef):
            raise ConfigError(
                f"Face recognition model not found: {rec_hef}\n"
                f"  FACE_RECOGNIZER={cfg.face_recognizer} requires {cfg.arcface.hef}.\n"
                f"  Download it:\n"
                f"    curl -fLo models/{cfg.arcface.hef} {MODEL_ZOO_BASE}/{cfg.arcface.hef}\n"
                f"  or re-run ./setup.sh, which fetches every recognition variant.\n"
                f"  Refusing to start: face recognition is a core task."
            )

        LOG.info(
            "Face recognizer: %s (%s, %d-dim, %dx%d crops)",
            cfg.face_recognizer, cfg.arcface.hef,
            cfg.arcface.embed_dim, cfg.arcface.crop_size, cfg.arcface.crop_size,
        )
        LOG.info(
            "Hailo device batch size: face=%s ocr=%s",
            cfg.face_batch_size or "<not configured>",
            cfg.ocr_batch_size or "<not configured>",
        )
        self.rec = configure_model(
            self.vdevice,
            rec_hef,
            input_format=hpf.FormatType.UINT8,
            output_format=hpf.FormatType.FLOAT32,
            batch_size=cfg.face_batch_size,
        )

        # CLIP backend selection
        self.clip_backend = cfg.clip_backend
        LOG.info("CLIP backend: %s", self.clip_backend)

        if self.clip_backend == "siglip":
            sc = cfg.siglip_image
            tc = cfg.siglip_text

            self.clip_img = configure_model(
                self.vdevice,
                cfg.hef_path(sc.hef),
                input_format=hpf.FormatType.UINT8,
                output_format=hpf.FormatType.UINT16,
            )

            self.clip_txt = configure_model(
                self.vdevice,
                cfg.hef_path(tc.hef),
                input_format=hpf.FormatType.UINT16,
                output_format=hpf.FormatType.UINT16,
            )

            w = np.load(cfg.hef_path(tc.weights_npz))
            self.token_embedding = np.asarray(w["token_embedding"], dtype=np.float32)
            self.positional_embedding = np.asarray(w["position_embedding"], dtype=np.float32)
            self.text_projection = None  # SigLIP pools internally, no CPU-side projection
            self.eot_token_id = None

            self.tokenizer = SiglipTokenizer(
                cfg.hef_path(tc.spiece_model),
                pad_token_id=tc.pad_token_id,
            )

            # SigLIP quantizes on the way in and dequantizes on the way out, for
            # both the text and image encoders — all three resolved here.
            self.clip_txt_in_quant = _resolve_quant(
                "siglip_text.input", self.clip_txt.input_quant,
                tc.input_qp_scale, tc.input_qp_zp, cfg.quant_from_hef,
            )
            self.clip_txt_out_quant = _resolve_quant(
                "siglip_text.output", self.clip_txt.output_quant,
                tc.output_qp_scale, tc.output_qp_zp, cfg.quant_from_hef,
            )
            self.clip_img_out_quant = _resolve_quant(
                "siglip_image.output", self.clip_img.output_quant,
                sc.output_qp_scale, sc.output_qp_zp, cfg.quant_from_hef,
            )

            LOG.info("SigLIP text assets loaded:")
            LOG.info("  token_embedding=%s", self.token_embedding.shape)
            LOG.info("  positional_embedding=%s", self.positional_embedding.shape)

        else:  # tinyclip (default)
            sc = cfg.tinyclip_image
            tc = cfg.tinyclip_text

            self.clip_img = configure_model(
                self.vdevice,
                cfg.hef_path(sc.hef),
                input_format=hpf.FormatType.UINT8,
                output_format=hpf.FormatType.FLOAT32,
            )

            self.clip_txt = configure_model(
                self.vdevice,
                cfg.hef_path(tc.hef),
                input_format=hpf.FormatType.UINT16,
                output_format=hpf.FormatType.FLOAT32,
            )

            w = np.load(cfg.hef_path(tc.weights_npz))
            self.token_embedding = np.asarray(w["token_embedding"], dtype=np.float32)
            self.positional_embedding = np.asarray(w["positional_embedding"], dtype=np.float32)
            self.text_projection = np.asarray(w["text_projection"], dtype=np.float32)
            self.eot_token_id = int(np.asarray(w["eot_token_id"]).reshape(()))

            self.tokenizer = SimpleTokenizer(cfg.hef_path(tc.bpe_gz))

            # TinyCLIP quantizes only on the way in, and only for the text
            # encoder. Both TinyCLIP models request FLOAT32 output, so HailoRT
            # dequantizes on-device and there are no output params to resolve.
            self.clip_txt_in_quant = _resolve_quant(
                "tinyclip_text.input", self.clip_txt.input_quant,
                tc.input_qp_scale, tc.input_qp_zp, cfg.quant_from_hef,
            )
            self.clip_txt_out_quant = None
            self.clip_img_out_quant = None

            LOG.info("TinyCLIP text assets loaded:")
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
                batch_size=cfg.ocr_batch_size,
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
    with _INIT_LOCK:
        if _PIPE is not None:
            return
        _init_pipeline_locked(cfg)


def _init_pipeline_locked(cfg: Optional[PipelineConfig]) -> None:
    global _PIPE

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
        note(image=f"{W0}x{H0}")
    else:
        H0, W0 = 0, 0

    if not isinstance(entries, dict):
        raise BadRequest(
            f"entries must be a JSON object mapping task name to config, "
            f"got {type(entries).__name__}"
        )

    note(tasks=",".join(entries.keys()) or "-")

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
        # Degrade per-task rather than failing the request: a ModelTask added
        # by a future Immich release should not take the whole worker down.
        LOG.warning("unsupported task in entries: %r", task_name)
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

    LOG.debug("FACEREC: H=%d W=%d min_score=%.3f iou_thr=%.3f", H0, W0, min_score, iou_thr)

    # Detection
    input_size = cfg.scrfd.input_size
    with _Timer("letterbox"):
        det_rgb, scale, pad_x, pad_y = letterbox_rgb(image_rgb, input_size, input_size)
    xb = np.ascontiguousarray(det_rgb[None, ...], dtype=np.uint8)

    with _Timer("det_infer"), device_lock():
        det_out = infer_single(_PIPE.det, xb)

    with _Timer("det_decode"):
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

    LOG.debug("FACEREC: detections=%d", len(dets))
    note(faces=len(dets))

    faces: List[Dict[str, Any]] = []

    if dets:
        crop_size = cfg.arcface.crop_size

        # Crop all faces
        with _Timer("crop_faces"):
            patches = [
                crop_and_resize_rgb(image_rgb, tuple(d["box"]), out_size=crop_size)
                for d in dets
            ]

        # Recognition, batched in a single activation.
        #
        # patches[i] was built from dets[i], and dets is in descending score
        # order out of NMS. Chunks are contiguous slices processed in order and
        # the results concatenated in the same order, so emb_all[i] stays the
        # embedding of dets[i] — each face keeps its own bounding box.
        def _to_batch(group: List[np.ndarray]) -> np.ndarray:
            if _PIPE.rec.input_format == hpf.FormatType.UINT8:
                b = np.stack(group, axis=0).astype(np.uint8)
            else:
                b = np.stack([
                    ((p.astype(np.float32) / 255.0) - 0.5) / 0.5 for p in group
                ], axis=0).astype(np.float32)
            return np.ascontiguousarray(b)

        rec_batch = _PIPE.rec.batch_size
        # No device batch configured: one call with everything, exactly as before.
        step = _aligned_chunk(cfg.arcface.rec_chunk_size, rec_batch) if rec_batch else len(patches)

        with _Timer("rec_infer_batch"), device_lock():
            with activate_model(_PIPE.rec) as rec_infer:
                parts: List[np.ndarray] = []
                for start in range(0, len(patches), step):
                    group = patches[start : start + step]
                    xb, n_real = _pad_to_batch(_to_batch(group), rec_batch)
                    rec_out = rec_infer(xb)
                    part = np.asarray(pick_output(rec_out, hint=cfg.arcface.output_hint), dtype=np.float32)
                    if part.ndim > 2:
                        part = part.reshape(part.shape[0], -1)
                    # Discard padded rows here, at the boundary, before this
                    # array is appended to anything. Nothing downstream ever
                    # sees a padded row.
                    parts.append(part[:n_real])

        emb_all = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

        if emb_all.shape[1] != cfg.arcface.embed_dim:
            LOG.warning(
                "face embeddings are %d-dim but %s is configured for %d — "
                "check ARCFACE_VARIANTS[%r]['embed_dim'] against the HEF",
                emb_all.shape[1], cfg.face_recognizer, cfg.arcface.embed_dim,
                cfg.face_recognizer,
            )

        if emb_all.shape[0] != len(dets):
            # Refuse to guess: attaching the wrong embedding to a bounding box
            # would corrupt Immich's face clusters silently.
            raise ValueError(
                f"face recognition returned {emb_all.shape[0]} embeddings for "
                f"{len(dets)} detections — refusing to pair them"
            )

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
    is_siglip = _PIPE.clip_backend == "siglip"

    # ── TEXT ──
    if wants_text:
        if not text:
            return {"clip": {"error": "missing text"}}

        LOG.debug("CLIP TEXT [%s]: len=%d", _PIPE.clip_backend, len(text))
        note(backend=_PIPE.clip_backend, clip="textual")

        if is_siglip:
            tc = cfg.siglip_text
            token_ids = _PIPE.tokenizer.tokenize(text, context_length=tc.context_length).astype(np.int32, copy=False)

            in_scale, in_zp = _PIPE.clip_txt_in_quant
            xb = prep_clip_text_input(
                token_ids,
                _PIPE.token_embedding,
                _PIPE.positional_embedding,
                qp_scale=in_scale,
                qp_zp=in_zp,
            )

            with _Timer("clip_text_infer"), device_lock():
                out = infer_single(_PIPE.clip_txt, xb)

            # SigLIP output is 1x1x768 UINT16 — dequantize
            out_scale, out_zp = _PIPE.clip_txt_out_quant
            emb_f32 = dequantize_uint16(
                np.asarray(pick_output(out)).squeeze(),
                out_scale, out_zp,
            )
            emb = l2_normalize(emb_f32)

        else:  # tinyclip
            tc = cfg.tinyclip_text
            token_ids = _PIPE.tokenizer.tokenize(text, context_length=tc.context_length).astype(np.int32, copy=False)

            eot_positions = np.where(token_ids == _PIPE.eot_token_id)[0]
            eot_pos = int(eot_positions[0]) if eot_positions.size > 0 else tc.context_length - 1

            in_scale, in_zp = _PIPE.clip_txt_in_quant
            xb = prep_clip_text_input(
                token_ids,
                _PIPE.token_embedding,
                _PIPE.positional_embedding,
                qp_scale=in_scale,
                qp_zp=in_zp,
            )

            with _Timer("clip_text_infer"), device_lock():
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

    LOG.debug("CLIP IMAGE [%s]: H=%d W=%d", _PIPE.clip_backend, H0, W0)
    note(backend=_PIPE.clip_backend, clip="visual")

    if is_siglip:
        sc = cfg.siglip_image
        with _Timer("clip_preprocess"):
            xb = prep_siglip_image(image_rgb, sc.crop_size)

        with _Timer("clip_image_infer"), device_lock():
            clip_out = infer_single(_PIPE.clip_img, xb)

        out_scale, out_zp = _PIPE.clip_img_out_quant
        emb_f32 = dequantize_uint16(
            np.asarray(pick_output(clip_out)).squeeze(),
            out_scale, out_zp,
        )
        emb = l2_normalize(emb_f32)

    else:  # tinyclip
        sc = cfg.tinyclip_image
        with _Timer("clip_preprocess"):
            xb = prep_tinyclip_image(image_rgb, sc.crop_size, _PIPE.clip_img.input_format)

        with _Timer("clip_image_infer"), device_lock():
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

    LOG.debug("OCR: H=%d W=%d min_det_score=%.3f min_rec_score=%.3f", H0, W0, min_det_score, min_rec_score)

    det_cfg = cfg.ocr_detection
    rec_cfg = cfg.ocr_recognition

    # ── Step 1: Detection — letterbox to model input size ──
    with _Timer("ocr_letterbox"):
        det_input, scale, pad_x, pad_y = letterbox_rgb(
            image_rgb, det_cfg.input_w, det_cfg.input_h, pad_value=0,
        )
    xb = np.ascontiguousarray(det_input[None, ...], dtype=np.uint8)

    with _Timer("ocr_det_infer"), device_lock():
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
    with _Timer("ocr_det_decode"):
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

    LOG.debug("OCR: %d text regions detected", len(text_regions))

    if not text_regions:
        note(ocr="0/0")
        return {"ocr": {"text": [], "box": [], "boxScore": [], "textScore": []},
                "imageHeight": int(H0), "imageWidth": int(W0)}

    # ── Step 2: Recognition — crop every text region, recognize in batches ──
    #
    # These four lists stay index-aligned: text[i], boxScore[i] and textScore[i]
    # describe one region, and box[8i:8i+8] are that region's four corners.
    # Nothing is appended to any of them unless all four are appended together.
    texts: List[str] = []
    boxes: List[float] = []
    box_scores: List[float] = []
    text_scores: List[float] = []

    # crops[i] corresponds to text_regions[i] — built in order, nothing filtered,
    # so a crop's list index is its region's index for the rest of this function.
    with _Timer("ocr_crop"):
        crops = [
            crop_text_region(
                image_rgb,
                region["box"],
                target_h=rec_cfg.input_h,
                target_w=rec_cfg.input_w,
            )
            for region in text_regions
        ]

    # Host chunk, aligned down to a whole number of device batches when one is
    # configured. The chunking structure below — and with it the index
    # alignment — is unchanged; only the step size moves.
    ocr_batch = _PIPE.ocr_rec.batch_size
    chunk_size = max(1, int(rec_cfg.rec_batch_size))
    if ocr_batch:
        chunk_size = _aligned_chunk(chunk_size, ocr_batch)

    with _Timer("ocr_rec_batch"), device_lock():
        with activate_model(_PIPE.ocr_rec) as rec_infer:
            for start in range(0, len(crops), chunk_size):
                chunk = crops[start : start + chunk_size]
                batch = np.ascontiguousarray(np.stack(chunk, axis=0), dtype=np.uint8)
                batch, n_real = _pad_to_batch(batch, ocr_batch)

                rec_out = rec_infer(batch)
                logits = np.asarray(pick_output(rec_out), dtype=np.float32)

                # Normalize to (N, T, C). Do NOT squeeze() — a chunk of one
                # would lose its batch axis and silently shift every index.
                if logits.ndim > 3:
                    logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
                elif logits.ndim == 2:
                    logits = logits[None, ...]

                # Drop padded rows before anything else looks at this array, so
                # the check below and the decode both see exactly the real
                # regions. n_real == len(chunk) by construction.
                if logits.ndim == 3:
                    logits = logits[:n_real]

                if logits.ndim != 3 or logits.shape[0] != len(chunk):
                    # Refuse to guess which row belongs to which region: a wrong
                    # guess would misattribute text to boxes. Drop the chunk.
                    LOG.warning(
                        "OCR recognition: unusable output shape %s for a batch of %d "
                        "— skipping regions %d-%d",
                        logits.shape, len(chunk), start, start + len(chunk) - 1,
                    )
                    continue

                decoded = _PIPE.ctc_decoder.decode(logits)

                # decoded[j] is row j of this chunk, which is crops[start + j],
                # which is text_regions[start + j]. Resolve the region first,
                # then filter — so a rejected region contributes to none of the
                # four lists and the alignment above is preserved.
                for j, (text, confidence) in enumerate(decoded):
                    if not text or confidence < min_rec_score:
                        continue
                    region = text_regions[start + j]
                    texts.append(text)
                    boxes.extend(region["box"])
                    box_scores.append(region["score"])
                    text_scores.append(confidence)

    LOG.debug("OCR: %d text regions recognized (of %d detected)", len(texts), len(text_regions))
    note(ocr=f"{len(texts)}/{len(text_regions)}")

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
