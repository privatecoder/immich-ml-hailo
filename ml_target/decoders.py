"""Post-processing decoders for detection model outputs."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from ml_target.config import ScrfdConfig

LOG = logging.getLogger("ml_target.decoders")


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Standard NMS on (x1, y1, x2, y2) boxes."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thr]
    return keep


def decode_scrfd(
    outputs: Dict[str, np.ndarray],
    score_thr: float,
    iou_thr: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    orig_w: int,
    orig_h: int,
    scrfd_cfg: ScrfdConfig,
) -> List[Dict[str, Any]]:
    """Decode SCRFD multi-scale outputs into face detections.

    Uses layer names from scrfd_cfg.output_layers so the decoder works
    with any SCRFD variant (2.5g, 10g, etc.) without code changes.
    """
    LOG.info(
        "SCRFD decode: score_thr=%.3f iou_thr=%.3f scale=%.6f pad=(%d,%d) orig=(%d,%d)",
        score_thr, iou_thr, scale, pad_x, pad_y, orig_w, orig_h,
    )

    feats: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for stride, cls_name, box_name in scrfd_cfg.output_layers:
        if cls_name in outputs and box_name in outputs:
            feats.append((stride, outputs[cls_name], outputs[box_name]))

    if not feats:
        LOG.warning("SCRFD decode: no matching output layers found in %s", list(outputs.keys()))
        return []

    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    for stride, cls_map, box_map in feats:
        cls_map = cls_map[0].astype(np.float32, copy=False)  # (H, W, 2)
        box_map = box_map[0].astype(np.float32, copy=False)  # (H, W, 8)
        H, W = cls_map.shape[:2]

        fg_score = np.maximum(cls_map[..., 0], cls_map[..., 1])

        if box_map.shape[2] != 8:
            raise ValueError(f"Unexpected bbox channels for SCRFD: {box_map.shape}")

        box_map_f = box_map.reshape(H, W, 2, 4)  # 2 anchors

        ys = (np.arange(H, dtype=np.float32) + 0.5) * float(stride)
        xs = (np.arange(W, dtype=np.float32) + 0.5) * float(stride)
        xv, yv = np.meshgrid(xs, ys)
        centers = np.stack([xv, yv], axis=-1)  # (H, W, 2)
        centers_a = np.repeat(centers[:, :, None, :], 2, axis=2)  # (H, W, 2, 2)

        s = float(stride)
        l = box_map_f[..., 0] * s
        t = box_map_f[..., 1] * s
        r = box_map_f[..., 2] * s
        b = box_map_f[..., 3] * s

        x1 = centers_a[..., 0] - l
        y1 = centers_a[..., 1] - t
        x2 = centers_a[..., 0] + r
        y2 = centers_a[..., 1] + b

        boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)
        scores = np.repeat(fg_score.reshape(-1), 2)

        keep_mask = scores >= score_thr
        if not np.any(keep_mask):
            continue

        all_boxes.append(boxes[keep_mask])
        all_scores.append(scores[keep_mask])

    if not all_boxes:
        LOG.info("SCRFD decode: no boxes passed score_thr=%.3f", score_thr)
        return []

    boxes_lb = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    boxes_lb[:, [0, 2]] -= float(pad_x)
    boxes_lb[:, [1, 3]] -= float(pad_y)
    boxes_orig = boxes_lb / float(scale)

    boxes_orig[:, 0] = np.clip(boxes_orig[:, 0], 0, orig_w - 1)
    boxes_orig[:, 1] = np.clip(boxes_orig[:, 1], 0, orig_h - 1)
    boxes_orig[:, 2] = np.clip(boxes_orig[:, 2], 0, orig_w - 1)
    boxes_orig[:, 3] = np.clip(boxes_orig[:, 3], 0, orig_h - 1)

    keep_idx = nms_xyxy(boxes_orig, scores, iou_thr)
    dets: List[Dict[str, Any]] = []
    for i in keep_idx:
        x1, y1, x2, y2 = boxes_orig[i].tolist()
        dets.append({"box": [x1, y1, x2, y2], "score": float(scores[i])})

    LOG.info("SCRFD decode: after NMS keep=%d", len(dets))
    return dets
