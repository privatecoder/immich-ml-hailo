"""PaddleOCR v5 post-processing: DBNet text detection + CTC text recognition."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from ml_target.config import OcrDetectionConfig

LOG = logging.getLogger("ml_target.ocr")


# ── DBNet post-processing (text detection) ────────────────────────────

def decode_db_detection(
    prob_map: np.ndarray,
    cfg: OcrDetectionConfig,
    scale_x: float,
    scale_y: float,
    pad_x: int,
    pad_y: int,
    orig_w: int,
    orig_h: int,
) -> List[Dict[str, Any]]:
    """Decode DBNet probability map into text region bounding boxes.

    Args:
        prob_map: (H, W) float32 probability map from detection model.
        cfg: Detection config with thresholds.
        scale_x, scale_y: Scale factors from original image to model input.
        pad_x, pad_y: Padding offsets from letterbox.
        orig_w, orig_h: Original image dimensions.

    Returns:
        List of {"box": [x1,y1,x2,y2,x3,y3,x4,y4], "score": float} dicts.
    """
    # Binarize
    mask = (prob_map > cfg.binary_thresh).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    results: List[Dict[str, Any]] = []
    for contour in contours[:cfg.max_candidates]:
        if contour.shape[0] < 4:
            continue

        # Minimum area rotated rectangle
        rect = cv2.minAreaRect(contour)
        w_rect, h_rect = rect[1]
        if min(w_rect, h_rect) < cfg.min_size:
            continue

        # Score: mean probability inside the contour
        score = _polygon_score(prob_map, contour)
        if score < cfg.box_thresh:
            continue

        # Unclip (expand the polygon)
        points = cv2.boxPoints(rect).astype(np.float32)
        expanded = _unclip_polygon(points, cfg.unclip_ratio)
        if expanded is None:
            continue

        # Get new bounding rect of expanded polygon
        expanded_rect = cv2.minAreaRect(expanded)
        w_exp, h_exp = expanded_rect[1]
        if min(w_exp, h_exp) < cfg.min_size + 2:
            continue

        box_points = cv2.boxPoints(expanded_rect).astype(np.float32)

        # Map back to original image coordinates
        box_points[:, 0] = (box_points[:, 0] - pad_x) / scale_x
        box_points[:, 1] = (box_points[:, 1] - pad_y) / scale_y

        # Clip to image bounds
        box_points[:, 0] = np.clip(box_points[:, 0], 0, orig_w - 1)
        box_points[:, 1] = np.clip(box_points[:, 1], 0, orig_h - 1)

        # Order points: top-left, top-right, bottom-right, bottom-left
        box_points = _order_points(box_points)

        results.append({
            "box": box_points.reshape(-1).tolist(),  # [x1,y1,x2,y2,x3,y3,x4,y4]
            "score": float(score),
        })

    LOG.info("DBNet decode: %d text regions (from %d contours)", len(results), len(contours))
    return results


def _polygon_score(prob_map: np.ndarray, contour: np.ndarray) -> float:
    """Compute mean probability score inside a contour."""
    h, w = prob_map.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour.reshape(-1, 2)], 1)
    return float(cv2.mean(prob_map, mask=mask)[0])


def _unclip_polygon(points: np.ndarray, unclip_ratio: float) -> Optional[np.ndarray]:
    """Expand a polygon by unclip_ratio using pyclipper."""
    poly = Polygon(points)
    if poly.area <= 0:
        return None

    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(
        [(int(p[0]), int(p[1])) for p in points],
        pyclipper.JT_ROUND,
        pyclipper.ET_CLOSEDPOLYGON,
    )
    expanded = offset.Execute(distance)
    if not expanded:
        return None

    return np.array(expanded[0], dtype=np.float32)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]  # bottom-right: largest x+y
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right: smallest y-x
    rect[3] = pts[np.argmax(d)]  # bottom-left: largest y-x
    return rect


# ── Perspective crop for recognition ──────────────────────────────────

def crop_text_region(
    image_rgb: np.ndarray,
    box: List[float],
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Crop and warp a text region from the original image for recognition.

    Args:
        image_rgb: Original image (H, W, 3) uint8.
        box: 8 floats [x1,y1, x2,y2, x3,y3, x4,y4] — ordered polygon corners.
        target_h: Recognition model input height (48).
        target_w: Recognition model input width (320).

    Returns:
        Cropped and resized image (target_h, target_w, 3) uint8, padded with 128.
    """
    pts = np.array(box, dtype=np.float32).reshape(4, 2)

    # Compute width and height of the text box
    w1 = np.linalg.norm(pts[1] - pts[0])
    w2 = np.linalg.norm(pts[2] - pts[3])
    crop_w = max(int(round(max(w1, w2))), 1)

    h1 = np.linalg.norm(pts[3] - pts[0])
    h2 = np.linalg.norm(pts[2] - pts[1])
    crop_h = max(int(round(max(h1, h2))), 1)

    # Perspective transform to rectangle
    dst = np.array([
        [0, 0],
        [crop_w, 0],
        [crop_w, crop_h],
        [0, crop_h],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    cropped = cv2.warpPerspective(image_rgb, M, (crop_w, crop_h))

    # Resize to target height, preserving aspect ratio
    aspect = crop_w / max(crop_h, 1)
    resize_w = min(int(round(target_h * aspect)), target_w)
    resized = cv2.resize(cropped, (resize_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target_w with gray (128)
    out = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    out[:, :resize_w] = resized

    return out


# ── CTC greedy decode (text recognition) ─────────────────────────────

class CTCDecoder:
    """CTC greedy decoder for PaddleOCR v5 recognition output."""

    def __init__(self, char_dict_path: str, blank_index: int = 0):
        self.blank_index = blank_index
        self.chars = self._load_dict(char_dict_path)
        LOG.info("CTC decoder loaded: %d characters from %s", len(self.chars), char_dict_path)

    def _load_dict(self, path: str) -> List[str]:
        """Load character dictionary. Format: one character per line."""
        with open(path, "r", encoding="utf-8") as f:
            chars = [line.rstrip("\n") for line in f]
        # Index 0 is reserved for CTC blank, so chars[0] maps to output index 1
        return chars

    def decode(self, logits: np.ndarray) -> List[Tuple[str, float]]:
        """Decode CTC logits into text strings.

        Args:
            logits: (batch, time_steps, num_classes) float32 array.

        Returns:
            List of (text, confidence) tuples, one per batch item.
        """
        if logits.ndim == 2:
            logits = logits[None, ...]  # Add batch dim

        results: List[Tuple[str, float]] = []
        for b in range(logits.shape[0]):
            seq = logits[b]  # (T, C)
            indices = np.argmax(seq, axis=1)  # (T,)
            probs = np.max(seq, axis=1)  # (T,)

            # CTC collapse: remove consecutive duplicates and blanks
            chars: List[str] = []
            char_probs: List[float] = []
            prev_idx = -1
            for t in range(len(indices)):
                idx = int(indices[t])
                if idx == prev_idx:
                    continue
                prev_idx = idx
                if idx == self.blank_index:
                    continue
                # Map index to character (index 1 -> chars[0], etc.)
                char_idx = idx - 1
                if 0 <= char_idx < len(self.chars):
                    chars.append(self.chars[char_idx])
                    char_probs.append(float(probs[t]))

            text = "".join(chars)
            confidence = float(np.mean(char_probs)) if char_probs else 0.0
            results.append((text, confidence))

        return results
