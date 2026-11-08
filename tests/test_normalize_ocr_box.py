#!/usr/bin/env python3
"""OCR boxes returned to Immich are fractions of the image, not pixels.

Runs without a Hailo device or the image libraries.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_target.ocr_coords import normalize_ocr_box


def main() -> int:
    pixel = [0, 10, 100, 10, 100, 50, 0, 50]
    scaled = normalize_ocr_box(pixel, 200, 100)
    expect = [0.0, 0.1, 0.5, 0.1, 0.5, 0.5, 0.0, 0.5]
    if scaled != expect:
        print(f"FAIL: scaled {scaled} != {expect}")
        return 1
    # x uses width, y uses height, including a non-square image.
    wide = normalize_ocr_box([50, 25], 200, 50)
    if wide != [0.25, 0.5]:
        print(f"FAIL: axis scale {wide}")
        return 1
    unchanged = normalize_ocr_box(pixel, 0, 100)
    if unchanged != [float(v) for v in pixel]:
        print(f"FAIL: zero width changed the box to {unchanged}")
        return 1
    if normalize_ocr_box([], 10, 10) != []:
        print("FAIL: empty box")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
