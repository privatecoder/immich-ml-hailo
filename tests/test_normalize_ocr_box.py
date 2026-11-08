#!/usr/bin/env python3
"""OCR response boxes stay in pixels unless OCR_BOX_SPACE=normalized.

Runs without a Hailo device or the image libraries. The default path must
return the same numbers detection produced, so the existing image build
keeps its current OCR response.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_target.config import ConfigError, PipelineConfig, parse_ocr_box_space
from ml_target.ocr_coords import box_for_response


def main() -> int:
    pixel = [0, 10, 100, 10, 100, 50, 0, 50]
    unchanged = box_for_response(pixel, 200, 100, "pixels")
    if unchanged != pixel:
        print(f"FAIL: pixel space changed the box to {unchanged}")
        return 1

    scaled = box_for_response(pixel, 200, 100, "normalized")
    expect = [0.0, 0.1, 0.5, 0.1, 0.5, 0.5, 0.0, 0.5]
    if scaled != expect:
        print(f"FAIL: scaled {scaled} != {expect}")
        return 1
    # x uses width, y uses height, including a non-square image.
    wide = box_for_response([50, 25], 200, 50, "normalized")
    if wide != [0.25, 0.5]:
        print(f"FAIL: axis scale {wide}")
        return 1
    zero = box_for_response(pixel, 0, 100, "normalized")
    if zero != [float(v) for v in pixel]:
        print(f"FAIL: zero width changed the box to {zero}")
        return 1
    if box_for_response([], 10, 10, "normalized") != []:
        print("FAIL: empty box")
        return 1
    try:
        box_for_response(pixel, 10, 10, "fractions")
    except ValueError:
        pass
    else:
        print("FAIL: unknown space was accepted")
        return 1

    if parse_ocr_box_space(None) != "pixels":
        print("FAIL: missing OCR_BOX_SPACE did not default to pixels")
        return 1
    if parse_ocr_box_space("  ") != "pixels":
        print("FAIL: blank OCR_BOX_SPACE did not default to pixels")
        return 1
    if parse_ocr_box_space("Normalized") != "normalized":
        print("FAIL: normalized was not accepted case-insensitively")
        return 1
    try:
        parse_ocr_box_space("fractions")
    except ConfigError:
        pass
    else:
        print("FAIL: config accepted an unknown OCR_BOX_SPACE")
        return 1

    # The image build leaves the variable unset. That must still construct
    # and must still select pixels. A set variable must be honoured.
    expected = parse_ocr_box_space(os.environ.get("OCR_BOX_SPACE"))
    built = PipelineConfig().ocr_box_space
    if built != expected:
        print(f"FAIL: PipelineConfig space {built} != env resolution {expected}")
        return 1
    if os.environ.get("OCR_BOX_SPACE") in (None, "") and built != "pixels":
        print(f"FAIL: unset OCR_BOX_SPACE built {built}")
        return 1
    opted = PipelineConfig(ocr_box_space="normalized").ocr_box_space
    if opted != "normalized":
        print(f"FAIL: explicit normalized built {opted}")
        return 1
    try:
        PipelineConfig(ocr_box_space="fractions")
    except ConfigError:
        pass
    else:
        print("FAIL: PipelineConfig accepted an unknown space")
        return 1

    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
