"""OCR box coordinates for the Immich response.

Kept free of OpenCV so the coordinate contract can be checked without a
Hailo device or the image libraries.
"""
from typing import List


def normalize_ocr_box(box: List[float], width: int, height: int) -> List[float]:
    """Scale one pixel quad into the 0-1 space Immich stores.

    The flat list is ``[x1, y1, x2, y2, x3, y3, x4, y4]``. Even indexes are
    x and are divided by width. Odd indexes are y and are divided by height.
    This matches Immich's OCR recognition, which divides box columns by the
    image width and height before returning them.

    Detection and the recognition crop stay in pixels. Call this only on the
    box copied into the HTTP response. A non-positive dimension is left in
    pixels so a zero-size image cannot divide by zero.
    """
    if width <= 0 or height <= 0:
        return [float(v) for v in box]
    return [float(v) / (width if i % 2 == 0 else height) for i, v in enumerate(box)]
