"""OCR box coordinates for the Immich response.

Kept free of OpenCV so the coordinate contract can be checked without a
Hailo device or the image libraries.

The worker has two response spaces, selected by ``OCR_BOX_SPACE``:

* ``pixels`` is the default and the quad detection already returned.
* ``normalized`` is the 0-1 space Immich stores.

Detection and the recognition crop stay in pixels in both cases.
"""
from typing import List


def box_for_response(
    box: List[float], width: int, height: int, space: str
) -> List[float]:
    """Copy one quad into the response space named by ``space``.

    ``pixels`` returns the same numbers detection produced. ``normalized``
    scales that copy. Any other name is refused here as well as at startup,
    so a typo cannot silently pick a space.
    """
    if space == "pixels":
        return list(box)
    if space == "normalized":
        return normalize_ocr_box(box, width, height)
    raise ValueError(
        f"OCR box space {space!r} is not known. Use 'pixels' or 'normalized'."
    )


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
