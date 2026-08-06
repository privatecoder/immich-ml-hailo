#!/usr/bin/env python3
"""Verify OCR text/box index alignment using a position-encoding grid image.

Each token in ocr-align.jpg encodes its own grid cell ("R3C5" = row 3, col 5).
So for every returned region we can compute which cell its BOX sits in, and
check that the TEXT reported at the same index names that cell. If chunked
batching shifted indices, text[i] will name a different cell than box[i] is in
— and this prints exactly which.

Usage:  python3 tests/verify_ocr_align.py response.json
"""
import json
import re
import sys
from collections import Counter

W, H, COLS, ROWS = 1920, 1088, 8, 8


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def cell_of(cx, cy):
    col = min(COLS, max(1, int(cx / (W / COLS)) + 1))
    row = min(ROWS, max(1, int(cy / (H / ROWS)) + 1))
    return row, col


def main():
    raw = open(sys.argv[1]).read()
    try:
        resp = json.loads(raw)
    except Exception as exc:
        print(f"FAIL: response is not JSON ({exc})\n--- raw ---\n{raw[:600]}")
        return 1

    ocr = resp.get("ocr")
    if ocr is None:
        print("FAIL: no 'ocr' key in the response.")
        if "error" in resp:
            print(f"  The worker rejected the request: {resp['error']}")
            print("  Almost always a malformed 'entries' payload — check that the JSON "
                  "survived shell quoting intact (a line break inside it will do this).")
        print(f"--- raw response ---\n{raw[:600]}")
        return 1
    if not isinstance(ocr, dict) or not isinstance(ocr.get("text"), list):
        print(f"FAIL: unexpected ocr payload: {ocr}")
        return 1
    if "error" in ocr:
        print(f"FAIL: worker returned an error: {ocr['error']}")
        return 1

    texts, boxes = ocr["text"], ocr["box"]
    box_scores, text_scores = ocr["boxScore"], ocr["textScore"]
    n = len(texts)

    print(f"regions returned: {n}   (image contains {ROWS * COLS})")

    # --- structural checks: these are alignment by definition -----------------
    ok = True
    if not (len(box_scores) == len(text_scores) == n):
        print(f"FAIL: list lengths disagree — text={n} boxScore={len(box_scores)} "
              f"textScore={len(text_scores)}")
        ok = False
    if len(boxes) != n * 8:
        print(f"FAIL: box has {len(boxes)} values; expected {n * 8} (8 per region)")
        ok = False
    if not ok:
        return 1
    print("structural: list lengths and box arity consistent")

    # --- positional check: does text[i] name the cell box[i] sits in? ---------
    expected_by_cell = {(r, c): f"R{r}C{c}" for r in range(1, ROWS + 1)
                        for c in range(1, COLS + 1)}
    token_to_cell = {v: k for k, v in expected_by_cell.items()}

    exact = noisy = shifted = offgrid = 0
    problems = []
    seen = Counter()

    for i, text in enumerate(texts):
        pts = boxes[i * 8:(i + 1) * 8]
        cx = sum(pts[0::2]) / 4.0
        cy = sum(pts[1::2]) / 4.0
        row, col = cell_of(cx, cy)
        want = expected_by_cell[(row, col)]
        got = norm(text)
        seen[got] += 1

        if got == want:
            exact += 1
        elif got in token_to_cell:
            # Reads as a *different* valid cell — the signature of a shift.
            shifted += 1
            r2, c2 = token_to_cell[got]
            problems.append(
                f"  [{i:2d}] box in {want} (center {cx:6.0f},{cy:5.0f}) but text says "
                f"{got}  <-- names cell R{r2}C{c2}, offset {r2 - row:+d} row {c2 - col:+d} col")
        elif len(got) == len(want) and sum(a != b for a, b in zip(got, want)) <= 1:
            noisy += 1  # single-character OCR misread, still the right cell
        else:
            offgrid += 1
            problems.append(f"  [{i:2d}] box in {want} but text is {text!r} (unrecognized)")

    print(f"positional: exact={exact}  ocr-noise={noisy}  "
          f"SHIFTED={shifted}  unreadable={offgrid}")

    dupes = [t for t, c in seen.items() if c > 1 and t in token_to_cell]
    if dupes:
        print(f"  note: token(s) reported more than once: {sorted(dupes)}")

    if problems:
        print("\ndetail:")
        for p in problems[:25]:
            print(p)
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")

    print()
    if shifted:
        print("VERDICT: MISALIGNED — text is attached to the wrong boxes. "
              "This is the batching bug; do not ship.")
        return 1
    if exact + noisy == n and n >= 32:
        print(f"VERDICT: ALIGNED — every region's text names its own cell, "
              f"across {n} regions (chunk size 32, so boundaries were crossed).")
        return 0
    if n < 32:
        print(f"VERDICT: INCONCLUSIVE — only {n} regions detected, so the 32-crop "
              f"chunk boundary was never crossed. Alignment looks right but is untested.")
        return 0
    print("VERDICT: ALIGNED, but some regions were unreadable — see detail above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
