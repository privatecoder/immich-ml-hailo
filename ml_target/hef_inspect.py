"""Print a HEF's input/output stream info, including quantization parameters.

The quantization parameters matter: they are a property of how a particular HEF
was *compiled*, not of the model architecture, so a Hailo-8 and a Hailo-8L build
of the same model generally carry different values. Using the wrong ones does not
crash — it silently produces subtly wrong embeddings. See the Hailo-8L section of
the README.

Usage:
    python3 -m ml_target.hef_inspect /app/models/<model>.hef
"""

from typing import Any, Optional, Tuple

import hailo_platform as hpf


def read_quant(info: Any) -> Optional[Tuple[float, float]]:
    """Return (qp_scale, qp_zp) for a vstream info, or None if unavailable.

    The attribute has moved across HailoRT versions, so every access is guarded —
    an older runtime that does not expose it must degrade to printing the rest,
    not crash.
    """
    try:
        quant = getattr(info, "quant_info", None)
        if quant is None:
            return None
        scale = getattr(quant, "qp_scale", None)
        zp = getattr(quant, "qp_zp", None)
        if scale is None or zp is None:
            return None
        return float(scale), float(zp)
    except Exception:
        return None


def _fmt_quant(info: Any) -> str:
    q = read_quant(info)
    if q is None:
        return "quant=<not exposed by this HailoRT>"
    return f"qp_scale={q[0]:.17g} qp_zp={q[1]:.17g}"


def _describe(info: Any) -> str:
    parts = [str(getattr(info, "name", "?"))]
    try:
        parts.append(f"shape={info.shape}")
    except Exception:
        pass
    try:
        f = info.format
        parts.append(f"type={f.type} order={f.order} flags={f.flags}")
    except Exception:
        pass
    parts.append(_fmt_quant(info))
    return "  ".join(parts)


def _dump(hef: Any, which: str) -> None:
    """Print every stream of one kind, degrading rather than crashing.

    The enumeration call itself is guarded, not just the per-stream attribute
    access: a HailoRT that renamed or removed these methods must still let the
    tool print whatever else it can.
    """
    try:
        infos = (hef.get_input_vstream_infos() if which == "input"
                 else hef.get_output_vstream_infos())
    except Exception as exc:
        print(f"  <could not enumerate {which} streams: {exc}>")
        return

    if not infos:
        print(f"  <no {which} streams reported>")
        return

    for info in infos:
        print(" ", _describe(info))


def main() -> None:
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 -m ml_target.hef_inspect /path/to/model.hef")
        raise SystemExit(2)

    hef_path = sys.argv[1]
    hef = hpf.HEF(hef_path)

    print("HEF:", hef_path)

    print("\n=== INPUTS ===")
    _dump(hef, "input")

    print("\n=== OUTPUTS ===")
    _dump(hef, "output")

    print(
        "\nNote: qp_scale/qp_zp above are specific to THIS compiled HEF.\n"
        "The pipeline reads them from the HEF at startup, so they normally need\n"
        "no manual action. The constants in config.py are only a fallback for\n"
        "HailoRT versions that do not expose quantization info."
    )


if __name__ == "__main__":
    main()
