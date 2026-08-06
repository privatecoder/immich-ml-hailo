"""Hailo model wrapper: load HEF, configure network groups, run inference.

Uses the InferVStreams API (stable across HailoRT 4.x). The newer InferModel API
(available since HailoRT 4.17+) is simpler but has a different batching model:

    infer_model = vdevice.create_infer_model(hef)
    infer_model.output().set_format_type(FormatType.FLOAT32)
    configured = infer_model.configure()
    bindings = configured.create_bindings()
    bindings.input().set_buffer(input_data)
    bindings.output().set_buffer(output_buffer)
    configured.run(bindings)  # or run_async()

Migration to InferModel would simplify activate_model() since it manages
activation internally, but requires testing with actual hardware.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

import numpy as np
import hailo_platform as hpf

LOG = logging.getLogger("ml_target.models")


@dataclass(frozen=True)
class QuantParams:
    """Quantization parameters for one vstream, as baked into a HEF.

    These describe a specific *compilation*, not the model architecture: a
    Hailo-8 and a Hailo-8L build of the same network generally carry different
    values. Applying the wrong ones does not raise — it silently yields subtly
    wrong numbers — which is why they are read from the HEF rather than
    hardcoded.
    """
    qp_scale: float
    qp_zp: float


@dataclass
class HailoModel:
    """Wraps a configured Hailo network group with its vstream parameters."""
    name: str
    hef_path: str
    ng: Any  # hailo_platform NetworkGroup
    in_params: Dict[str, Any]
    out_params: Dict[str, Any]
    in_key: str
    input_format: hpf.FormatType
    output_format: hpf.FormatType
    # Read from the HEF when this HailoRT exposes them; None otherwise, in which
    # case callers fall back to the constants in config.py.
    input_quant: Optional[QuantParams] = None
    output_quant: Optional[QuantParams] = None
    # Device batch size actually configured, or None if left at HailoRT's
    # default. Callers size their infer() calls from this.
    batch_size: Optional[int] = None


def configure_model(
    vdevice: hpf.VDevice,
    hef_path: str,
    *,
    input_format: Optional[hpf.FormatType] = None,
    output_format: hpf.FormatType = hpf.FormatType.FLOAT32,
    batch_size: Optional[int] = None,
) -> HailoModel:
    """Load a HEF and configure it on the given VDevice.

    `batch_size` sets how many frames the device processes per batch. Pass None
    to leave HailoRT's default in place. Only worth setting for models that
    actually receive multi-frame input.
    """
    hef = hpf.HEF(hef_path)

    if input_format is None:
        input_format = _guess_input_format(hef)

    cfg = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
    applied_batch = _apply_batch_size(cfg, batch_size, hef_path)
    ng = vdevice.configure(hef, cfg)[0]

    in_params = hpf.InputVStreamParams.make_from_network_group(ng, format_type=input_format)
    out_params = hpf.OutputVStreamParams.make_from_network_group(ng, format_type=output_format)

    if not isinstance(in_params, dict) or len(in_params) != 1:
        raise RuntimeError(f"Expected exactly 1 input vstream, got: {list(in_params.keys())}")

    in_key = next(iter(in_params.keys()))
    name = in_key.split("/")[0] if "/" in in_key else in_key

    input_quant = _read_stream_quant(hef, "input", hef_path)
    output_quant = _read_stream_quant(hef, "output", hef_path)

    LOG.info("Configured model: %s", hef_path)
    LOG.info("  in_key=%s  out_keys=%s", in_key, list(out_params.keys()))
    LOG.info("  in_format=%s  out_format=%s", input_format, output_format)
    LOG.info("  in_quant=%s  out_quant=%s", _fmt_quant(input_quant), _fmt_quant(output_quant))
    LOG.info("  batch_size=%s", applied_batch if applied_batch else "<HailoRT default>")

    return HailoModel(
        name=name,
        hef_path=hef_path,
        ng=ng,
        in_params=in_params,
        out_params=out_params,
        in_key=in_key,
        input_format=input_format,
        output_format=output_format,
        input_quant=input_quant,
        output_quant=output_quant,
        batch_size=applied_batch,
    )


def _apply_batch_size(
    cfg_params: Any, batch_size: Optional[int], hef_path: str
) -> Optional[int]:
    """Set batch_size on every network group in the configure params.

    Where this belongs is not a guess. Per the HailoRT 4.23.0 API reference,
    batch_size is a member of `hailo_configure_network_group_params_t` — which
    is what `ConfigureParams.create_from_hef()` returns, keyed by network-group
    name. The vstream params have no such field: `InputVStreamParams` /
    `OutputVStreamParams.make_from_network_group()` accept only
    `format_type`, `timeout_ms`, `queue_size` and `network_name`.

    The reference also warns against setting both the network-group batch size
    and the per-network one (`hailo_network_parameters_t.batch_size`) — "Not
    both". We want one value for the whole group, so the network-group
    parameter is the documented choice and the per-network one is left alone.

    Caveat worth knowing: the reference notes that the network-group batch_size
    "is only used in multi-context network_groups". For a single-context HEF
    this is accepted and ignored. `hailortcli parse-hef <model>.hef` reports the
    context count if a model shows no improvement.

    Returns the value actually applied, or None.
    """
    if not batch_size or batch_size < 1:
        return None

    applied = None
    try:
        names = list(cfg_params)
    except Exception as exc:
        LOG.warning("batch_size: cannot enumerate configure params for %s: %s", hef_path, exc)
        return None

    for name in names:
        try:
            cfg_params[name].batch_size = batch_size
            applied = batch_size
        except Exception as exc:
            LOG.warning(
                "batch_size: could not set %d on network group %r of %s: %s "
                "— leaving HailoRT's default",
                batch_size, name, hef_path, exc,
            )
    return applied


def _read_quant(info: Any) -> Optional[QuantParams]:
    """Extract quantization params from one vstream info, or None.

    Every access is guarded: the attribute name has varied across HailoRT
    versions, and a runtime that does not expose it must degrade to None rather
    than break model configuration.
    """
    try:
        quant = getattr(info, "quant_info", None)
        if quant is None:
            return None
        scale = getattr(quant, "qp_scale", None)
        zp = getattr(quant, "qp_zp", None)
        if scale is None or zp is None:
            return None
        return QuantParams(float(scale), float(zp))
    except Exception:
        return None


def _read_stream_quant(hef: hpf.HEF, which: str, hef_path: str) -> Optional[QuantParams]:
    """Read quant params for a HEF's single input or output vstream.

    Returns None when the runtime does not expose quantization info, or when the
    HEF has multiple streams of that kind — SCRFD has six outputs, and there is
    no single set of params to attribute to the model. Those models all request
    FLOAT32 output, so HailoRT dequantizes on-device and the values are unused.
    """
    try:
        infos = (hef.get_input_vstream_infos() if which == "input"
                 else hef.get_output_vstream_infos())
        if not infos or len(infos) != 1:
            return None
        return _read_quant(infos[0])
    except Exception as exc:
        LOG.warning("Could not read %s quantization info from %s: %s", which, hef_path, exc)
        return None


def _fmt_quant(q: Optional[QuantParams]) -> str:
    if q is None:
        return "<none>"
    return f"scale={q.qp_scale:.17g} zp={q.qp_zp:.17g}"


def _guess_input_format(hef: hpf.HEF) -> hpf.FormatType:
    try:
        infos = hef.get_input_vstream_infos()
        if infos and len(infos) == 1:
            s = str(
                getattr(infos[0], "format", None)
                or getattr(infos[0], "data_type", None)
                or getattr(infos[0], "dtype", None)
            ).lower()
            if "uint16" in s:
                return hpf.FormatType.UINT16
            if "float" in s:
                return hpf.FormatType.FLOAT32
    except Exception:
        pass
    return hpf.FormatType.UINT8


def validate_input(xb: np.ndarray) -> np.ndarray:
    if xb.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D input, got {xb.shape}")
    if xb.dtype not in (np.uint8, np.float32, np.uint16):
        raise ValueError(f"Expected uint8/float32/uint16, got {xb.dtype}")
    if not xb.flags["C_CONTIGUOUS"]:
        xb = np.ascontiguousarray(xb)
    return xb


@contextmanager
def activate_model(model: HailoModel) -> Generator:
    """Activate a network group for the duration of the context.

    Yields a function: infer(xb) -> Dict[str, np.ndarray].
    All inferences within the context share a single activation cycle.
    """
    with model.ng.activate(model.ng.create_params()):
        with hpf.InferVStreams(model.ng, model.in_params, model.out_params) as pipe:
            def _infer(xb: np.ndarray) -> Dict[str, np.ndarray]:
                xb = validate_input(xb)
                return pipe.infer({model.in_key: xb})
            yield _infer


def infer_single(model: HailoModel, xb: np.ndarray) -> Dict[str, np.ndarray]:
    """Convenience: activate model, run one inference, deactivate."""
    xb = validate_input(xb)
    with activate_model(model) as infer:
        return infer(xb)


def pick_output(outputs: Dict[str, np.ndarray], hint: Optional[str] = None) -> np.ndarray:
    """Select the main output tensor from an inference result dict.

    If there's only one output, return it. Otherwise use `hint` to find a matching key.
    """
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    if hint:
        for k in outputs:
            if hint in k:
                return outputs[k]
    return next(iter(outputs.values()))
