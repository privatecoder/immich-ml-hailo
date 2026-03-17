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


def configure_model(
    vdevice: hpf.VDevice,
    hef_path: str,
    *,
    input_format: Optional[hpf.FormatType] = None,
    output_format: hpf.FormatType = hpf.FormatType.FLOAT32,
) -> HailoModel:
    """Load a HEF and configure it on the given VDevice."""
    hef = hpf.HEF(hef_path)

    if input_format is None:
        input_format = _guess_input_format(hef)

    cfg = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
    ng = vdevice.configure(hef, cfg)[0]

    in_params = hpf.InputVStreamParams.make_from_network_group(ng, format_type=input_format)
    out_params = hpf.OutputVStreamParams.make_from_network_group(ng, format_type=output_format)

    if not isinstance(in_params, dict) or len(in_params) != 1:
        raise RuntimeError(f"Expected exactly 1 input vstream, got: {list(in_params.keys())}")

    in_key = next(iter(in_params.keys()))
    name = in_key.split("/")[0] if "/" in in_key else in_key

    LOG.info("Configured model: %s", hef_path)
    LOG.info("  in_key=%s  out_keys=%s", in_key, list(out_params.keys()))
    LOG.info("  in_format=%s  out_format=%s", input_format, output_format)

    return HailoModel(
        name=name,
        hef_path=hef_path,
        ng=ng,
        in_params=in_params,
        out_params=out_params,
        in_key=in_key,
        input_format=input_format,
        output_format=output_format,
    )


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
