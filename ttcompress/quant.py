"""
INT8 baselines and combinations (the comparison the original brief promised).

* ``dynamic_int8``: torch dynamic quantization of every nn.Linear (weights
  INT8, activations
  quantized on the fly). LowRankLinear is built from nn.Linear, so
  low-rank + INT8 composes automatically.
* ``export_onnx`` / ``ort_session`` / ``ort_dynamic_int8``: the ONNX Runtime
  path, which is how a CPU edge deployment would actually ship.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn


def dynamic_int8(model: nn.Module) -> nn.Module:
    from torch.ao.quantization import quantize_dynamic

    engines = torch.backends.quantized.supported_engines
    for eng in ("x86", "fbgemm", "onednn"):
        if eng in engines:
            torch.backends.quantized.engine = eng
            break
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def export_onnx(model: nn.Module, example: torch.Tensor, path: str | Path, opset: int = 17) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (example,), str(path), opset_version=opset, input_names=["x"],
                          output_names=["logits"], dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
                          dynamo=False)
    return path


def ort_dynamic_int8(fp32_path: str | Path, out_path: str | Path, per_channel: bool = True) -> Path:
    """ONNX Runtime dynamic quantization: weights INT8 (per output channel if
    `per_channel`), activations quantized per tensor at run time."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Only MatMul (= nn.Linear): matches torch dynamic INT8 scope; ORT CPU has no
    # ConvInteger kernel, so quantizing the patch-embedding Conv breaks the session.
    quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8, per_channel=per_channel,
                     op_types_to_quantize=["MatMul"])
    return Path(out_path)


def onnx_op_histogram(path: str | Path) -> dict:
    """Operator counts of an ONNX graph, plus how many MatMul/Gemm remain in
    floating point. Used to *verify* that low-rank factors were quantized
    rather than assuming it."""
    import collections
    import onnx

    g = onnx.load(str(path), load_external_data=False).graph
    ops = collections.Counter(n.op_type for n in g.node)
    float_mm = ops.get("MatMul", 0) + ops.get("Gemm", 0)
    int_mm = ops.get("MatMulInteger", 0) + ops.get("DynamicQuantizeMatMul", 0) + ops.get("QLinearMatMul", 0)         + ops.get("MatMulIntegerToFloat", 0)
    return {"ops": dict(ops), "float_matmuls": float_mm, "int8_matmuls": int_mm}


def torch_int8_coverage(model: nn.Module) -> dict:
    """Count Linear layers that are / are not dynamically quantized."""
    from torch.ao.nn.quantized.dynamic import Linear as QLinear
    q = sum(1 for m in model.modules() if isinstance(m, QLinear))
    f = sum(1 for m in model.modules() if type(m) is nn.Linear)
    return {"int8_linear": q, "fp32_linear": f}


def fake_quant_weight(W: torch.Tensor, per_channel: bool = True, bits: int = 8) -> torch.Tensor:
    """Symmetric round-to-nearest weight quantization, returned dequantized.
    Matches the weight scheme of torch/ORT dynamic INT8 (symmetric, max-abs
    scale) closely enough to study operation order (quantize->factorize vs
    factorize->quantize); it is not bit-exact with either backend."""
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=1, keepdim=True) if per_channel else W.abs().max()
    scale = (amax / qmax).clamp_min(1e-12)
    return (W / scale).round().clamp(-qmax, qmax) * scale


def ort_session(path: str | Path, threads: int | None = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def file_size_mb(path: str | Path) -> float:
    return os.path.getsize(path) / 2 ** 20
