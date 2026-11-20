"""Shared setup for experiment scripts: threads, provenance, model loading, IO."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def setup_threads(n: int | None = None) -> int:
    """Pin intra-op threads (default: physical cores via psutil)."""
    if not n:
        try:
            import psutil
            n = psutil.cpu_count(logical=False) or 1
        except Exception:
            n = max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(n)
    return n


def _git() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "ttcompress", "experiments"))}


def env_info() -> dict:
    cpu = platform.processor()
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=20)
        cpu = out.stdout.strip() or cpu
    except Exception:
        pass
    import numpy
    info = {"cpu": cpu, "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "numpy": numpy.__version__, "threads": torch.get_num_threads(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(), "git": _git(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    info["power"] = power_state()
    for mod in ("timm", "onnxruntime", "onnx"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


def power_state() -> dict:
    """AC/battery state and active power scheme (Windows). Latency measured
    on battery is not comparable and is flagged by the benchmark script."""
    out = {"on_ac": None, "battery_pct": None, "scheme": None}
    try:
        import psutil
        b = psutil.sensors_battery()
        if b is not None:
            out["on_ac"], out["battery_pct"] = bool(b.power_plugged), round(b.percent, 1)
    except Exception:
        pass
    try:
        r = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10)
        out["scheme"] = r.stdout.strip().split("(")[-1].rstrip(")") if "(" in r.stdout else r.stdout.strip()
    except Exception:
        pass
    return out


def checkpoint_sha256(model_name: str) -> str | None:
    """SHA-256 of the pretrained weight file in the HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        repo = "timm/" + model_name
        for fn in ("model.safetensors", "pytorch_model.bin"):
            p = try_to_load_from_cache(repo, fn)
            if isinstance(p, str) and os.path.exists(p):
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 22), b""):
                        h.update(chunk)
                return h.hexdigest()
    except Exception:
        pass
    return None


def load_model(name: str):
    """Load a pinned timm checkpoint (full name incl. pretrained tag) and its
    own data config. Refuses untagged names (audit H1)."""
    import timm
    from timm.data import resolve_data_config
    if "." not in name:
        raise ValueError(f"pin the pretrained tag explicitly, e.g. '{name}.fb_in1k' (audit H1)")
    model = timm.create_model(name, pretrained=True).eval()
    cfg = resolve_data_config({}, model=model)
    if model.num_classes != 1000:
        raise ValueError(f"{name} has {model.num_classes} classes; expected an ImageNet-1k head")
    return model, cfg


def model_card(name: str, model, cfg) -> dict:
    return {"name": name, "sha256": checkpoint_sha256(name), "params": sum(p.numel() for p in model.parameters()),
            "num_classes": model.num_classes, "input_size": list(cfg["input_size"]), "mean": list(cfg["mean"]),
            "std": list(cfg["std"]), "crop_pct": cfg.get("crop_pct"), "interpolation": cfg.get("interpolation")}


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)
    return path


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()
