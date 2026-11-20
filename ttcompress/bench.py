"""
Latency benchmarking that survives scrutiny on a laptop CPU.

The earlier scripts timed the compressed model for 5 iterations and *then*
the baseline, so thermal/boost-clock drift and background load landed on
one side only. Here all candidates are timed in interleaved rounds
(A B C A B C ...), randomised order per round, with warm-up, and we report
the median, IQR and a bootstrap 95% interval of each candidate's speed-up
over the reference.
"""

from __future__ import annotations

import io
import random
import statistics
import time
from typing import Callable, Dict

import torch
import torch.nn as nn


def model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 2 ** 20


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _bootstrap_ratio_ci(ref, cand, n_boot=2000, seed=0):
    rng = random.Random(seed)
    ratios = []
    for _ in range(n_boot):
        r = statistics.median(rng.choices(ref, k=len(ref)))
        c = statistics.median(rng.choices(cand, k=len(cand)))
        ratios.append(r / c)
    ratios.sort()
    return ratios[int(0.025 * n_boot)], ratios[int(0.975 * n_boot)]


_FLUSH_BUF = None


def flush_cpu_cache(mb: int = 64) -> None:
    """Evict caches by streaming through a buffer larger than the LLC
    (Ryzen 7 7730U: 16 MB L3). Used for cold-weight microbenchmarks."""
    global _FLUSH_BUF
    if _FLUSH_BUF is None or _FLUSH_BUF.numel() * 4 < mb * 2 ** 20:
        _FLUSH_BUF = torch.empty(mb * 2 ** 20 // 4)
    _FLUSH_BUF.add_(1.0)


def _stats(t, ref, name, reference, batch):
    t_sorted = sorted(t)
    q = statistics.quantiles(t_sorted, n=20)  # 5% steps
    lo, hi = _bootstrap_ratio_ci(ref, t) if name != reference else (1.0, 1.0)
    med = statistics.median(t_sorted)
    return {"n": len(t), "median_ms": round(med, 3), "mean_ms": round(statistics.fmean(t), 3),
            "std_ms": round(statistics.stdev(t), 3) if len(t) > 1 else 0.0,
            "p25_ms": round(q[4], 3), "p75_ms": round(q[14], 3), "p90_ms": round(q[17], 3), "p95_ms": round(q[18], 3),
            "throughput_per_s": round(batch / med * 1e3, 2) if batch else None,
            "speedup": round(statistics.median(ref) / med, 4), "speedup_ci": [round(lo, 4), round(hi, 4)]}


@torch.inference_mode()
def compare_latency(fns: Dict[str, Callable[[], object]], reference: str, rounds: int = 30, warmup: int = 5,
                    seed: int = 0, batch: int | None = None, cold: bool = False) -> Dict[str, dict]:
    """Time each zero-arg callable in interleaved, shuffled rounds.

    speedup = median(reference) / median(name) (>1 means faster), with a
    bootstrap 95% interval over rounds. `cold=True` evicts the CPU caches
    before every timed call (the eviction itself is not timed)."""
    names = list(fns)
    for _ in range(warmup):
        for n in names:
            fns[n]()
    times = {n: [] for n in names}
    rng = random.Random(seed)
    for _ in range(rounds):
        order = names[:]
        rng.shuffle(order)
        for n in order:
            if cold:
                flush_cpu_cache()
            t0 = time.perf_counter()
            fns[n]()
            times[n].append((time.perf_counter() - t0) * 1e3)
    ref = times[reference]
    return {n: _stats(times[n], ref, n, reference, batch) for n in names}


class PeakRSS:
    """Context manager sampling the process RSS every `interval` s in a
    background thread; `.peak_mb` and `.base_mb` after exit."""

    def __init__(self, interval: float = 0.005):
        import psutil
        import threading
        self._p = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.base_mb = self.peak_mb = 0.0

    def _run(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._p.memory_info().rss / 2 ** 20)
            self._stop.wait(self._interval)

    def __enter__(self):
        self.base_mb = self.peak_mb = self._p.memory_info().rss / 2 ** 20
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, self._p.memory_info().rss / 2 ** 20)


def count_macs(model: nn.Module, example: torch.Tensor) -> int:
    """Multiply-accumulates of one forward pass, from torch's FLOP counter
    (FLOPs / 2), for FP32 models (quantized kernels are not counted).

    Fused scaled_dot_product_attention is *not* counted by FlopCounterMode on
    CPU, which silently drops the QK^T and AV products; the math SDPA backend
    is forced so they decompose into counted matmuls."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as fc, torch.inference_mode():
        model(example)
    return int(fc.get_total_flops() // 2)


@torch.no_grad()
def benchmark_latency(model: nn.Module, example_input, n_warmup: int = 3, n_iters: int = 15) -> Dict[str, float]:
    """Single-model timing (kept for the legacy experiment scripts). Prefer
    compare_latency for any A-vs-B claim."""
    model.eval()
    for _ in range(n_warmup):
        model(example_input)
    t = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        model(example_input)
        t.append(time.perf_counter() - t0)
    t = torch.tensor(t)
    return {"mean_ms": float(t.mean() * 1000), "p50_ms": float(t.median() * 1000),
            "p95_ms": float(t.quantile(0.95) * 1000),
            "throughput_samples_per_s": float(example_input.shape[0] / t.mean())}
