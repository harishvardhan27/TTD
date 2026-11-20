"""Tests for the Phase-1 research infrastructure: error metrics, allocation,
data protocol, quantization/ONNX, distillation, runner config expansion."""

import itertools
import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ttcompress import data as D
from ttcompress.lowrank import (LowRankLinear, allocate_ranks, allocate_ranks_dp, compress_lowrank, factorize,
                                max_useful_rank, output_rel_error, weight_rel_error, PREDICATES, find_linears)
from ttcompress.quant import fake_quant_weight

ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(0)


def aniso_inputs(n, N=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g) * torch.logspace(0, -2, n)
    return torch.randn(N, n, generator=g) @ A.T


# ------------------------------------------------------------ error metrics
@pytest.mark.parametrize("damp", [0.0, 1e-2])
def test_output_rel_error_matches_direct_measurement(damp):
    W = torch.randn(30, 20)
    X = aniso_inputs(20)
    C = X.T @ X / len(X)
    for cov in (None, C):
        U, V = factorize(W, cov, damp=damp).factors(6)
        direct = float(((W - U @ V) @ X.T).norm() / (W @ X.T).norm())
        assert abs(output_rel_error(W, U @ V, C) - direct) < 1e-4


def test_predicted_error_exact_only_without_damping_and_data_aware_wins_on_heldout():
    W = torch.randn(40, 32)
    X_fit, X_ho = aniso_inputs(32, seed=1), aniso_inputs(32, seed=1)[:0]
    X_ho = aniso_inputs(32, N=3000, seed=1)  # same distribution
    C_fit = X_fit.T @ X_fit / len(X_fit)
    C_ho = X_ho.T @ X_ho / len(X_ho)
    f0 = factorize(W, C_fit, damp=0.0)
    U, V = f0.factors(8)
    assert abs(f0.rel_error(8) - output_rel_error(W, U @ V, C_fit)) < 1e-6
    fd = factorize(W, C_fit, damp=1e-2)
    Ud, Vd = fd.factors(8)
    # with damping the prediction is for C + damp*I, not C: not exact
    assert abs(fd.rel_error(8) - output_rel_error(W, Ud @ Vd, C_fit)) > 1e-6
    Up, Vp = factorize(W).factors(8)
    assert output_rel_error(W, Ud @ Vd, C_ho) < output_rel_error(W, Up @ Vp, C_ho)
    assert weight_rel_error(W, Up @ Vp) <= weight_rel_error(W, Ud @ Vd) + 1e-9  # Eckart-Young


# ------------------------------------------------------------ allocation
def _objective(facs, alloc):
    return sum(facs[n].rel_error(r) ** 2 if r else 0.0 for n, r in alloc.items())


def _cost(shapes, alloc):
    return sum(r * (o + i) if r else o * i for (n, r), (o, i) in zip(alloc.items(), (shapes[n] for n in alloc)))


def make_problem(seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = {"a": (48, 32), "b": (32, 32), "c": (64, 16)}
    facs = {}
    for k, (o, i) in shapes.items():
        decay = torch.logspace(0, -float(torch.rand(1, generator=g) * 3), min(o, i))
        U, _ = torch.linalg.qr(torch.randn(o, min(o, i), generator=g))
        V, _ = torch.linalg.qr(torch.randn(i, min(o, i), generator=g))
        facs[k] = factorize(U @ torch.diag(decay) @ V.T)
    return facs, shapes


def test_max_useful_rank():
    assert max_useful_rank(768, 768, 0.0, 8) == 376  # break-even 383
    assert max_useful_rank(768, 768, 0.1, 8) == 344  # 0.9*768*768/1536 = 345.6
    assert max_useful_rank(768, 768, 0.1, 8) * 1536 <= 0.9 * 768 * 768


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dp_is_exact_and_greedy_is_feasible(seed):
    facs, shapes = make_problem(seed)
    dense = sum(o * i for o, i in shapes.values())
    budget = int(0.5 * dense)
    gr = allocate_ranks(facs, shapes, budget, granularity=4)
    assert _cost(shapes, gr) <= budget
    dp = allocate_ranks_dp(facs, shapes, budget, granularity=4, allow_dense=False, unit=1)
    assert _cost(shapes, dp) <= budget
    # brute force over all rank combinations
    opts = {n: list(range(4, max_useful_rank(i, o, 0.0, 4) + 1, 4)) for n, (o, i) in shapes.items()}
    best = min((_objective(facs, dict(zip(opts, combo))) for combo in itertools.product(*opts.values())
                if _cost(shapes, dict(zip(opts, combo))) <= budget))
    assert abs(_objective(facs, dp) - best) < 1e-12
    assert _objective(facs, gr) >= best - 1e-12


def test_greedy_rejects_infeasible_budget():
    facs, shapes = make_problem()
    with pytest.raises(ValueError):
        allocate_ranks(facs, shapes, 10, granularity=8)


def test_layer_scopes_on_vit_names():
    import timm
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    count = {k: len(find_linears(m, p)) for k, p in PREDICATES.items()}
    assert count == {"qkv": 12, "attn_out": 12, "attn": 24, "mlp": 24, "attn+mlp": 48, "all": 49}


# ------------------------------------------------------------ data protocol
V2 = D.DATA_ROOT / D.V2_DIR


@pytest.mark.skipif(not V2.exists(), reason="ImageNetV2 not downloaded")
def test_v2_split_is_disjoint_stratified_and_deterministic():
    dev, test = D.get_items("v2_dev"), D.get_items("v2_test")
    assert len(dev) == 2000 and len(test) == 8000
    assert not ({p for _, p in dev} & {p for _, p in test})
    assert torch.bincount(torch.tensor([c for c, _ in dev]), minlength=1000).eq(2).all()
    assert D.get_items("v2_dev") == dev


def test_role_guard_blocks_eval_data_in_training():
    with pytest.raises(ValueError):
        D.get_items("v2_test", role="train")
    with pytest.raises(ValueError):
        D.get_items("imagenette_train", role="eval")


# ------------------------------------------------------------ quantization / ONNX
def test_fake_quant_error_bound():
    W = torch.randn(64, 128)
    Wq = fake_quant_weight(W, per_channel=True)
    step = W.abs().amax(1, keepdim=True) / 127
    assert ((W - Wq).abs() <= step / 2 + 1e-6).all()
    assert len(torch.unique(Wq[0])) <= 255


@pytest.mark.slow
def test_onnx_lowrank_int8_graph_and_parity(tmp_path):
    pytest.importorskip("onnxruntime")
    import numpy as np
    import timm
    from ttcompress import quant as Q
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False).eval()
    rep = compress_lowrank(m, PREDICATES["attn+mlp"], rank=32)
    n_lr = sum(1 for l in rep.layers if l["rank"])
    x = torch.randn(2, 3, 224, 224)
    f = Q.export_onnx(m, x, tmp_path / "m.onnx")
    q = Q.ort_dynamic_int8(f, tmp_path / "q.onnx")
    h = Q.onnx_op_histogram(q)
    assert h["int8_matmuls"] == 2 * n_lr + 1          # two factors per layer + head
    assert h["float_matmuls"] == 2 * 12               # QK^T and AV per block only
    ref = m(x).detach().numpy()
    y = Q.ort_session(f).run(None, {"x": x.numpy()})[0]
    assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-4
    Q.ort_session(q).run(None, {"x": x.numpy()})      # int8 graph executes
    assert Q.file_size_mb(q) < 0.5 * Q.file_size_mb(f)


def test_torch_int8_quantizes_lowrank_factors():
    from ttcompress.quant import dynamic_int8, torch_int8_coverage
    m = nn.Sequential(LowRankLinear(64, 64, 8), nn.Linear(64, 10))
    q = dynamic_int8(m)
    assert torch_int8_coverage(q) == {"int8_linear": 3, "fp32_linear": 0}


# ------------------------------------------------------------ distillation
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Linear(3 * 8 * 8, 32)
        self.ln = nn.LayerNorm(32)
        self.head = nn.Linear(32, 5)

    def forward(self, x):
        return self.head(self.ln(self.f(x.flatten(1))))


def _toy_distill(steps, ckpt, crash_at_eval=None):
    from ttcompress.distill import distill
    from ttcompress.data import normalize
    torch.manual_seed(0)
    teacher = TinyNet().eval()
    student = TinyNet()
    student.f = LowRankLinear.from_linear(teacher.f, 4)
    student.ln.load_state_dict(teacher.ln.state_dict())
    student.head.load_state_dict(teacher.head.state_dict())
    x = (torch.rand(64, 3, 8, 8) * 255).to(torch.uint8)
    cfg = {"mean": (0.5, 0.5, 0.5), "std": (0.25, 0.25, 0.25)}
    with torch.no_grad():
        tl = teacher(normalize(x, cfg))
    calls = {"n": 0}

    def eval_fn(m):
        calls["n"] += 1
        if crash_at_eval is not None and calls["n"] == crash_at_eval:
            raise KeyboardInterrupt("simulated interruption")
        with torch.no_grad():
            return -float(torch.nn.functional.mse_loss(m(normalize(x, cfg)), tl))

    res = distill(student, x, tl, cfg, steps=steps, labels=tl.argmax(1), alpha=0.3, batch_size=8, accum=1,
                  lr=1e-2, eval_every=5, ckpt_path=ckpt, log=lambda s: None, eval_fn=eval_fn)
    return student, res


def test_distillation_improves_and_resume_is_exact(tmp_path):
    s_full, r_full = _toy_distill(20, tmp_path / "a.pt")
    assert r_full["best_score"] > r_full["evals"][0][1]
    # crash at the 4th evaluation (step 15): last checkpoint is step 10
    with pytest.raises(KeyboardInterrupt):
        _toy_distill(20, tmp_path / "b.pt", crash_at_eval=4)
    assert torch.load(tmp_path / "b.pt", weights_only=False)["step"] == 10
    s_res, r_res = _toy_distill(20, tmp_path / "b.pt")
    assert r_res["best_step"] == r_full["best_step"]
    for p, q in zip(s_full.parameters(), s_res.parameters()):
        assert torch.equal(p, q)


def test_distill_requires_labels_for_ce():
    from ttcompress.distill import distill
    with pytest.raises(ValueError):
        distill(TinyNet(), torch.zeros(2, 3, 8, 8, dtype=torch.uint8), torch.zeros(2, 5), {"mean": (0,) * 3, "std": (1,) * 3},
                steps=1, alpha=0.5)


# ------------------------------------------------------------ runner config
def test_runner_configs_expand_to_unique_names():
    import yaml
    sys.path.insert(0, str(ROOT / "experiments"))
    from run import expand
    for f in (ROOT / "configs").glob("*.yaml"):
        c = yaml.safe_load(open(f))
        for suite in c["suites"].values():
            names = [s["name"] for s in expand(suite)]
            assert len(names) == len(set(names)), f
        assert "." in c["model"], "pretrained tag must be pinned (audit H1)"


def test_latency_stats_fields():
    from ttcompress.bench import compare_latency
    r = compare_latency({"a": lambda: sum(range(1000)), "b": lambda: sum(range(2000))}, "a", rounds=8, warmup=1, batch=1)
    for k in ("median_ms", "mean_ms", "std_ms", "p90_ms", "p95_ms", "speedup", "speedup_ci", "throughput_per_s"):
        assert k in r["b"]
    assert r["a"]["speedup"] == 1.0


def test_count_macs_includes_attention():
    """Regression (found in Phase 1): FlopCounterMode skips fused SDPA on CPU."""
    import timm
    from ttcompress.bench import count_macs
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False).eval()
    d, N = 192, 197
    lin = sum(l.weight.numel() for l in m.modules() if isinstance(l, nn.Linear) and l.out_features != 1000) * N
    lin += 1000 * d + m.patch_embed.proj.weight.numel() * 196
    assert count_macs(m, torch.randn(1, 3, 224, 224)) == lin + 12 * 2 * N * N * d


def test_runner_reports_fp32_param_count_for_int8_variants():
    """Regression: quantized Linear weights are not nn.Parameters, so counting
    parameters after dynamic INT8 would report ~0 weights."""
    sys.path.insert(0, str(ROOT / "experiments"))
    from run import build_variant
    m = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64)).eval()
    ctx = {"model": m}
    q, info = build_variant({"kind": "int8"}, ctx)
    assert info["_fp32_params"] == sum(p.numel() for p in m.parameters())
    assert sum(p.numel() for p in q.parameters()) < info["_fp32_params"]


def test_sensitivity_weights_rank_layers_by_output_impact():
    from ttcompress.sensitivity import layer_sensitivities

    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(32, 32)
            self.dead = nn.Linear(32, 32)   # output scaled by ~0 downstream
            self.head = nn.Linear(32, 10)

        def forward(self, x):
            return self.head(self.used(x) + 1e-6 * self.dead(x))

    torch.manual_seed(0)
    m = Two().eval()
    facs = {n: factorize(getattr(m, n).weight.detach()) for n in ("used", "dead")}
    s = layer_sensitivities(m, facs, [torch.randn(64, 32)], granularity=4, min_saving=0.0)
    assert s["used"]["weight"] > 100 * s["dead"]["weight"]
    assert torch.equal(m.used.weight, m.used.weight)  # weights restored (no exception path)
    w_before = m.used.weight.clone()
    layer_sensitivities(m, facs, [torch.randn(8, 32)], granularity=4, min_saving=0.0)
    assert torch.equal(m.used.weight, w_before)
