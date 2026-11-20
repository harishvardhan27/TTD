import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from ttcompress.calib import collect_input_covariances
from ttcompress.evaluate import mcnemar_exact_p, score, wilson_interval
from ttcompress.lowrank import (LowRankLinear, allocate_ranks, break_even_rank, compress_lowrank, factorize,
                                rank_for_target_error)
from ttcompress.tt import TTLinear, factorize_balanced, tt_reconstruct, tt_svd
from ttcompress.ttconv import TTConv2d

torch.manual_seed(0)


def low_rank_matrix(m, n, r, noise=1e-3):
    return torch.randn(m, r) @ torch.randn(r, n) / math.sqrt(r) + noise * torch.randn(m, n)


# ---------------------------------------------------------------- TT
def test_factorize_balanced():
    for n in (768, 3072, 1000, 97):
        for k in (1, 2, 3):
            assert math.prod(factorize_balanced(n, k)) == n


def test_tt_svd_full_rank_is_exact():
    T = np.random.randn(4, 5, 6)
    cores = tt_svd(T, [100, 100])
    assert np.allclose(tt_reconstruct(cores), T, atol=1e-10)


def test_ttlinear_forward_matches_dense():
    lin = nn.Linear(48, 60)
    tt = TTLinear.from_linear(lin, num_modes=3, ranks=64)  # generous rank -> exact
    x = torch.randn(5, 48)
    assert torch.allclose(tt(x), lin(x), atol=1e-4)
    assert torch.allclose(tt(x), x @ tt.to_dense_weight().T + tt.bias, atol=1e-4)


def test_ttconv_matches_conv():
    conv = nn.Conv2d(8, 12, 3, padding=1, stride=2)
    ttc = TTConv2d.from_conv(conv, num_modes=2, ranks=200)
    x = torch.randn(2, 8, 9, 9)
    assert torch.allclose(ttc(x), conv(x), atol=1e-4)


# ---------------------------------------------------------------- low-rank
def test_plain_lowrank_is_eckart_young_optimal():
    W = torch.randn(40, 30)
    fac = factorize(W)
    U, V = fac.factors(10)
    err = (W - U @ V).norm() / W.norm()
    S = torch.linalg.svdvals(W)
    assert abs(float(err) - float((S[10:] ** 2).sum().sqrt() / W.norm())) < 1e-5
    assert abs(fac.rel_error(10) - float(err)) < 1e-5


def test_data_aware_error_identity_and_beats_plain_on_output():
    m, n, N = 50, 40, 4000
    W = torch.randn(m, n)
    # anisotropic inputs: a few directions dominate
    A = torch.randn(n, n) * torch.logspace(0, -2, n)
    X = torch.randn(N, n) @ A.T
    cov = X.T @ X / N
    fac_d = factorize(W, cov, damp=0.0)
    fac_p = factorize(W)
    r = 8
    Ud, Vd = fac_d.factors(r)
    Up, Vp = fac_p.factors(r)
    out_err = lambda U, V: float(((W - U @ V) @ X.T).norm() / (W @ X.T).norm())
    # reported error equals the true output error on the calibration data
    assert abs(fac_d.rel_error(r) - out_err(Ud, Vd)) < 1e-3
    assert out_err(Ud, Vd) < out_err(Up, Vp)


def test_rank_for_target_error_reports_unreachable_targets():
    W = torch.randn(64, 64)  # flat spectrum: 5% error unreachable below break-even (32)
    fac = factorize(W)
    ch = rank_for_target_error(fac, 64, 64, target_rel_error=0.05)
    assert ch.rank is None and not ch.met_target and "kept dense" in ch.reason
    W2 = low_rank_matrix(64, 64, 4)
    ch2 = rank_for_target_error(factorize(W2), 64, 64, target_rel_error=0.05)
    assert ch2.rank == 8 and ch2.met_target


def test_allocate_ranks_prefers_compressible_layers_and_respects_budget():
    facs = {"easy": factorize(low_rank_matrix(64, 64, 8)), "hard": factorize(torch.randn(64, 64))}
    shapes = {"easy": (64, 64), "hard": (64, 64)}
    budget = 64 * 64  # half the dense total
    alloc = allocate_ranks(facs, shapes, budget, granularity=8)
    assert alloc["easy"] == 8
    used = sum(r * 128 if r else 64 * 64 for r in alloc.values())
    assert used <= budget


def test_break_even():
    assert break_even_rank(768, 2304) == 575
    assert 575 * (768 + 2304) < 768 * 2304 <= 576 * (768 + 2304)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.ModuleDict({"qkv": nn.Linear(32, 96), "proj": nn.Linear(32, 32)})
        self.mlp = nn.ModuleDict({"fc1": nn.Linear(32, 64)})

    def forward(self, x):
        return self.mlp["fc1"](self.attn["proj"](self.attn["qkv"](x)[..., :32]))


def test_compress_lowrank_end_to_end_and_int8_composes():
    model = Toy().eval()
    x = torch.randn(4, 10, 32)
    names = ["attn.qkv", "attn.proj"]
    covs = collect_input_covariances(model, names, [x])
    assert set(covs) == set(names) and covs["attn.qkv"].shape == (32, 32)
    rep = compress_lowrank(model, covs=covs, budget_fraction=0.5, granularity=4)
    assert isinstance(model.attn["qkv"], LowRankLinear)
    assert isinstance(model.mlp["fc1"], nn.Linear)  # not targeted
    assert rep.compressed_params <= rep.dense_params
    from ttcompress.quant import dynamic_int8
    q = dynamic_int8(model)
    assert torch.isfinite(q(x)).all()


# ---------------------------------------------------------------- stats
def test_stats_helpers():
    lo, hi = wilson_interval(93, 100)
    assert 0.85 < lo < 0.93 < hi < 0.97
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(0, 10) < 0.01
    logits = torch.randn(100, 10)
    y = logits.argmax(1)
    r = score(logits, y, base_logits=logits)
    assert r.top1 == 1.0 and r.agreement == 1.0 and abs(r.kl_to_base) < 1e-6 and r.mcnemar_p == 1.0
