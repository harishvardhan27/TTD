"""
Experiment 1: low-rank compression of a real ViT, done properly.

Questions:
  Q1  Was the old HO result ("target_err<=0.15 / <=0.30") what it claimed?
  Q2  At a matched parameter budget, does data-aware (activation-whitened)
      SVD beat plain weight SVD on real accuracy?
  Q3  With data-aware SVD, do the MLP layers (near full-rank as *weights*)
      become compressible?

Eval: Imagenette val (3925 imgs, scored 1000-way) + ImageNet sample (1000
imgs, 1000 classes). Calibration: 256 Imagenette *train* images (disjoint).

Usage: python experiments/vit_lowrank.py [--model vit_base_patch16_224_miil] [--quick]
"""

from __future__ import annotations

import argparse
import copy
import time

import torch

from common import EVAL_SPLITS, Logger, RESULTS, env_info, load_model, save_json, setup_threads
from ttcompress import data as D
from ttcompress.bench import count_parameters, model_size_mb
from ttcompress.calib import collect_input_covariances
from ttcompress.evaluate import predict_logits, score
from ttcompress.lowrank import (LowRankLinear, attention_predicate, attn_and_mlp_predicate, compress_lowrank,
                                factorize, find_linears)


def legacy_ho_ranks(model, target):
    """Exact re-implementation of HO/low_rank_compress.suggest_rank_by_target_error."""
    cands = (4, 8, 16, 32, 64, 96, 128, 192, 256)
    out = {}
    for n in find_linears(model, attention_predicate):
        W = dict(model.named_modules())[n].weight.detach()
        fac = factorize(W)
        r_pick, err_pick = None, None
        for r in cands:
            if fac.rel_error(r) <= target:
                r_pick, err_pick = r, fac.rel_error(r)
                break
        if r_pick is None:
            r_pick, err_pick = cands[-1], fac.rel_error(cands[-1])
        out[n] = (r_pick, err_pick)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch16_224_miil.in21k_ft_in1k")
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--quick", action="store_true", help="small eval subset for a smoke run")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    threads = setup_threads()
    tag = args.model.split("_patch")[0] + ("_quick" if args.quick else "")
    log = Logger(RESULTS / f"vit_lowrank_{tag}.log")
    log(f"env: {env_info()}")

    model, cfg = load_model(args.model)
    log(f"model {args.model}: {count_parameters(model):,} params, data cfg mean={cfg['mean']} std={cfg['std']} "
        f"crop_pct={cfg.get('crop_pct')} interp={cfg.get('interpolation')}")

    lim = 200 if args.quick else None
    evalsets = {s: D.load_cached(s, cfg, limit=lim) for s in EVAL_SPLITS}
    xcal, _ = D.load_cached("imagenette_train", cfg, limit=args.calib, seed=1)
    log("eval sets: " + ", ".join(f"{s}={len(x)}" for s, (x, _) in evalsets.items()) + f"; calib={len(xcal)}")

    t0 = time.time()
    base_logits = {s: predict_logits(model, x, cfg, args.batch) for s, (x, y) in evalsets.items()}
    n_eval = sum(len(x) for x, _ in evalsets.values())
    log(f"baseline forward over {n_eval} images: {time.time()-t0:.0f}s ({(time.time()-t0)/n_eval*1e3:.0f} ms/img)")
    all_y = torch.cat([y for _, y in evalsets.values()])
    all_base = torch.cat(list(base_logits.values()))

    results = {"env": env_info(), "model": args.model, "params": count_parameters(model),
               "size_mb": round(model_size_mb(model), 2), "calib_images": len(xcal), "configs": []}
    base_scores = {s: score(base_logits[s], evalsets[s][1]).to_dict() for s in evalsets}
    base_scores["all"] = score(all_base, all_y).to_dict()
    results["baseline"] = base_scores
    log("baseline: " + " | ".join(f"{s}: {score(base_logits[s], evalsets[s][1]).short()}" for s in evalsets))
    torch.save({"base_logits": base_logits}, RESULTS / f"base_logits_{tag}.pt")

    # --- calibration covariances for every attention + MLP linear (one pass)
    t0 = time.time()
    all_names = find_linears(model, attn_and_mlp_predicate)
    covs = collect_input_covariances(model, all_names, D.batches(xcal, cfg, args.batch))
    log(f"calibration: {len(covs)} layers, {time.time()-t0:.0f}s")

    def run(name, build, extra=None):
        m = copy.deepcopy(model)
        t0 = time.time()
        rep = build(m)
        t_build = time.time() - t0
        logits = {s: predict_logits(m, x, cfg, args.batch) for s, (x, _) in evalsets.items()}
        per = {s: score(logits[s], evalsets[s][1], base_logits[s]) for s in evalsets}
        allr = score(torch.cat(list(logits.values())), all_y, all_base)
        row = {"name": name, "params": count_parameters(m), "size_mb": round(model_size_mb(m), 2),
               "model_ratio": round(results["params"] / count_parameters(m), 3), "build_s": round(t_build, 1),
               "scores": {**{s: r.to_dict() for s, r in per.items()}, "all": allr.to_dict()}}
        if rep is not None:
            row["targeted_ratio"] = round(rep.layer_ratio, 3)
            row["layers"] = rep.layers
            row["summary"] = rep.summary()
        if extra:
            row.update(extra)
        results["configs"].append(row)
        save_json(results, f"vit_lowrank_{tag}.json")
        log(f"{name:38s} params={row['params']/1e6:6.2f}M ({row['model_ratio']:.2f}x) | {allr.short()}")
        if rep is not None:
            log(f"    {rep.summary()}")
        return m

    # Q1: replicate the old HO configurations exactly
    for tgt in (0.15, 0.30):
        ranks = legacy_ho_ranks(model, tgt)
        hit = sum(1 for r, e in ranks.values() if e <= tgt)
        errs = [e for _, e in ranks.values()]
        log(f"legacy HO target {tgt}: {hit}/{len(ranks)} layers actually met the target; "
            f"ranks used={sorted(set(r for r, _ in ranks.values()))}; achieved err mean={sum(errs)/len(errs):.3f} "
            f"max={max(errs):.3f}")

        def build(m, ranks=ranks):
            mods = dict(m.named_modules())
            for n, (r, _) in ranks.items():
                parent = mods[n.rsplit(".", 1)[0]]
                lin = mods[n]
                setattr(parent, n.rsplit(".", 1)[1], LowRankLinear.from_linear(lin, r))
            return None
        run(f"legacy_HO_target{tgt}", build, {"legacy_layers_meeting_target": hit,
                                              "legacy_ranks": {n: r for n, (r, _) in ranks.items()},
                                              "legacy_errors": {n: round(e, 4) for n, (_, e) in ranks.items()}})
    run("uniform_rank256_attn_plain", lambda m: compress_lowrank(m, attention_predicate, rank=256))

    # Q2 / Q3: matched budgets, plain vs data-aware, attention-only vs attention+MLP
    grid = [("attn", attention_predicate, (0.75, 0.5, 0.35, 0.25)),
            ("attn+mlp", attn_and_mlp_predicate, (0.85, 0.7, 0.5))]
    for scope, pred, budgets in grid:
        for b in budgets:
            for method in ("plain", "data"):
                c = covs if method == "data" else None
                run(f"{scope}_{method}_budget{b}",
                    lambda m, pred=pred, b=b, c=c: compress_lowrank(m, pred, covs=c, budget_fraction=b),
                    {"scope": scope, "method": method, "budget_fraction": b})

    log("done")


if __name__ == "__main__":
    main()
