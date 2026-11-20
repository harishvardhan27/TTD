"""
Config-driven experiment runner.

    python experiments/run.py --config configs/deit_small.yaml --suite core --split v2_test

Each config entry ("variant") is built from the pretrained model, evaluated on
the requested *evaluation* split, and written to

    results/raw/<model>/<split>/<suite>/<variant>.json   (metrics, layers, provenance)
    results/raw/<model>/<split>/<suite>/<variant>.logits.pt  (fp16 logits)

Finished variants are skipped on re-run (resume), so a long CPU run can be
interrupted safely. Nothing is ever overwritten unless --force is given.

Variant spec keys (see configs/*.yaml):
  kind:      baseline | int8 | lowrank | tt | ort
  method:    plain | data                      (lowrank)
  scope:     qkv | attn_out | attn | mlp | attn+mlp | all
  budget:    fraction of targeted weight params kept (lowrank: greedy/dp; tt: per-layer)
  policy:    budget (default) | uniform_budget | uniform_rank | target
  allocator: greedy | dp
  rank / target: for uniform_rank / target policies
  quant:     none | int8 (torch dynamic, applied after compression)
  order:     fq (factorize->quantize, default) | qf (quantize weights->factorize->quantize)
  calib:     {source, n, seed}  overrides the config-level calibration
  damp, min_saving, granularity
  ort:       {precision: fp32|int8, per_channel: bool, base: <variant spec>}
A `grid:` entry expands the Cartesian product of its list-valued keys.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, Logger, env_info, load_model, model_card, setup_threads, write_json  # noqa: E402

from ttcompress import data as D  # noqa: E402
from ttcompress import quant as Q  # noqa: E402
from ttcompress.bench import count_parameters, model_size_mb  # noqa: E402
from ttcompress.calib import collect_input_covariances  # noqa: E402
from ttcompress.evaluate import per_class_shift, predict_logits, score  # noqa: E402
from ttcompress.lowrank import (PREDICATES, compress_lowrank, factorize, find_linears,  # noqa: E402
                                _resolve_parent)


# --------------------------------------------------------------------------
# Config handling
# --------------------------------------------------------------------------

def expand(entries):
    out = []
    for e in entries:
        if "grid" in e:
            g = e["grid"]
            keys = list(g)
            vals = [v if isinstance(v, list) else [v] for v in g.values()]
            for combo in itertools.product(*vals):
                spec = {k: copy.deepcopy(v) for k, v in e.items() if k != "grid"}
                spec.update(dict(zip(keys, combo)))
                out.append(spec)
        else:
            out.append(dict(e))
    for s in out:
        s.setdefault("name", auto_name(s))
    return out


def auto_name(s: dict) -> str:
    k = s.get("kind", "lowrank")
    if k in ("baseline", "int8"):
        return k
    if k == "ort":
        return f"ort_{s['ort'].get('precision', 'fp32')}_" + (auto_name(s["ort"]["base"]) if s["ort"].get("base") else "dense")
    parts = [k if k == "tt" else s.get("method", "data"), s.get("scope", "attn+mlp")]
    pol = s.get("policy", "budget")
    if pol == "budget":
        parts.append(f"b{s['budget']}")
        if s.get("allocator", "greedy") != "greedy":
            parts.append(s["allocator"])
        if s.get("allocator_weights"):
            parts.append("sens")
    elif pol == "uniform_budget":
        parts.append(f"ub{s['budget']}")
    elif pol == "uniform_rank":
        parts.append(f"r{s['rank']}")
    elif pol == "target":
        parts.append(f"t{s['target']}")
    if "calib" in s:
        c = s["calib"]
        parts.append(f"cal-{c.get('source','')}-{c.get('n','')}-s{c.get('seed','')}")
    if "damp" in s:
        parts.append(f"damp{s['damp']:g}")
    if "min_saving" in s:
        parts.append(f"ms{s['min_saving']:g}")
    if s.get("quant", "none") != "none":
        parts.append(s["quant"] + ("_qf" if s.get("order") == "qf" else ""))
    return "_".join(str(p) for p in parts)


# --------------------------------------------------------------------------
# Calibration covariance cache
# --------------------------------------------------------------------------

def get_covs(ctx, calib: dict, heldout: bool):
    """Input covariances for every nn.Linear, cached on disk per
    (model, source, n, seed, part). The held-out part uses a disjoint
    subset of the same source (for unbiased output-error reporting)."""
    model, cfg, mname = ctx["model"], ctx["cfg"], ctx["model_name"]
    src, n, seed = calib["source"], int(calib["n"]), int(calib.get("seed", 1))
    n_ho = int(calib.get("heldout_n", 128))
    key = hashlib.sha1(json.dumps([mname, src, n, seed, n_ho, heldout]).encode()).hexdigest()[:12]
    path = D.DATA_ROOT / "cache" / f"covs_{mname.split('.')[0]}_{key}.pt"
    if path.exists():
        return torch.load(path)
    items = D.get_items(src, role="train")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(items), generator=g).tolist()
    if n + n_ho > len(items):
        raise ValueError(f"calibration source {src} has {len(items)} images < {n}+{n_ho}")
    idx = perm[n:n + n_ho] if heldout else perm[:n]
    sel = [items[i] for i in sorted(idx)]
    # preprocess via the shared cache machinery
    x = _load_items(sel, cfg, f"calib_{src}")
    names = find_linears(model, PREDICATES["all"])
    t0 = time.time()
    covs = collect_input_covariances(model, names, D.batches(x, cfg, ctx["batch"]))
    covs = {k: v.float() for k, v in covs.items()}
    torch.save(covs, path)
    ctx["log"](f"  calibration covs ({src}, n={len(sel)}, seed={seed}, heldout={heldout}): {time.time()-t0:.0f}s")
    return covs


def _load_items(items, cfg, tag):
    from PIL import Image
    files = "\n".join(str(p) for _, p in items)
    key = hashlib.sha1((files + json.dumps({k: cfg.get(k) for k in ("input_size", "crop_pct", "interpolation")})).encode()).hexdigest()[:12]
    path = D.DATA_ROOT / "cache" / f"{tag}_{key}.pt"
    if path.exists():
        return torch.load(path)["x"]
    tf = D._uint8_transform(cfg)
    x = torch.stack([tf(Image.open(p).convert("RGB")) for _, p in items])
    torch.save({"x": x}, path)
    return x


# --------------------------------------------------------------------------
# Variant construction
# --------------------------------------------------------------------------

def build_lowrank(m, spec, ctx):
    scope = spec.get("scope", "attn+mlp")
    pred = PREDICATES[scope]
    calib = {**ctx["calib"], **spec.get("calib", {})}
    damp = float(spec.get("damp", ctx["damp"]))
    gran = int(spec.get("granularity", ctx["granularity"]))
    min_saving = float(spec.get("min_saving", ctx["min_saving"]))
    covs = get_covs(ctx, calib, heldout=False) if spec.get("method", "data") == "data" else None
    eval_covs = get_covs(ctx, calib, heldout=True)

    if spec.get("order") == "qf":
        # quantize-then-factorize: factorize the INT8-rounded weights
        for n in find_linears(m, pred):
            lin = getattr(*_resolve_parent(m, n))
            with torch.no_grad():
                lin.weight.copy_(Q.fake_quant_weight(lin.weight, per_channel=True))

    pol = spec.get("policy", "budget")
    kw = dict(covs=covs, eval_covs=eval_covs, damp=damp, granularity=gran, min_saving=min_saving)
    sens_info = None
    if spec.get("allocator_weights") == "sensitivity":
        kw["weights"], sens_info = sensitivity_weights(m, pred, covs, damp, gran, min_saving, calib, ctx,
                                                       int(spec.get("sens_images", 64)))
    if pol == "budget":
        rep = compress_lowrank(m, pred, budget_fraction=float(spec["budget"]), allocator=spec.get("allocator", "greedy"), **kw)
    elif pol == "uniform_budget":
        names = find_linears(m, pred)
        shapes = [tuple(getattr(*_resolve_parent(m, n)).weight.shape) for n in names]
        r = int(spec["budget"] * sum(o * i for o, i in shapes) / sum(o + i for o, i in shapes))
        r = max(gran, (r // gran) * gran)
        rep = compress_lowrank(m, pred, rank=r, **kw)
    elif pol == "uniform_rank":
        rep = compress_lowrank(m, pred, rank=int(spec["rank"]), **kw)
    elif pol == "target":
        rep = compress_lowrank(m, pred, target_rel_error=float(spec["target"]), **kw)
    else:
        raise ValueError(pol)
    res = {"targeted_dense_params": rep.dense_params, "targeted_compressed_params": rep.compressed_params,
           "targeted_ratio": round(rep.layer_ratio, 4), "summary": rep.summary(), "layers": rep.layers}
    if sens_info is not None:
        res["sensitivity"] = sens_info
    return res


def sensitivity_weights(m, pred, covs, damp, gran, min_saving, calib, ctx, n_img):
    """Per-layer KL sensitivity on training-side hold-out images (the images
    after the calibration fit set and its covariance hold-out in the same
    seeded permutation, so they are disjoint from both)."""
    from ttcompress.sensitivity import layer_sensitivities
    items = D.get_items(calib["source"], role="train")
    g = torch.Generator().manual_seed(int(calib.get("seed", 1)))
    perm = torch.randperm(len(items), generator=g).tolist()
    start = int(calib["n"]) + int(calib.get("heldout_n", 128))
    sel = [items[i] for i in sorted(perm[start:start + n_img])]
    x = _load_items(sel, ctx["cfg"], f"sens_{calib['source']}")
    names = find_linears(m, pred)
    facs = {n: factorize(getattr(*_resolve_parent(m, n)).weight.detach(), covs[n] if covs is not None else None, damp=damp)
            for n in names}
    t0 = time.time()
    sens = layer_sensitivities(m, facs, D.batches(x, ctx["cfg"], ctx["batch"]), granularity=gran, min_saving=min_saving)
    ctx["log"](f"  sensitivity weights for {len(names)} layers on {len(sel)} images: {time.time()-t0:.0f}s")
    return {n: v["weight"] for n, v in sens.items()}, {n: {k: round(v2, 6) if isinstance(v2, float) else v2 for k, v2 in v.items()} for n, v in sens.items()}


def build_tt(m, spec, ctx):
    """TT baseline at a matched per-layer parameter budget: for each targeted
    layer, the largest uniform TT rank whose cores fit within budget * m*n."""
    from ttcompress.lowrank import output_rel_error, weight_rel_error
    from ttcompress.tt import TTLinear
    pred = PREDICATES[spec.get("scope", "attn+mlp")]
    eval_covs = get_covs(ctx, ctx["calib"], heldout=True)
    num_modes = int(spec.get("num_modes", 3))
    layers, dense, comp = [], 0, 0
    for n in find_linears(m, pred):
        lin = getattr(*_resolve_parent(m, n))
        o, i = lin.weight.shape
        cap = spec["budget"] * o * i
        best = None
        lo, hi = 1, 512
        while lo <= hi:  # TT params are monotone in the uniform rank
            r = (lo + hi) // 2
            t = TTLinear.from_linear(lin, num_modes=num_modes, ranks=r)
            p = sum(c.numel() for c in t.cores)
            if p <= cap:
                best, lo = (r, t, p), r + 1
            else:
                hi = r - 1
        if best is None:
            layers.append({"name": n, "rank": None, "note": "no TT rank fits budget; kept dense"})
            dense += o * i
            comp += o * i
            continue
        r, t, p = best
        W, W_hat = lin.weight.detach(), t.to_dense_weight()
        layers.append({"name": n, "shape": [o, i], "tt_rank": r, "in_modes": list(t.in_modes), "out_modes": list(t.out_modes),
                       "realized_ranks": list(t.full_ranks), "params": p,
                       "weight_rel_error": round(weight_rel_error(W, W_hat), 5),
                       "output_rel_error": round(output_rel_error(W, W_hat, eval_covs[n]), 5)})
        parent, attr = _resolve_parent(m, n)
        setattr(parent, attr, t)
        dense += o * i
        comp += p
    return {"targeted_dense_params": dense, "targeted_compressed_params": comp,
            "targeted_ratio": round(dense / comp, 4), "layers": layers}


def build_variant(spec, ctx):
    kind = spec.get("kind", "lowrank")
    m = copy.deepcopy(ctx["model"])
    info = {}
    if kind == "baseline":
        pass
    elif kind == "int8":
        info["_fp32_params"] = count_parameters(m)  # INT8 packed weights are not nn.Parameters
        m = Q.dynamic_int8(m)
        info["int8_coverage"] = Q.torch_int8_coverage(m)
    elif kind == "lowrank":
        info = build_lowrank(m, spec, ctx)
        if spec.get("quant", "none") == "int8":
            info["_fp32_params"] = count_parameters(m)
            m = Q.dynamic_int8(m)
            info["int8_coverage"] = Q.torch_int8_coverage(m)
    elif kind == "tt":
        info = build_tt(m, spec, ctx)
    elif kind == "ort":
        o = spec["ort"]
        if o.get("base"):
            m, info = build_variant(dict(o["base"]), ctx)
        tmp = Path(tempfile.mkdtemp(prefix="ttc_onnx_"))
        x = torch.randn(1, *ctx["cfg"]["input_size"])
        f = Q.export_onnx(m, x, tmp / "model.onnx")
        if o.get("precision", "fp32") == "int8":
            f = Q.ort_dynamic_int8(f, tmp / "model_int8.onnx", per_channel=o.get("per_channel", True))
        info.update({"onnx_ops": Q.onnx_op_histogram(f), "onnx_size_mb": round(Q.file_size_mb(f), 3)})
        sess = Q.ort_session(f, threads=torch.get_num_threads())
        info["_ort_session"] = sess
        info["_fp32_params"] = count_parameters(m)
        return m, info
    else:
        raise ValueError(kind)
    return m, info


def make_ctx(cfgf: dict, model, cfg: dict, log) -> dict:
    """Shared build context (also used by distill_run.py)."""
    return {"model": model, "cfg": cfg, "model_name": cfgf["model"], "batch": int(cfgf.get("batch", 32)), "log": log,
            "calib": cfgf["calibration"], "damp": float(cfgf.get("damp", 1e-2)),
            "granularity": int(cfgf.get("granularity", 8)), "min_saving": float(cfgf.get("min_saving", 0.1))}


def find_spec(cfgf: dict, name: str) -> dict:
    for suite in cfgf["suites"].values():
        for s in expand(suite):
            if s["name"] == name:
                return s
    raise KeyError(f"variant {name} not found in any suite")


@torch.no_grad()
def ort_logits(sess, x_uint8, cfg, batch):
    out = []
    for xb in D.batches(x_uint8, cfg, batch):
        out.append(torch.from_numpy(sess.run(None, {"x": xb.numpy()})[0]).float())
    return torch.cat(out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--split", required=True, choices=D.EVAL_SPLITS)
    ap.add_argument("--only", nargs="*", help="run only these variant names")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfgf = yaml.safe_load(open(args.config))
    threads = setup_threads(cfgf.get("threads"))
    torch.manual_seed(cfgf.get("seed", 0))
    mname = cfgf["model"]
    short = cfgf.get("short", mname.split(".")[0])
    outdir = RAW / short / args.split / args.suite
    log = Logger(outdir / "run.log")
    log(f"=== suite {args.suite} on {args.split}, config {args.config}, threads={threads}")

    model, cfg = load_model(mname)
    card = model_card(mname, model, cfg)
    x, y = D.load_cached(args.split, cfg, role="eval")
    log(f"model {mname} sha256={str(card['sha256'])[:16]}… params={card['params']:,}; eval {args.split}: {len(x)} images")

    ctx = make_ctx(cfgf, model, cfg, log)

    # Baseline logits (shared reference for every paired comparison)
    base_path = RAW / short / args.split / "baseline.logits.pt"
    if base_path.exists():
        base_logits = torch.load(base_path).float()
    else:
        t0 = time.time()
        base_logits = predict_logits(model, x, cfg, ctx["batch"])
        base_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(base_logits.half(), base_path)
        torch.save(y, RAW / short / args.split / "labels.pt")
        log(f"baseline logits: {time.time()-t0:.0f}s")
    base_logits = base_logits.float()
    base_size = model_size_mb(model)

    specs = expand(cfgf["suites"][args.suite])
    if args.only:
        specs = [s for s in specs if s["name"] in args.only]
    log(f"{len(specs)} variants: {[s['name'] for s in specs]}")
    provenance = {"env": env_info(), "model": card, "config_file": args.config, "config_sha1":
                  hashlib.sha1(Path(args.config).read_bytes()).hexdigest(), "split": args.split,
                  "split_manifest_sha256": D.manifest_digest(args.split)}

    for spec in specs:
        out = outdir / f"{spec['name']}.json"
        if out.exists() and not args.force:
            log(f"skip {spec['name']} (exists)")
            continue
        t0 = time.time()
        try:
            m, info = build_variant(spec, ctx)
            t_build = time.time() - t0
            t1 = time.time()
            if spec.get("kind") == "ort":
                logits = ort_logits(info.pop("_ort_session"), x, cfg, ctx["batch"])
                params = info.pop("_fp32_params")
                size = info["onnx_size_mb"]
            elif spec.get("kind") == "baseline":
                logits = base_logits  # identical model; reuse the reference logits
                params, size = count_parameters(m), model_size_mb(m)
            else:
                logits = predict_logits(m, x, cfg, ctx["batch"])
                params = info.pop("_fp32_params", None) or count_parameters(m)
                size = model_size_mb(m)
            t_eval = time.time() - t1
        except Exception as e:  # record failures, never skip silently
            import traceback
            write_json(outdir / f"{spec['name']}.FAILED.json", {"spec": spec, "error": repr(e),
                                                                  "traceback": traceback.format_exc(), **provenance})
            log(f"FAILED {spec['name']}: {e!r}")
            continue
        sc = score(logits, y, base_logits if spec.get("kind") != "baseline" else None)
        rec = {"spec": spec, "params": params, "param_ratio": round(card["params"] / max(params, 1), 4),
               "size_mb": round(size, 3), "size_ratio": round(base_size / max(size, 1e-9), 4),
               "build_s": round(t_build, 1), "eval_s": round(t_eval, 1), "score": sc.to_dict(),
               "per_class_shift": per_class_shift(logits, base_logits, y), **info, **provenance}
        torch.save(logits.half(), outdir / f"{spec['name']}.logits.pt")
        write_json(out, rec)
        log(f"{spec['name']:45s} params={params/1e6:6.2f}M ({rec['param_ratio']:.2f}x) size={size:7.1f}MB | {sc.short()} "
            f"[build {t_build:.0f}s, eval {t_eval:.0f}s]")
        if "summary" in info:
            log(f"    {info['summary']}")
    log("=== done")


if __name__ == "__main__":
    main()
