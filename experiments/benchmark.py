"""
Phase 5: CPU latency, throughput, memory and MACs (protocol §8).

    python experiments/benchmark.py --config configs/deit_small.yaml --what model --backend torch
    python experiments/benchmark.py --config configs/deit_small.yaml --what model --backend ort
    python experiments/benchmark.py --config configs/deit_small.yaml --what rss
    python experiments/benchmark.py --what layers            # dense vs low-rank vs INT8 microbenchmarks

Design decisions
* Backends are timed in separate processes (`--backend`): ONNX Runtime's
  worker threads spin-wait after a call and would steal cycles from a torch
  call interleaved right after it. Within a process, variants are timed in
  shuffled interleaved rounds (bench.compare_latency).
* Peak RSS is measured in a fresh subprocess per variant, so other loaded
  models do not inflate it.
* Refuses to run on battery unless --allow-battery (then results are flagged).
Outputs: results/raw/<model>/bench/*.json, results/raw/layers/bench_layers.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, Logger, env_info, load_model, model_card, power_state, setup_threads, write_json  # noqa: E402
from run import build_variant, find_spec, make_ctx  # noqa: E402

from ttcompress import quant as Q  # noqa: E402
from ttcompress.bench import PeakRSS, compare_latency, count_macs, count_parameters, model_size_mb  # noqa: E402
from ttcompress.lowrank import LowRankLinear, break_even_rank  # noqa: E402

DEFAULT_VARIANTS = ["baseline", "int8", "data_attn+mlp_b0.9", "data_attn+mlp_b0.8", "data_attn+mlp_b0.7",
                    "data_attn+mlp_b0.6", "data_attn+mlp_b0.5", "data_attn+mlp_b0.8_int8", "data_attn+mlp_b0.6_int8"]


def check_power(args, log):
    p = power_state()
    log(f"power: {p}")
    if p["on_ac"] is False and not args.allow_battery:
        sys.exit("On battery power: latency would not be comparable. Plug in, or pass --allow-battery (flagged).")
    return p


def ort_session_nospin(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def build_named(cfgf, ctx, name):
    if name == "baseline":
        return build_variant({"kind": "baseline"}, ctx)
    if name == "int8":
        return build_variant({"kind": "int8"}, ctx)
    return build_variant(dict(find_spec(cfgf, name)), ctx)


def bench_models(args, cfgf, log):
    model, cfg = load_model(cfgf["model"])
    card = model_card(cfgf["model"], model, cfg)
    ctx = make_ctx(cfgf, model, cfg, log)
    threads = torch.get_num_threads()
    names = args.variants or DEFAULT_VARIANTS
    built, meta = {}, {}
    tmp = Path(tempfile.mkdtemp(prefix="ttc_bench_"))
    for n in names:
        if args.backend == "torch":
            m, info = build_named(cfgf, ctx, n)
            m.eval()
            fp32_params = info.get("_fp32_params") or count_parameters(m)
            meta[n] = {"params": fp32_params, "size_mb": round(model_size_mb(m), 3)}
            if "int8" not in n:
                meta[n]["macs_per_image"] = count_macs(m, torch.randn(1, *cfg["input_size"]))
            built[n] = m
        else:  # ort: export the FP32 variant, then FP32 and INT8 graphs
            base = n.replace("_int8", "")
            if base == "int8":
                base = "baseline"
            m, _ = build_named(cfgf, ctx, base)
            f = Q.export_onnx(m, torch.randn(1, *cfg["input_size"]), tmp / f"{base}.onnx")
            for prec in ("fp32", "int8"):
                key = f"ort_{prec}_{base}"
                if key in built:
                    continue
                path = f if prec == "fp32" else Q.ort_dynamic_int8(f, tmp / f"{base}_int8.onnx")
                built[key] = ort_session_nospin(path, threads)
                meta[key] = {"onnx_size_mb": round(Q.file_size_mb(path), 3), "onnx_ops": Q.onnx_op_histogram(path)}
        log(f"built {n}")
    results = {}
    for bs in args.batches:
        x = torch.randn(bs, *cfg["input_size"], generator=torch.Generator().manual_seed(0))
        xn = x.numpy()
        if args.backend == "torch":
            fns = {k: (lambda m=m: m(x)) for k, m in built.items()}
            ref = "baseline"
        else:
            fns = {k: (lambda s=s: s.run(None, {"x": xn})) for k, s in built.items()}
            ref = "ort_fp32_baseline"
        t0 = time.time()
        r = compare_latency(fns, ref, rounds=args.rounds, warmup=args.warmup, batch=bs)
        log(f"batch {bs}: {args.rounds} rounds x {len(fns)} variants in {time.time()-t0:.0f}s")
        for k, v in r.items():
            log(f"  {k:32s} median {v['median_ms']:8.2f} ms  p95 {v['p95_ms']:8.2f}  speed-up {v['speedup']:.3f} "
                f"[{v['speedup_ci'][0]:.3f}, {v['speedup_ci'][1]:.3f}]")
        results[f"batch{bs}"] = r
    return {"backend": args.backend, "variants": meta, "latency": results, "rounds": args.rounds,
            "warmup": args.warmup, "model": card}


def rss_child(args, cfgf):
    """Runs in a fresh process: build ONE variant, run inference, report memory."""
    import psutil
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 2 ** 20
    model, cfg = load_model(cfgf["model"])
    ctx = make_ctx(cfgf, model, cfg, lambda s: None)
    m, _ = build_named(cfgf, ctx, args.rss_variant)
    del model, ctx
    import gc
    gc.collect()
    rss_loaded = proc.memory_info().rss / 2 ** 20
    x = torch.randn(args.rss_batch, *cfg["input_size"])
    with torch.inference_mode():
        m(x)
        with PeakRSS() as pr:
            for _ in range(10):
                m(x)
    print(json.dumps({"variant": args.rss_variant, "batch": args.rss_batch, "rss_start_mb": round(rss0, 1),
                      "rss_after_load_mb": round(rss_loaded, 1), "peak_inference_mb": round(pr.peak_mb, 1),
                      "inference_increment_mb": round(pr.peak_mb - pr.base_mb, 1)}))


def bench_rss(args, cfgf, log):
    out = []
    for n in args.variants or DEFAULT_VARIANTS:
        for bs in args.batches:
            cmd = [sys.executable, __file__, "--config", args.config, "--what", "rss-child", "--rss-variant", n,
                   "--rss-batch", str(bs), "--allow-battery"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            line = [l for l in r.stdout.splitlines() if l.startswith("{")]
            if not line:
                log(f"RSS FAILED {n} b{bs}: {r.stderr[-400:]}")
                continue
            d = json.loads(line[-1])
            out.append(d)
            log(f"  rss {n:28s} b{bs}: after load {d['rss_after_load_mb']:.0f} MB, peak {d['peak_inference_mb']:.0f} MB")
    return {"rss": out, "note": "rss_after_load includes the teacher model that was deleted before measuring "
                                 "(allocator may not return memory to the OS); compare variants relative to each other"}


def bench_layers(args, log):
    """Dense vs low-rank vs INT8 for the real ViT layer shapes, hot and cold cache."""
    from ttcompress.quant import dynamic_int8
    shapes = {"deit_qkv": (1152, 384), "deit_proj": (384, 384), "deit_fc1": (1536, 384), "deit_fc2": (384, 1536),
              "vitb_qkv": (2304, 768), "vitb_proj": (768, 768), "vitb_fc1": (3072, 768), "vitb_fc2": (768, 3072)}
    out = []
    for sname, (m_, n_) in shapes.items():
        be = break_even_rank(n_, m_)
        dense = torch.nn.Linear(n_, m_).eval()
        variants = {"dense_fp32": dense, "dense_int8": dynamic_int8(torch.nn.Sequential(dense))}
        for frac in (0.25, 0.5, 0.75):
            r = max(8, int(frac * be) // 8 * 8)
            lr = LowRankLinear(n_, m_, r).eval()
            variants[f"lr{frac}_fp32"] = lr
            variants[f"lr{frac}_int8"] = dynamic_int8(torch.nn.Sequential(LowRankLinear(n_, m_, r)))
        for tokens in (197, 197 * 8):
            x = torch.randn(tokens, n_)
            for cold in (False, True):
                r = compare_latency({k: (lambda v=v: v(x)) for k, v in variants.items()}, "dense_fp32",
                                    rounds=args.rounds, warmup=args.warmup, cold=cold)
                for k, v in r.items():
                    rank = None if k.startswith("dense") else int(max(8, int(float(k[2:].split("_")[0]) * be) // 8 * 8))
                    macs = tokens * (m_ * n_ if rank is None else rank * (m_ + n_))
                    out.append({"shape": sname, "m": m_, "n": n_, "tokens": tokens, "cold": cold, "variant": k,
                                "rank": rank, "macs": macs, **v})
                log(f"  {sname} T={tokens} {'cold' if cold else 'hot '}: " + ", ".join(
                    f"{k}={v['median_ms']:.3f}ms(x{v['speedup']:.2f})" for k, v in r.items()))
    return {"layers": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--what", choices=["model", "rss", "rss-child", "layers"], required=True)
    ap.add_argument("--backend", choices=["torch", "ort"], default="torch")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 8])
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--rss-variant")
    ap.add_argument("--rss-batch", type=int, default=1)
    args = ap.parse_args()
    setup_threads(8)
    cfgf = yaml.safe_load(open(args.config)) if args.config else {}
    if args.what == "rss-child":
        return rss_child(args, cfgf)
    short = cfgf.get("short", "layers")
    out_dir = RAW / short / "bench" if args.config else RAW / "layers"
    log = Logger(out_dir / "bench.log")
    pw = check_power(args, log)
    t0 = time.time()
    if args.what == "model":
        res = bench_models(args, cfgf, log)
        fname = f"latency_{args.backend}.json"
    elif args.what == "rss":
        res = bench_rss(args, cfgf, log)
        fname = "rss.json"
    else:
        res = bench_layers(args, log)
        fname = "bench_layers.json"
    res.update({"env": env_info(), "power_at_start": pw, "valid": bool(pw.get("on_ac")) or pw.get("on_ac") is None,
                "wall_s": round(time.time() - t0, 1)})
    path = out_dir / fname
    if path.exists():  # never overwrite: keep earlier runs
        path = out_dir / f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_json(path, res)
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
