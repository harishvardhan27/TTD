"""
Phase 4: knowledge-distillation recovery of a compressed student.

Selection runs (dev only):
    python experiments/distill_run.py --config configs/deit_small.yaml --student data_attn+mlp_b0.6 \
        --tag T2_a0_lr1e-4 --T 2 --alpha 0 --lr 1e-4 --steps 200 --eval-every 50

Final run (selected hyperparameters; evaluates the dev-selected checkpoint ONCE on v2_test,
plus its dynamic-INT8 version = B8):
    ... --tag final --steps 1500 --final

Artifacts: results/raw/<model>/distill/<student>/<tag>/{result.json, train.log}
           checkpoints/<model>/<student>/<tag>.pt       (resumable; gitignored)
Final test records are also written in run.py's format to
           results/raw/<model>/v2_test/distill/<student>__<tag>[_int8].json (+ .logits.pt)
so experiments/analyze.py treats them like any other variant.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, ROOT, Logger, env_info, load_model, model_card, setup_threads, write_json  # noqa: E402
from run import _load_items, build_variant, find_spec, make_ctx  # noqa: E402

from ttcompress import data as D  # noqa: E402
from ttcompress import quant as Q  # noqa: E402
from ttcompress.bench import count_parameters, model_size_mb  # noqa: E402
from ttcompress.distill import distill, teacher_logits  # noqa: E402
from ttcompress.evaluate import per_class_shift, predict_logits, score  # noqa: E402


def transfer_set(cfg, limit, seed):
    items = D.subset(D.get_items("transfer", role="train"), limit, seed)
    x = _load_items(items, cfg, "transfer")
    y = torch.tensor([c for c, _ in items])
    return x, y, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--student", required=True, help="variant name from the config's suites")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--transfer-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="evaluate the selected checkpoint once on v2_test")
    args = ap.parse_args()

    cfgf = yaml.safe_load(open(args.config))
    setup_threads(cfgf.get("threads"))
    torch.manual_seed(args.seed)
    short = cfgf.get("short", cfgf["model"].split(".")[0])
    out = RAW / short / "distill" / args.student / args.tag
    log = Logger(out / "train.log")
    log(f"=== distill {args.student} tag={args.tag} T={args.T} alpha={args.alpha} lr={args.lr} steps={args.steps} "
        f"batch={args.batch}x{args.accum} seed={args.seed}")

    teacher, cfg = load_model(cfgf["model"])
    card = model_card(cfgf["model"], teacher, cfg)
    ctx = make_ctx(cfgf, teacher, cfg, log)
    spec = find_spec(cfgf, args.student)
    student, info = build_variant(dict(spec), ctx)

    xt, yt, items = transfer_set(cfg, args.transfer_limit, seed=1)
    key = hashlib.sha1(("\n".join(str(p) for _, p in items) + cfgf["model"]).encode()).hexdigest()[:12]
    tpath = D.DATA_ROOT / "cache" / f"teacher_{short}_{key}.pt"
    if tpath.exists():
        tl, tlf = torch.load(tpath)
    else:
        t0 = time.time()
        tl = teacher_logits(teacher, xt, cfg, ctx["batch"])
        tlf = teacher_logits(teacher, xt, cfg, ctx["batch"], hflip=True)
        torch.save((tl, tlf), tpath)
        log(f"teacher logits for {len(xt)} transfer images (+flip): {time.time()-t0:.0f}s")

    xdev, ydev = D.load_cached("v2_dev", cfg, role="eval")
    base_dev_path = RAW / short / "v2_dev" / "baseline.logits.pt"
    base_dev = torch.load(base_dev_path).float() if base_dev_path.exists() else predict_logits(teacher, xdev, cfg, ctx["batch"])
    pre_dev = score(predict_logits(student, xdev, cfg, ctx["batch"]), ydev, base_dev)
    log(f"student before KD (dev): {pre_dev.short()}")

    def eval_fn(m):
        return score(predict_logits(m, xdev, cfg, ctx["batch"]), ydev).top1

    ck = ROOT / "checkpoints" / short / args.student / f"{args.tag}.pt"
    hist = distill(student, xt, tl, cfg, steps=args.steps, labels=yt if args.alpha > 0 else None, alpha=args.alpha,
                   temperature=args.T, batch_size=args.batch, accum=args.accum, lr=args.lr, t_logits_flip=tlf,
                   eval_fn=eval_fn, eval_every=args.eval_every, patience=args.patience, ckpt_path=ck, seed=args.seed,
                   log=log)
    post_dev = score(predict_logits(student, xdev, cfg, ctx["batch"]), ydev, base_dev)
    log(f"student after KD, selected step {hist['best_step']} (dev): {post_dev.short()}")

    rec = {"student": args.student, "spec": spec, "tag": args.tag,
           "hparams": {k: getattr(args, k) for k in ("T", "alpha", "lr", "steps", "batch", "accum", "eval_every",
                                                       "patience", "transfer_limit", "seed")},
           "transfer_images": len(xt), "dev_before": pre_dev.to_dict(), "dev_after": post_dev.to_dict(), **hist,
           "params": count_parameters(student), "checkpoint": str(ck.relative_to(ROOT)), "env": env_info(), "model": card}
    write_json(out / "result.json", rec)

    if args.final:
        xte, yte = D.load_cached("v2_test", cfg, role="eval")
        base_te = torch.load(RAW / short / "v2_test" / "baseline.logits.pt").float()
        tdir = RAW / short / "v2_test" / "distill"
        prov = {"env": env_info(), "model": card, "config_file": args.config, "split": "v2_test",
                "split_manifest_sha256": D.manifest_digest("v2_test"), "distill": rec["hparams"],
                "distill_best_step": hist["best_step"], "distill_train_seconds": hist["train_seconds"]}
        tdir.mkdir(parents=True, exist_ok=True)
        fp32_params = count_parameters(student)  # INT8 weights are not nn.Parameters; report the FP32 count
        for suffix, m in (("", student), ("_int8", Q.dynamic_int8(student))):
            name = f"{args.student}__kd-{args.tag}{suffix}"
            lg = predict_logits(m, xte, cfg, ctx["batch"])
            sc = score(lg, yte, base_te)
            torch.save(lg.half(), tdir / f"{name}.logits.pt")
            spec_out = {**spec, "name": name, "kd": args.tag, "quant": "int8" if suffix else spec.get("quant", "none")}
            write_json(tdir / f"{name}.json", {
                "spec": spec_out, "params": fp32_params, "param_ratio": round(card["params"] / fp32_params, 4),
                "size_mb": round(model_size_mb(m), 3), "size_ratio": round(model_size_mb(teacher) / model_size_mb(m), 4),
                "score": sc.to_dict(), "per_class_shift": per_class_shift(lg, base_te, yte),
                **{k: v for k, v in info.items() if k in ("layers", "summary", "targeted_ratio")}, **prov})
            log(f"TEST {name}: {sc.short()}")
    log("=== done")


if __name__ == "__main__":
    main()
