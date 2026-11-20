"""
Phase 4 orchestrator: the pre-registered distillation selection
(docs/EXPERIMENTAL_PROTOCOL.md §7, fixed 2026-11-20 16:35), executed
mechanically so no choice is made by hand after seeing results.

    python experiments/distill_select.py --config configs/deit_small.yaml [--dry-run]

Stages (b = 0.6 student, 200 steps, dev eval every 50 steps):
  a) T in {1, 2, 4} at alpha = 0, lr = 1e-4
  b) lr = 3e-4 at the best T
  c) alpha = 0.5 at the best (T, lr)
Each stage keeps the configuration with the highest best-checkpoint dev top-1;
ties within 0.25 pp go to the earlier (simpler) configuration.
Final: selected configuration, 1,500 steps, eval every 100, patience 5, for
students b = 0.6 and b = 0.8, each evaluated once on v2_test (FP32 and INT8).

Every run is a separate subprocess (memory fully released between runs) and
is skipped if its result.json already exists (resumable).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, Logger, write_json  # noqa: E402

TIE = 0.0025  # 0.25 pp in accuracy units
PRIMARY, SECONDARY = "data_attn+mlp_b0.6", "data_attn+mlp_b0.8"


def tag_of(T, alpha, lr, steps):
    return f"T{T:g}_a{alpha:g}_lr{lr:g}_s{steps}"


def run(args, short, student, T, alpha, lr, steps, eval_every, patience=None, final=False, log=print):
    tag = ("final_" if final else "sel_") + tag_of(T, alpha, lr, steps)
    res = RAW / short / "distill" / student / tag / "result.json"
    if res.exists():
        log(f"  skip {student}/{tag} (exists)")
    else:
        cmd = [sys.executable, str(Path(__file__).parent / "distill_run.py"), "--config", args.config, "--student",
               student, "--tag", tag, "--T", str(T), "--alpha", str(alpha), "--lr", str(lr), "--steps", str(steps),
               "--eval-every", str(eval_every)]
        if patience:
            cmd += ["--patience", str(patience)]
        if final:
            cmd += ["--final"]
        log("  $ " + " ".join(cmd[1:]))
        if args.dry_run:
            return None
        r = subprocess.run(cmd)
        if r.returncode != 0 or not res.exists():
            raise RuntimeError(f"distillation run failed: {student}/{tag} (exit {r.returncode})")
    d = json.loads(res.read_text())
    log(f"  {student}/{tag}: best dev top-1 {d['best_score']*100:.2f}% at step {d['best_step']} "
        f"(before KD {d['dev_before']['top1']*100:.2f}%)")
    return d["best_score"]


def pick(cands, log):
    """cands: list of (config, score) in registration order; earlier wins ties."""
    best_cfg, best = cands[0]
    for cfg, sc in cands[1:]:
        if sc is not None and (best is None or sc > best + TIE):
            best_cfg, best = cfg, sc
    log(f"  -> selected {best_cfg} (dev top-1 {best*100:.2f}%)" if best is not None else "  -> (dry run)")
    return best_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-final", action="store_true")
    args = ap.parse_args()
    short = yaml.safe_load(open(args.config)).get("short")
    log = Logger(RAW / short / "distill" / "selection.log")
    log(f"=== distillation selection for {short} (protocol §7)")

    log("stage a: temperature")
    a = [((T, 0.0, 1e-4), run(args, short, PRIMARY, T, 0.0, 1e-4, 200, 50, log=log)) for T in (1, 2, 4)]
    T, alpha, lr = pick(a, log)
    log("stage b: learning rate")
    b = [((T, alpha, lr), dict(a)[(T, alpha, lr)]),
         ((T, alpha, 3e-4), run(args, short, PRIMARY, T, alpha, 3e-4, 200, 50, log=log))]
    T, alpha, lr = pick(b, log)
    log("stage c: CE weight")
    c = [((T, alpha, lr), dict(b)[(T, alpha, lr)]),
         ((T, 0.5, lr), run(args, short, PRIMARY, T, 0.5, lr, 200, 50, log=log))]
    T, alpha, lr = pick(c, log)
    sel = {"T": T, "alpha": alpha, "lr": lr, "stages": {"a": a, "b": b, "c": c}}
    if not args.dry_run:
        write_json(RAW / short / "distill" / "selected.json", sel)
    if args.skip_final:
        return
    log("final runs (1,500 steps, patience 5, one test evaluation each)")
    for student in (PRIMARY, SECONDARY):
        run(args, short, student, T, alpha, lr, 1500, 100, patience=5, final=True, log=log)
    log("=== done")


if __name__ == "__main__":
    main()
