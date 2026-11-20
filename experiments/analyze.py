"""
Phase 7: generate every table, statistic and figure from raw artifacts.

    python experiments/analyze.py --model deit_small --split v2_dev --suite dev

Reads results/raw/<model>/<split>/<suite>/*.json (+ .logits.pt) and writes
  results/processed/<model>_<split>_<suite>.csv          one row per variant
  results/processed/<model>_<split>_<suite>_paired.csv   method-vs-method paired tests (Holm per family)
  results/tables/<model>_<split>_<suite>.md              markdown tables used by the paper
  results/figures/<model>_<split>_<suite>_*.png          figures
Nothing here is hand-edited; the paper cites these files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, RESULTS  # noqa: E402

from ttcompress.evaluate import mcnemar_exact_p  # noqa: E402

PROC, TABLES, FIGS = RESULTS / "processed", RESULTS / "tables", RESULTS / "figures"
for d in (PROC, TABLES, FIGS):
    d.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (dataviz reference instance, light surface)
C = {"plain": "#2a78d6", "data": "#eb6834", "tt": "#1baf7a", "int8": "#eda100", "other": "#6250d6"}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"


def load(model, split, suite):
    d = RAW / model / split / suite
    rows = []
    for f in sorted(d.glob("*.json")):
        if f.name.endswith(".FAILED.json"):
            r = json.loads(f.read_text())
            rows.append({"name": f.name[:-12], "failed": True, "error": r["error"], "spec": r["spec"]})
            continue
        r = json.loads(f.read_text())
        r["name"] = r["spec"]["name"]
        r["failed"] = False
        rows.append(r)
    return d, rows


def flat(r):
    s, sc = r["spec"], r.get("score", {})
    out = {"name": r["name"], "kind": s.get("kind", "lowrank"), "method": s.get("method", ""),
           "scope": s.get("scope", ""), "policy": s.get("policy", "budget" if "budget" in s else ""),
           "budget": s.get("budget", ""), "allocator": s.get("allocator", ""), "quant": s.get("quant", ""),
           "order": s.get("order", ""), "damp": s.get("damp", ""), "calib": json.dumps(s["calib"]) if "calib" in s else "",
           "min_saving": s.get("min_saving", "")}
    if r["failed"]:
        out.update(failed=True, error=r["error"])
        return out
    out.update({"failed": False, "params": r["params"], "param_ratio": r["param_ratio"], "size_mb": r["size_mb"],
                "size_ratio": r["size_ratio"], "targeted_ratio": r.get("targeted_ratio", ""),
                "top1": sc["top1"], "top1_lo": sc["top1_ci"][0], "top1_hi": sc["top1_ci"][1], "top5": sc["top5"],
                "delta": sc.get("delta_vs_base"), "delta_lo": (sc.get("delta_ci") or [None, None])[0],
                "delta_hi": (sc.get("delta_ci") or [None, None])[1], "mcnemar_p": sc.get("mcnemar_p"),
                "agreement": sc.get("agreement"), "kl": sc.get("kl_to_base"), "n": sc["n"],
                "classes_worse_2plus": r.get("per_class_shift", {}).get("worse_by_2plus")})
    layers = [l for l in r.get("layers", []) if l.get("rank") or l.get("tt_rank")]
    if layers:
        oe = [l["output_rel_error"] for l in layers if l.get("output_rel_error") is not None]
        we = [l["weight_rel_error"] for l in layers if l.get("weight_rel_error") is not None]
        out["mean_out_err"] = round(sum(oe) / len(oe), 4) if oe else ""
        out["sum_sq_out_err"] = round(sum(e * e for e in oe), 4) if oe else ""
        out["mean_w_err"] = round(sum(we) / len(we), 4) if we else ""
    return out


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = running
    return adj


def paired(d, a, b, y):
    """Paired comparison of variant b against variant a (b - a)."""
    la = torch.load(d / f"{a}.logits.pt").float()
    lb = torch.load(d / f"{b}.logits.pt").float()
    ca, cb = la.argmax(1) == y, lb.argmax(1) == y
    diff = cb.float() - ca.float()
    n = len(y)
    m, se = float(diff.mean()), float(diff.std()) / math.sqrt(n)
    nb, nc = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return {"a": a, "b": b, "delta_b_minus_a": round(m, 5), "ci_lo": round(m - 1.96 * se, 5), "ci_hi": round(m + 1.96 * se, 5),
            "a_only_correct": nb, "b_only_correct": nc, "mcnemar_p": mcnemar_exact_p(nb, nc),
            "agreement": round(float((la.argmax(1) == lb.argmax(1)).float().mean()), 5)}


def comparisons(rows):
    """Families of pre-defined pairwise questions, matched on all other spec keys."""
    ok = {r["name"]: r for r in rows if not r["failed"]}
    def key(s, drop):
        return json.dumps({k: v for k, v in s.items() if k not in drop and k != "name"}, sort_keys=True)
    fams = {"A1 plain vs data": ("method", "plain", "data"),
            "A2 uniform vs adaptive": ("policy", "uniform_budget", "budget"),
            "A6 fp32 vs int8": ("quant", None, "int8"),
            "greedy vs dp": ("allocator", None, "dp"),
            "fq vs qf": ("order", None, "qf")}
    out = []
    for fam, (k, va, vb) in fams.items():
        for nb_, rb in ok.items():
            sb = rb["spec"]
            if sb.get(k) != vb:
                continue
            for na_, ra in ok.items():
                sa = ra["spec"]
                if sa.get(k) == va and key(sa, {k}) == key(sb, {k}) and sa.get("kind", "lowrank") == sb.get("kind", "lowrank"):
                    out.append((fam, na_, nb_))
    # iso-parameter family: same method & quantization, different scope/budget, whole-model params within 0.5%
    lr = [r for r in ok.values() if r["spec"].get("kind", "lowrank") == "lowrank" and r["spec"].get("policy", "budget") == "budget"]
    same = ("method", "quant", "order", "damp", "calib", "allocator", "allocator_weights", "min_saving")
    for i, ra in enumerate(lr):
        for rb in lr[i + 1:]:
            sa, sb = ra["spec"], rb["spec"]
            if (all(sa.get(k) == sb.get(k) for k in same)
                    and sa.get("scope") != sb.get("scope") and abs(ra["params"] - rb["params"]) <= 0.005 * ra["params"]):
                out.append(("iso-params scope", ra["name"], rb["name"]))
    return out


def anisotropy(model):
    """Input-covariance anisotropy per layer type from cached calibration covariances."""
    from ttcompress.data import DATA_ROOT
    files = sorted((DATA_ROOT / "cache").glob(f"covs_{model.split('_')[0]}*.pt"))
    stats = {}
    for f in files[:1]:
        covs = torch.load(f)
        for name, C_ in covs.items():
            ev = torch.linalg.eigvalsh(C_.double()).clamp_min(0).flip(0)
            p = ev / ev.sum()
            eff_rank = float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
            k = max(1, len(ev) // 100)
            typ = name.split(".")[-1]
            stats.setdefault(typ, []).append((eff_rank / len(ev), float(p[:k].sum())))
    return {t: {"layers": len(v), "eff_rank_frac_mean": round(sum(a for a, _ in v) / len(v), 4),
                "top1pct_energy_mean": round(sum(b for _, b in v) / len(v), 4)} for t, v in stats.items()}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9, "legend.frameon": False, "lines.linewidth": 2})
    return plt


def fig_budget(flat_rows, base_top1, tag):
    rows = [r for r in flat_rows if not r["failed"] and r["kind"] == "lowrank" and r["scope"] == "attn+mlp"
            and r["policy"] == "budget" and not r["quant"] and not r["allocator"] and not r["damp"]
            and not r["calib"] and not r["min_saving"]]
    if len({r["budget"] for r in rows}) < 2:
        return None
    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for meth in ("plain", "data"):
        pts = sorted((r["budget"], r["top1"], r["top1_lo"], r["top1_hi"]) for r in rows if r["method"] == meth)
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] * 100 for p in pts]
        err = [[(p[1] - p[2]) * 100 for p in pts], [(p[3] - p[1]) * 100 for p in pts]]
        ax.errorbar(xs, ys, yerr=err, color=C[meth], marker="o", markersize=5, capsize=0, elinewidth=1,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{meth} SVD")
        ax.annotate(f"{meth}", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", color=INK, va="center")
    ax.axhline(base_top1 * 100, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("FP32 baseline", (ax.get_xlim()[0], base_top1 * 100), xytext=(2, 3), textcoords="offset points",
                color=MUTED, fontsize=8)
    ax.set_xlabel("budget b (fraction of attn+MLP weights kept)")
    ax.set_ylabel("top-1 (%)  ±95% Wilson CI")
    ax.legend(loc="lower right")
    fig.tight_layout()
    p = FIGS / f"{tag}_acc_vs_budget.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_layers(rows_by_name, names, tag):
    """Held-out output error and allocated rank fraction per layer for given variants."""
    sel = [rows_by_name[n] for n in names if n in rows_by_name and rows_by_name[n].get("layers")]
    if not sel:
        return None
    plt = setup_mpl()
    types = ["qkv", "proj", "fc1", "fc2"]
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.4), sharex=True, sharey="row")
    for r in sel:
        meth = r["spec"].get("method", "data")
        for j, t in enumerate(types):
            L = [l for l in r["layers"] if l["name"].endswith("." + t)]
            blocks = [int(l["name"].split(".")[1]) for l in L]
            err = [l.get("output_rel_error") or 0 for l in L]
            kept = [l["rank"] * sum(l["shape"]) / (l["shape"][0] * l["shape"][1]) if l.get("rank") else 1.0 for l in L]
            for ax, v in ((axes[0, j], err), (axes[1, j], kept)):
                ax.plot(blocks, v, color=C[meth], marker="o", markersize=3.5, linewidth=1.5,
                        markeredgecolor=SURFACE, markeredgewidth=0.8, label=f"{meth} SVD")
            axes[0, j].set_title(t, fontsize=9, color=INK)
    axes[0, 0].set_ylabel("held-out output\nrel. error")
    axes[1, 0].set_ylabel("params kept\n(fraction of layer)")
    from matplotlib.ticker import MaxNLocator
    for ax in axes[1]:
        ax.set_xlabel("block")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(" vs ".join(r["name"] for r in sel), fontsize=8, color=MUTED)
    fig.tight_layout()
    p = FIGS / f"{tag}_per_layer.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def md_table(rows, cols, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def pct(x, d=2):
    return "" if x in (None, "") else f"{x*100:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--suite", required=True)
    args = ap.parse_args()
    tag = f"{args.model}_{args.split}_{args.suite}"
    d, rows = load(args.model, args.split, args.suite)
    y = torch.load(RAW / args.model / args.split / "labels.pt")
    fr = [flat(r) for r in rows]
    write_csv(PROC / f"{tag}.csv", fr)

    pairs = []
    for fam, a, b in comparisons(rows):
        pairs.append({"family": fam, **paired(d, a, b, y)})
    for fam in {p["family"] for p in pairs}:
        idx = [i for i, p in enumerate(pairs) if p["family"] == fam]
        for i, adj in zip(idx, holm([pairs[i]["mcnemar_p"] for i in idx])):
            pairs[i]["mcnemar_p_holm"] = adj
    if pairs:
        write_csv(PROC / f"{tag}_paired.csv", pairs)

    base = next((r for r in fr if r["kind"] == "baseline" and not r["failed"]), None)
    disp = []
    for r in fr:
        if r["failed"]:
            disp.append({"name": r["name"], "top1": "FAILED: " + r["error"][:60]})
            continue
        disp.append({"name": r["name"], "ratio": f"{r['param_ratio']:.2f}x", "size": f"{r['size_mb']:.1f}",
                     "top1": f"{pct(r['top1'])} [{pct(r['top1_lo'],1)}, {pct(r['top1_hi'],1)}]",
                     "delta": "" if r["delta"] is None else f"{r['delta']*100:+.2f} [{r['delta_lo']*100:+.1f}, {r['delta_hi']*100:+.1f}]",
                     "agree": pct(r["agreement"], 1), "kl": "" if r["kl"] is None else f"{r['kl']:.3f}",
                     "err": r.get("mean_out_err", "")})
    md = [f"# {tag}\n", f"Generated by `experiments/analyze.py` from `results/raw/{args.model}/{args.split}/{args.suite}/`.\n",
          md_table(disp, ["name", "ratio", "size", "top1", "delta", "agree", "kl", "err"],
                   ["variant", "params ratio", "size MB", "top-1 % [95% CI]", "Δ vs FP32 pp [95% CI]",
                    "agreement %", "KL", "mean held-out layer out-err"])]
    if pairs:
        md.append("\n## Paired comparisons (b − a), exact McNemar, Holm-adjusted within family\n")
        md.append(md_table([{**p, "delta": f"{p['delta_b_minus_a']*100:+.2f} [{p['ci_lo']*100:+.1f}, {p['ci_hi']*100:+.1f}]",
                             "p": f"{p['mcnemar_p']:.2g}", "ph": f"{p['mcnemar_p_holm']:.2g}"} for p in pairs],
                           ["family", "a", "b", "delta", "a_only_correct", "b_only_correct", "p", "ph"],
                           ["family", "a", "b", "Δ top-1 pp [95% CI]", "only a correct", "only b correct", "p", "p (Holm)"]))
    an = anisotropy(args.model)
    if an:
        md.append("\n## Input anisotropy (calibration covariances)\n")
        md.append("effective rank = exp(entropy of normalised eigenvalues) / dim; top-1% energy = share of the trace in the largest 1% of eigen-directions.\n")
        md.append(md_table([{"type": k, **v} for k, v in an.items()], ["type", "layers", "eff_rank_frac_mean", "top1pct_energy_mean"],
                           ["layer input", "layers", "effective rank / dim", "top-1% energy"]))
    (TABLES / f"{tag}.md").write_text("\n".join(md), encoding="utf-8")

    figs = []
    if base:
        figs.append(fig_budget(fr, base["top1"], tag))
    byname = {r["name"]: r for r in rows if not r["failed"]}
    cand = [n for n in ("plain_attn+mlp_b0.6", "data_attn+mlp_b0.6") if n in byname]
    figs.append(fig_layers(byname, cand, tag))
    print(f"wrote {PROC / (tag + '.csv')}, {TABLES / (tag + '.md')}, figures: {[str(f) for f in figs if f]}")


if __name__ == "__main__":
    main()
