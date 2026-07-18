"""
validate_finalized_project.py
================================
Validates the finalized project's actual method (plain low-rank
factorization of attention layers) against real ViT-Base weights and
real ImageNet images -- the same rigor applied to the superseded
Tensor-Train approach, now applied to what replaced it.
"""
import copy
import torch
import timm
from timm.data import resolve_data_config, create_transform

from low_rank_compress import replace_linear_with_lowrank, LowRankLinear
from compress_model import count_parameters, model_size_mb, benchmark_latency
from real_model_eval import load_samples


@torch.no_grad()
def evaluate_top1_with_transform(model, samples, transform, batch_size=16):
    from PIL import Image
    model.eval()
    correct, total = 0, 0
    imgs, labels = [], []

    def flush():
        nonlocal correct, total
        if not imgs:
            return
        x = torch.stack(imgs)
        preds = model(x).argmax(dim=1)
        for p, l in zip(preds.tolist(), labels):
            correct += int(p == l)
            total += 1
        imgs.clear()
        labels.clear()

    for label, path in samples:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        imgs.append(transform(img))
        labels.append(label)
        if len(imgs) == batch_size:
            flush()
    flush()
    return correct / total, total


def main():
    samples = load_samples()
    eval_samples = samples[:100]

    print("Loading real pretrained ViT-Base...")
    model = timm.create_model("vit_base_patch16_224_miil", pretrained=True)
    model.eval()
    base_params = count_parameters(model)
    base_size = model_size_mb(model)

    # Use the model's OWN correct preprocessing config (caught a real bug:
    # this variant expects mean=0,std=1 raw-pixel input, not standard
    # ImageNet normalization -- using the wrong one silently tanked
    # accuracy to 49% before this fix).
    data_cfg = resolve_data_config({}, model=model)
    transform = create_transform(**data_cfg)
    print(f"Using model's real data config: mean={data_cfg.get('mean')}, std={data_cfg.get('std')}")

    acc_base, n = evaluate_top1_with_transform(model, eval_samples, transform)
    print(f"Baseline: {base_params:,} params, {base_size:.2f} MB, top-1={acc_base:.4f} ({n} real images)")

    for target_err in [0.15, 0.30]:
        print(f"\n{'='*60}\nLow-rank compressing attention layers, target_rel_error={target_err}\n{'='*60}")
        m = copy.deepcopy(model)
        replace_linear_with_lowrank(m, target_rel_error=target_err, verbose=True)
        m.eval()

        tt_params = count_parameters(m)
        tt_size = model_size_mb(m)
        acc, _ = evaluate_top1_with_transform(m, eval_samples, transform)

        x = torch.randn(4, 3, 224, 224)
        lat = benchmark_latency(m, x, n_warmup=2, n_iters=5)
        lat_base = benchmark_latency(model, x, n_warmup=2, n_iters=5)

        print(f"\nResult: {tt_params:,} params ({base_params/tt_params:.2f}x), "
              f"{tt_size:.2f} MB, top-1={acc:.4f} (baseline={acc_base:.4f}), "
              f"latency={lat['mean_ms']:.1f}ms (baseline={lat_base['mean_ms']:.1f}ms)")


if __name__ == "__main__":
    main()
