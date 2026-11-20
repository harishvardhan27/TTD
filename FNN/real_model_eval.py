"""
real_model_eval.py
=====================
The strongest validation available in this sandbox: a REAL pretrained
MobileNetV2 (ImageNet-trained weights, fetched from a GitHub release --
see README for provenance) evaluated on REAL ImageNet photos (200 images
spanning all 1000 classes, fetched from EliSchwartz/imagenet-sample-images),
before and after TT-compressing its conv layers.

This directly answers the two open questions from the project review:
  1. Does TT compression work on genuinely trained weights (not just
     hand-constructed decaying-spectrum synthetic matrices)?
  2. Does it preserve real classification accuracy, not just a synthetic
     distillation proxy?

No GPU is used or needed (this runs on CPU); no ImageNet val set download
was needed (200 real, correctly-labeled photos were sufficient for an
honest, if small-sample, accuracy comparison).
"""

import json
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import timm

from ttcompress.surgery import replace_conv_with_tt, count_parameters, model_size_mb, benchmark_latency

torch.manual_seed(0)

HERE = Path(__file__).parent
SAMPLE_DIR = HERE / "imagenet_sample"
SAMPLE_LIST = HERE / "sample_files.txt"

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_samples():
    items = []
    with open(SAMPLE_LIST) as f:
        for line in f:
            idx, fname = line.strip().split("\t")
            path = SAMPLE_DIR / fname
            if path.exists():
                items.append((int(idx), path))
    return items


@torch.no_grad()
def evaluate_top1(model, samples, batch_size=16):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    imgs_batch, labels_batch = [], []

    def flush():
        nonlocal correct, total
        if not imgs_batch:
            return
        x = torch.stack(imgs_batch)
        logits = model(x)
        preds = logits.argmax(dim=1)
        for p, l in zip(preds.tolist(), labels_batch):
            all_preds.append(p)
            correct += int(p == l)
            total += 1
        imgs_batch.clear()
        labels_batch.clear()

    for label, path in samples:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        imgs_batch.append(preprocess(img))
        labels_batch.append(label)
        if len(imgs_batch) == batch_size:
            flush()
    flush()

    return correct / total, total, all_preds


def main():
    samples = load_samples()
    print(f"Loaded {len(samples)} real ImageNet images spanning {len(set(l for l,_ in samples))} classes")

    print("\nLoading REAL pretrained MobileNetV2 (ImageNet weights, via timm/GitHub release)...")
    model = timm.create_model("mobilenetv2_100", pretrained=True)
    base_params = count_parameters(model)
    base_size = model_size_mb(model)
    print(f"Baseline: {base_params:,} params, {base_size:.2f} MB")

    print("\nEvaluating baseline top-1 accuracy on real images...")
    t0 = time.time()
    acc_base, n, preds_base = evaluate_top1(model, samples)
    print(f"Baseline top-1 accuracy: {acc_base:.4f} ({int(acc_base*n)}/{n})  [{time.time()-t0:.1f}s]")

    results = {"baseline": {"params": base_params, "size_mb": round(base_size, 3),
                             "top1_acc": round(acc_base, 4), "n_images": n}}

    for tt_rank in [32, 16]:
        print(f"\n{'='*60}\nTT-compressing conv layers at rank={tt_rank}\n{'='*60}")
        compressed = copy.deepcopy(model)
        replace_conv_with_tt(compressed, num_modes=2, rank=tt_rank)

        tt_params = count_parameters(compressed)
        tt_size = model_size_mb(compressed)
        print(f"Compressed: {tt_params:,} params, {tt_size:.2f} MB, "
              f"ratio={base_params/tt_params:.2f}x")

        print("Evaluating TT-compressed (pre-fine-tune) top-1 accuracy on real images...")
        t0 = time.time()
        acc_tt, n2, preds_tt = evaluate_top1(compressed, samples)
        agreement = sum(int(a == b) for a, b in zip(preds_base, preds_tt)) / len(preds_base)
        print(f"TT rank={tt_rank} top-1 accuracy: {acc_tt:.4f} ({int(acc_tt*n2)}/{n2})  "
              f"[{time.time()-t0:.1f}s]  agreement-with-baseline-predictions={agreement:.4f}")

        x = torch.randn(8, 3, 224, 224)
        lat = benchmark_latency(compressed, x, n_warmup=2, n_iters=8)
        lat_base = benchmark_latency(model, x, n_warmup=2, n_iters=8)

        results[f"tt_rank_{tt_rank}"] = {
            "params": tt_params, "size_mb": round(tt_size, 3),
            "compression_ratio": round(base_params / tt_params, 2),
            "top1_acc": round(acc_tt, 4), "n_images": n2,
            "agreement_with_baseline": round(agreement, 4),
            "latency_ms": round(lat["mean_ms"], 2),
            "baseline_latency_ms": round(lat_base["mean_ms"], 2),
        }

    print("\n" + "=" * 60)
    print("FINAL SUMMARY (real pretrained model, real ImageNet images)")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    with open(HERE / "results_real_model.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_real_model.json")


if __name__ == "__main__":
    main()
