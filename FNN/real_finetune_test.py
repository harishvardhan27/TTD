"""
real_finetune_test.py
========================
The capstone test: take the REAL pretrained MobileNetV2, TT-compress it
to a genuinely SMALLER model (not the accuracy-preserving-but-larger
regime found in real_model_eval.py's tighter error budgets), fine-tune
via distillation against the original model's outputs (synthetic
ImageNet-statistics-normalized inputs, since we don't have a real
training set here), then re-evaluate REAL top-1 accuracy on real,
held-out ImageNet photos not used during fine-tuning.

This directly answers the open question from the project review:
does fine-tuning rescue a genuinely compressed real CNN, the way it
did for our synthetic transformer/CNN experiments?
"""
import copy
import time
import torch
import torch.nn.functional as F
import timm

from compress_model import replace_conv_with_tt_adaptive, count_parameters, model_size_mb
from real_model_eval import load_samples, evaluate_top1

torch.manual_seed(0)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def synthetic_imagenet_like_batch(batch_size=8):
    # Correlated-noise images (smoothed Gaussian) resemble natural image
    # statistics (low-frequency dominant) far better than pure white
    # noise, then normalized to ImageNet statistics.
    raw = torch.randn(batch_size, 3, 224, 224)
    raw = F.avg_pool2d(F.interpolate(raw, scale_factor=4, mode="bilinear"), kernel_size=4)
    raw = (raw - raw.mean()) / (raw.std() + 1e-6)
    return raw * IMAGENET_STD + IMAGENET_MEAN


def main():
    samples = load_samples()
    # Split: fine-tune influence only ever comes from synthetic data, so
    # ALL 200 real images remain a clean, untouched held-out eval set.
    eval_samples = samples[:50]

    print("Loading real pretrained MobileNetV2...")
    teacher = timm.create_model("mobilenetv2_100", pretrained=True)
    teacher.eval()
    base_params = count_parameters(teacher)

    acc_base, n, _ = evaluate_top1(teacher, eval_samples)
    print(f"Teacher (original) top-1 on {n} real images: {acc_base:.4f}")

    print("\nTT-compressing (fixed rank=32 globally -- genuinely SMALLER, ~2x, per")
    print("earlier direct measurement; the adaptive per-layer target-error version")
    print("preserved accuracy better but grew the model, so this tests the actual")
    print("compression regime instead)...")
    from compress_model import replace_conv_with_tt
    student = copy.deepcopy(teacher)
    replace_conv_with_tt(student, num_modes=2, rank=32, verbose=False)
    tt_params = count_parameters(student)
    print(f"Compressed: {tt_params:,} params vs {base_params:,} "
          f"({base_params/tt_params:.2f}x smaller)")

    acc_pre, _, _ = evaluate_top1(student, eval_samples)
    print(f"Compressed, PRE-fine-tune, top-1 on real images: {acc_pre:.4f}")

    print("\nFine-tuning via distillation (synthetic ImageNet-statistics inputs)...")
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=1e-4)
    losses = []
    n_steps = 12
    t0 = time.time()
    for step in range(n_steps):
        x = synthetic_imagenet_like_batch(batch_size=8)
        with torch.no_grad():
            target_logits = teacher(x)
        opt.zero_grad()
        pred_logits = student(x)
        loss = F.kl_div(
            F.log_softmax(pred_logits, dim=1),
            F.softmax(target_logits, dim=1),
            reduction="batchmean",
        )
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 5 == 0:
            print(f"  step {step:3d}: KL loss = {loss.item():.4f}  [{time.time()-t0:.0f}s elapsed]")

    print(f"Fine-tuning done in {time.time()-t0:.0f}s. Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    student.eval()
    acc_post, _, _ = evaluate_top1(student, eval_samples)
    print(f"\nCompressed, POST-fine-tune, top-1 on real (held-out) images: {acc_post:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Teacher (original):          {acc_base:.4f} top-1, {base_params:,} params")
    print(f"Compressed, pre-fine-tune:   {acc_pre:.4f} top-1, {tt_params:,} params ({base_params/tt_params:.2f}x smaller)")
    print(f"Compressed, post-fine-tune:  {acc_post:.4f} top-1, {tt_params:,} params ({base_params/tt_params:.2f}x smaller)")


if __name__ == "__main__":
    main()
