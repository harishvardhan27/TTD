"""
real_cnn_example.py
=====================
Reference script: TT-compress a REAL pretrained CNN (torchvision
ResNet-18 / MobileNetV2) and fine-tune on CIFAR-10.

Requires internet access (torchvision model weights + CIFAR-10 download),
so — like hf_real_model_example.py — this is provided ready-to-run rather
than executed inside this sandbox (no route to download hosts here). Run
on a machine with internet access:

    pip install torch torchvision
    python real_cnn_example.py --model resnet18 --rank 24
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from ttcompress.surgery import replace_conv_with_tt, count_parameters, model_size_mb, benchmark_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "mobilenet_v2"])
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--num_modes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading pretrained {args.model} ...")
    if args.model == "resnet18":
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 10)  # adapt head to CIFAR-10
    else:
        model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

    base_params = count_parameters(model)
    base_size = model_size_mb(model)
    print(f"Baseline: {base_params:,} params, {base_size:.2f} MB")

    print("Compressing Conv2d layers with TT decomposition ...")
    # default predicate already skips the tiny stem conv and depthwise
    # (groups>1) convs, which TTConv2d does not currently support.
    replace_conv_with_tt(model, num_modes=args.num_modes, rank=args.rank)

    tt_params = count_parameters(model)
    tt_size = model_size_mb(model)
    print(f"Compressed: {tt_params:,} params, {tt_size:.2f} MB, ratio={base_params/tt_params:.2f}x")

    # ---------------- Fine-tune on CIFAR-10 ----------------
    print("Loading CIFAR-10 ...")
    tfm_train = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tfm_test = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm_train)
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm_test)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if i % 50 == 0:
                print(f"epoch {epoch} step {i} loss {running_loss / (i+1):.4f}")

        # eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        print(f"epoch {epoch}: test accuracy = {correct/total:.4f}")

    # ---------------- Latency benchmark ----------------
    model.eval().to("cpu")
    example = torch.randn(8, 3, 224, 224)
    lat = benchmark_latency(model, example)
    print("Latency:", lat)

    torch.save(model.state_dict(), "tt_compressed_cnn.pt")
    print("Saved compressed model to tt_compressed_cnn.pt")


if __name__ == "__main__":
    main()
