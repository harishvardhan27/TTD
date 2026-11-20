"""
Image data for CPU-only experiments, with an explicit train/eval separation.

Evaluation side (never used for calibration, distillation or fitting):
  * ``v2_dev``  ImageNetV2 matched-frequency (Recht et al., 2019), 2 images per
    class (2,000): hyperparameter / checkpoint selection only.
  * ``v2_test`` the remaining 8 images per class (8,000): final reporting only.
  ImageNetV2 folders are named by ImageNet-1k class index. The per-class split
  is a fixed seeded permutation of the sorted file names (seed 0).

Training side (calibration statistics, distillation transfer set):
  * ``imagenette_train`` / ``imagenette_val``: fast.ai Imagenette2-320 (10
    classes). NOTE: most files are ImageNet-1k *train* images (audit C1), which
    is why Imagenette is training-side data in this project.
  * ``imagenet_sample``: EliSchwartz/imagenet-sample-images, 1 image per class,
    also ImageNet-train provenance (duplicates of Imagenette files exist).
  * ``transfer``: union of the three above.

Images are preprocessed once per data config (resize + center crop, uint8, no
normalisation) and cached, so evaluating many compressed variants only costs
forward passes. Normalisation uses the model's own timm data config.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

DATA_ROOT = Path(os.environ.get("TTC_DATA", Path(__file__).resolve().parent.parent / "data"))
V2_DIR = "imagenetv2/imagenetv2-matched-frequency-format-val"
V2_DEV_PER_CLASS = 2
SPLIT_SEED = 0

EVAL_SPLITS = ("v2_dev", "v2_test")
TRAIN_SPLITS = ("imagenette_train", "imagenette_val", "imagenet_sample", "transfer")

# Imagenette WordNet ids -> ImageNet-1k class index (verified against the
# sorted WordNet-id list of the 1000-class sample set)
IMAGENETTE_WNID_TO_IDX = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491, "n03028079": 497,
    "n03394916": 566, "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701,
}

Item = Tuple[int, Path]


# --------------------------------------------------------------------------
# Item lists
# --------------------------------------------------------------------------

def imagenet_sample_items(root: Path = DATA_ROOT) -> List[Item]:
    d = root / "imagenet-sample-images"
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png") and p.name.startswith("n"))
    wnids = sorted({p.name.split("_")[0] for p in files})
    if len(wnids) != 1000:
        raise RuntimeError(f"expected 1000 ImageNet classes in {d}, found {len(wnids)}")
    idx = {w: i for i, w in enumerate(wnids)}
    return [(idx[p.name.split("_")[0]], p) for p in files]


def imagenette_items(split: str, root: Path = DATA_ROOT) -> List[Item]:
    d = root / "imagenette2-320" / split
    items = []
    for wnid in sorted(IMAGENETTE_WNID_TO_IDX):
        for p in sorted((d / wnid).iterdir()):
            items.append((IMAGENETTE_WNID_TO_IDX[wnid], p))
    return items


def imagenetv2_split(split: str, root: Path = DATA_ROOT) -> List[Item]:
    d = root / V2_DIR
    if not d.exists():
        raise FileNotFoundError(f"ImageNetV2 not found at {d}")
    items = []
    for c in range(1000):
        files = sorted((d / str(c)).iterdir())
        g = torch.Generator().manual_seed(SPLIT_SEED * 100003 + c)
        perm = torch.randperm(len(files), generator=g).tolist()
        dev = set(perm[:V2_DEV_PER_CLASS])
        for i, p in enumerate(files):
            if (i in dev) == (split == "v2_dev"):
                items.append((c, p))
    return items


def get_items(name: str, role: Optional[str] = None) -> List[Item]:
    """Item list for a named split. `role` ("eval" | "train"), when given, is
    enforced so that evaluation images can never leak into calibration or
    distillation by a typo."""
    if role == "eval" and name not in EVAL_SPLITS:
        raise ValueError(f"{name} is not an evaluation split ({EVAL_SPLITS})")
    if role == "train" and name not in TRAIN_SPLITS:
        raise ValueError(f"{name} is not a training-side split ({TRAIN_SPLITS})")
    if name in EVAL_SPLITS:
        return imagenetv2_split(name)
    if name == "imagenet_sample":
        return imagenet_sample_items()
    if name.startswith("imagenette_"):
        return imagenette_items(name.split("_", 1)[1])
    if name == "transfer":
        return imagenette_items("train") + imagenette_items("val") + imagenet_sample_items()
    raise ValueError(name)


def subset(items: List[Item], limit: Optional[int], seed: int) -> List[Item]:
    if limit is None or limit >= len(items):
        return items
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(items), generator=g)[:limit].sort().values.tolist()
    return [items[i] for i in idx]


# --------------------------------------------------------------------------
# Manifests and overlap checks
# --------------------------------------------------------------------------

def _rel(p: Path) -> str:
    return p.relative_to(DATA_ROOT).as_posix()


def write_manifest(name: str, out_dir: Optional[Path] = None) -> Path:
    """Record exact file list, labels and SHA-1 of every file of a split."""
    items = get_items(name)
    rows = [{"path": _rel(p), "label": c, "sha1": hashlib.sha1(p.read_bytes()).hexdigest()} for c, p in items]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    out_dir = out_dir or DATA_ROOT / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps({"split": name, "n": len(rows), "n_classes": len({r['label'] for r in rows}),
                                "sha256": digest, "items": rows}, indent=1))
    return path


def manifest_digest(name: str) -> Optional[str]:
    p = DATA_ROOT / "manifests" / f"{name}.json"
    return json.loads(p.read_text())["sha256"] if p.exists() else None


def _fingerprint(p: Path, size: int = 16) -> torch.Tensor:
    from PIL import Image
    a = torch.tensor(list(Image.open(p).convert("L").resize((size, size)).getdata()), dtype=torch.float32)
    a = a - a.mean()
    return a / a.norm().clamp_min(1e-6)


def near_duplicates(eval_items: List[Item], train_items: List[Item], threshold: float = 0.98) -> List[dict]:
    """Same-label near-duplicate search via 16x16 grayscale fingerprints
    (robust to resizing / re-encoding, not to crops or flips). Returns the
    eval images whose best same-class training-side match exceeds
    `threshold`, with the matching file."""
    by_class: Dict[int, List[Path]] = {}
    for c, p in train_items:
        by_class.setdefault(c, []).append(p)
    fp_cache: Dict[Path, torch.Tensor] = {}

    def fp(p):
        if p not in fp_cache:
            fp_cache[p] = _fingerprint(p)
        return fp_cache[p]

    hits = []
    for c, p in eval_items:
        cands = by_class.get(c)
        if not cands:
            continue
        F = torch.stack([fp(q) for q in cands])
        sims = F @ fp(p)
        j = int(sims.argmax())
        if float(sims[j]) >= threshold:
            hits.append({"eval": _rel(p), "train": _rel(cands[j]), "similarity": round(float(sims[j]), 4)})
    return hits


# --------------------------------------------------------------------------
# Cached preprocessing
# --------------------------------------------------------------------------

def _uint8_transform(data_cfg: dict):
    """timm eval transform minus ToTensor/Normalize (resize shorter side to
    size/crop_pct, center crop)."""
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode

    size = data_cfg["input_size"][-1]
    crop_pct = data_cfg.get("crop_pct", 0.875)
    interp = {"bicubic": InterpolationMode.BICUBIC, "bilinear": InterpolationMode.BILINEAR}.get(
        data_cfg.get("interpolation", "bilinear"), InterpolationMode.BILINEAR)
    scale = math.floor(size / crop_pct)
    return T.Compose([T.Resize(scale, interpolation=interp), T.CenterCrop(size), T.PILToTensor()])


def load_cached(name: str, data_cfg: dict, limit: Optional[int] = None, seed: int = 0,
                role: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (uint8 images (N,3,H,W), int64 labels (N,)), cached on disk.
    `limit` takes a fixed seeded random subset."""
    from PIL import Image

    items = subset(get_items(name, role), limit, seed)
    files_hash = hashlib.sha1("\n".join(_rel(p) for _, p in items).encode()).hexdigest()[:10]
    key_src = json.dumps({"n": name, "cfg": {k: data_cfg.get(k) for k in ("input_size", "crop_pct", "interpolation")},
                          "files": files_hash}, sort_keys=True)
    key = hashlib.sha1(key_src.encode()).hexdigest()[:12]
    cache = DATA_ROOT / "cache" / f"{name}_{key}.pt"
    if cache.exists():
        # mmap: image pages are file-backed, so the OS can evict and re-read them under
        # memory pressure instead of holding ~1.2 GB (v2_test) of anonymous memory.
        d = torch.load(cache, mmap=True)
        return d["x"], d["y"]
    tf = _uint8_transform(data_cfg)
    xs, ys = [], []
    for label, path in items:
        xs.append(tf(Image.open(path).convert("RGB")))
        ys.append(label)
    x, y = torch.stack(xs), torch.tensor(ys)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y, "files": [_rel(p) for _, p in items]}, cache)
    return x, y


def normalize(x_uint8: torch.Tensor, data_cfg: dict) -> torch.Tensor:
    mean = torch.tensor(data_cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(data_cfg["std"]).view(1, 3, 1, 1)
    return (x_uint8.float() / 255.0 - mean) / std


def batches(x_uint8: torch.Tensor, data_cfg: dict, batch_size: int):
    for i in range(0, len(x_uint8), batch_size):
        yield normalize(x_uint8[i:i + batch_size], data_cfg)
