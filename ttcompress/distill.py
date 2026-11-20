"""
Knowledge-distillation recovery fine-tuning, sized for a CPU laptop.

Objective (Hinton et al., 2015):

    L = alpha * CE(student, y) + (1 - alpha) * T^2 * KL(softmax(t/T) || softmax(s/T))

* Teacher logits are computed once and cached (optionally also for the
  horizontally flipped view), so a step costs one student forward+backward.
* Only `trainable` parameters are updated (default: low-rank factors and
  LayerNorms); optimiser state stays small and drift from the pretrained
  model is limited.
* Resumable: the trainable state, optimiser, scheduler, RNG and history are
  checkpointed every `eval_every` steps to `ckpt_path`.
* Model selection: `eval_fn(student) -> float` (higher is better) is called
  every `eval_every` steps; it must use a *dev* split (ImageNetV2-dev), never
  the test split. The best state is restored at the end.

Caveat (documented in the protocol): the CE term uses training-side labels.
The transfer set is dominated by 10 Imagenette classes, so alpha > 0 can shift
the classifier's class prior; this is measured, not assumed away.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import normalize
from .lowrank import LowRankLinear


def default_trainable(model: nn.Module) -> List[nn.Parameter]:
    seen, out = set(), []
    for m in model.modules():
        if isinstance(m, (LowRankLinear, nn.LayerNorm)):
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
    return out


@torch.no_grad()
def teacher_logits(teacher: nn.Module, x_uint8: torch.Tensor, data_cfg: dict, batch_size: int = 32,
                   hflip: bool = False) -> torch.Tensor:
    teacher.eval()
    out = []
    for i in range(0, len(x_uint8), batch_size):
        xb = x_uint8[i:i + batch_size]
        if hflip:
            xb = xb.flip(-1)
        out.append(teacher(normalize(xb, data_cfg)).float())
    return torch.cat(out)


def kd_loss(s_logits: torch.Tensor, t_logits: torch.Tensor, labels: Optional[torch.Tensor], alpha: float,
            temperature: float) -> torch.Tensor:
    T = temperature
    kd = F.kl_div(F.log_softmax(s_logits / T, 1), F.log_softmax(t_logits / T, 1),
                  reduction="batchmean", log_target=True) * T * T
    if alpha <= 0 or labels is None:
        return kd
    return alpha * F.cross_entropy(s_logits, labels) + (1 - alpha) * kd


def distill(student: nn.Module, x_uint8: torch.Tensor, t_logits: torch.Tensor, data_cfg: dict, *,
            steps: int, labels: Optional[torch.Tensor] = None, alpha: float = 0.0, temperature: float = 2.0,
            batch_size: int = 16, accum: int = 2, lr: float = 1e-4, weight_decay: float = 0.0,
            t_logits_flip: Optional[torch.Tensor] = None, trainable: Optional[List[nn.Parameter]] = None,
            eval_fn: Optional[Callable[[nn.Module], float]] = None, eval_every: int = 50,
            patience: Optional[int] = None, ckpt_path: Optional[Path] = None, log_every: int = 10, seed: int = 0,
            log: Callable[[str], None] = print) -> Dict[str, object]:
    """Distil into `student` in place. Returns the training history, the
    best eval step/score and the wall time. With `eval_fn`, the best
    trainable state (by eval score) is restored before returning; `patience`
    (in evaluations) enables early stopping."""
    if alpha > 0 and labels is None:
        raise ValueError("alpha > 0 requires labels")
    params = trainable if trainable is not None else default_trainable(student)
    for p in student.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    g = torch.Generator().manual_seed(seed)

    state = {"step": 0, "hist": {"step": [], "loss": [], "sec": []}, "evals": [], "best": None, "best_step": -1,
             "best_params": None, "elapsed": 0.0, "ema": None, "bad_evals": 0}
    if ckpt_path is not None and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, weights_only=False)
        with torch.no_grad():
            for p, v in zip(params, ck["params"]):
                p.copy_(v)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        g.set_state(ck["rng"])
        state.update({k: ck[k] for k in state if k in ck})
        log(f"  resumed from {ckpt_path} at step {state['step']}")

    def snapshot():
        return [p.detach().clone() for p in params]

    def save():
        if ckpt_path is None:
            return
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(ckpt_path) + ".tmp")
        torch.save({"params": snapshot(), "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": g.get_state(),
                    **{k: state[k] for k in state}}, tmp)
        tmp.replace(ckpt_path)

    n = len(x_uint8)
    t_start = time.time() - state["elapsed"]
    if eval_fn is not None and state["step"] == 0 and state["best"] is None:
        student.eval()
        s0 = eval_fn(student)
        state["evals"].append((0, s0))
        state["best"], state["best_step"], state["best_params"] = s0, 0, snapshot()
        log(f"  eval step 0: {s0:.4f}")
    student.train()
    stopped_early = False
    for step in range(state["step"], steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(accum):
            idx = torch.randint(0, n, (batch_size,), generator=g)
            xb, tl = x_uint8[idx], t_logits[idx]
            if t_logits_flip is not None:
                flip = torch.rand(batch_size, generator=g) < 0.5
                xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
                tl = torch.where(flip[:, None], t_logits_flip[idx], tl)
            yb = labels[idx] if labels is not None else None
            loss = kd_loss(student(normalize(xb, data_cfg)), tl, yb, alpha, temperature)
            (loss / accum).backward()
            tot += loss.item() / accum
        opt.step()
        sched.step()
        state["ema"] = tot if state["ema"] is None else 0.9 * state["ema"] + 0.1 * tot
        state["step"] = step + 1
        state["elapsed"] = time.time() - t_start
        if step % log_every == 0 or step == steps - 1:
            state["hist"]["step"].append(step)
            state["hist"]["loss"].append(round(state["ema"], 5))
            state["hist"]["sec"].append(round(state["elapsed"], 1))
            log(f"  step {step:5d}/{steps} loss(ema)={state['ema']:.4f} lr={sched.get_last_lr()[0]:.2e} "
                f"{state['elapsed']:.0f}s")
        if (step + 1) % eval_every == 0 or step == steps - 1:
            if eval_fn is not None:
                student.eval()
                sc = eval_fn(student)
                student.train()
                state["evals"].append((step + 1, sc))
                if state["best"] is None or sc > state["best"]:
                    state["best"], state["best_step"], state["best_params"] = sc, step + 1, snapshot()
                    state["bad_evals"] = 0
                else:
                    state["bad_evals"] += 1
                log(f"  eval step {step+1}: {sc:.4f} (best {state['best']:.4f} @ {state['best_step']})")
            save()
            if patience is not None and state["bad_evals"] >= patience:
                log(f"  early stop: {patience} evaluations without improvement")
                stopped_early = True
                break
    if state["best_params"] is not None:
        with torch.no_grad():
            for p, v in zip(params, state["best_params"]):
                p.copy_(v)
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    return {"hist": state["hist"], "evals": state["evals"], "best_step": state["best_step"],
            "best_score": state["best"], "steps_run": state["step"], "stopped_early": stopped_early,
            "train_seconds": round(state["elapsed"], 1), "samples_seen": state["step"] * batch_size * accum}
