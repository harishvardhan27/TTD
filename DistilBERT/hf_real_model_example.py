"""
hf_real_model_example.py
=========================
End-to-end example: TT-compress a REAL pretrained Hugging Face model.

This script needs internet access to download model weights and the
SST-2 dataset from the Hugging Face Hub, so it is provided as a
ready-to-run reference rather than executed inside the sandbox that
produced this project (which has no route to huggingface.co). Run it
on your laptop / edge dev box / CI runner with internet access:

    pip install -r requirements.txt
    python hf_real_model_example.py --model distilbert-base-uncased --rank 32

Swap --model for any of: distilbert-base-uncased, gpt2, huawei-noah/TinyBERT_General_4L_312D,
TinyLlama/TinyLlama-1.1B-Chat-v1.0, google/mobilebert-uncased (a few need
architecture-specific tweaks to the layer name filter below -- see the
NOTE next to `predicate`).
"""

import argparse
import copy
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import evaluate as hf_evaluate

from compress_model import replace_linear_with_tt, count_parameters, model_size_mb, benchmark_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--num_modes", type=int, default=3)
    parser.add_argument("--epochs", type=float, default=1.0)
    args = parser.parse_args()

    print(f"Loading {args.model} + tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    base_params = count_parameters(model)
    base_size = model_size_mb(model)
    print(f"Baseline: {base_params:,} params, {base_size:.2f} MB")

    # NOTE: layer-name filter is architecture specific. For DistilBERT /
    # BERT-family models the big feed-forward matrices are named
    # `...intermediate.dense` and `...output.dense` inside each
    # transformer block; attention q/k/v/o projections also match
    # `nn.Linear` and can be included too (set include_attention=True).
    include_attention = False

    def predicate(name, module):
        big_enough = module.in_features >= 256 and module.out_features >= 256
        is_ffn = ("intermediate.dense" in name) or ("output.dense" in name and "attention" not in name)
        is_attn = "attention" in name and ("query" in name or "key" in name or "value" in name or "output.dense" in name)
        if include_attention:
            return big_enough and (is_ffn or is_attn)
        return big_enough and is_ffn

    print("Compressing feed-forward layers with TT decomposition ...")
    replace_linear_with_tt(model, num_modes=args.num_modes, rank=args.rank, predicate=predicate)

    tt_params = count_parameters(model)
    tt_size = model_size_mb(model)
    print(f"Compressed: {tt_params:,} params, {tt_size:.2f} MB, ratio={base_params/tt_params:.2f}x")

    # ---------------- Fine-tune on SST-2 (GLUE) to recover accuracy ----------------
    print("Loading SST-2 ...")
    ds = load_dataset("glue", "sst2")

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    metric = hf_evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir="./tt_finetune_out",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        learning_rate=3e-4,  # TT cores usually need a slightly higher LR than a full fine-tune
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning TT-compressed model ...")
    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    # ---------------- Latency benchmark on CPU ----------------
    model.eval().to("cpu")
    example = tokenizer(["This movie was great!"] * 8, return_tensors="pt", padding="max_length", max_length=128)
    with torch.no_grad():
        for _ in range(3):
            model(**example)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            model(**example)
            times.append(time.perf_counter() - t0)
    mean_ms = sum(times) / len(times) * 1000
    print(f"Latency: mean={mean_ms:.2f} ms over batch=8, CPU")

    model.save_pretrained("./tt_compressed_model")
    print("Saved compressed model to ./tt_compressed_model")


if __name__ == "__main__":
    main()
