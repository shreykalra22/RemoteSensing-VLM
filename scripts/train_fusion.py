"""Reproducible, CPU-aware S1/S2 fusion training; never replaces the baseline."""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data.benge800 import BenGe800FusionDataset, build_records, split_summary
from app.evaluation.baseline import multilabel_metrics
from app.models.fusion import CrossModalFusionClassifier, FusionConfig
from configilm.extra.BENv2_utils import NEW_LABELS


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def target_matrix(dataset: BenGe800FusionDataset) -> np.ndarray:
    return np.asarray([[label in record.target_labels for label in dataset.output_labels] for record in dataset.records], dtype=np.float32)


def gradient_norm(model: torch.nn.Module) -> float:
    squared = sum(float(parameter.grad.detach().pow(2).sum()) for parameter in model.parameters() if parameter.grad is not None)
    return squared ** 0.5


def run_epoch(model, loader, loss_fn, device, optimizer=None, limit=None, log_every=0) -> dict:
    training = optimizer is not None; model.train(training)
    losses, scores, targets, gradients, batch_seconds = [], [], [], [], []
    for batch_index, batch in enumerate(loader):
        if limit is not None and batch_index >= limit: break
        started = time.perf_counter(); target = batch["target"].to(device)
        logits = model(batch["s1"].to(device), batch["s2"].to(device)); loss = loss_fn(logits, target)
        if training:
            optimizer.zero_grad(); loss.backward(); gradients.append(gradient_norm(model)); optimizer.step()
        losses.append(float(loss.detach().cpu())); scores.append(torch.sigmoid(logits).detach().cpu().numpy()); targets.append(target.detach().cpu().numpy()); batch_seconds.append(time.perf_counter() - started)
        if log_every and (batch_index + 1) % log_every == 0:
            print(json.dumps({"phase": "train" if training else "validation", "batch": batch_index + 1, "loss": losses[-1]}), flush=True)
    if not losses: raise RuntimeError("No batches were processed.")
    scores_array, targets_array = np.concatenate(scores), np.concatenate(targets)
    metrics = multilabel_metrics(scores_array, targets_array)
    metrics["predicted_positive_rate"] = (scores_array >= 0.50).mean(axis=0).tolist()
    return {"loss": sum(losses) / len(losses), "metrics": metrics, "batches": len(losses), "mean_batch_seconds": sum(batch_seconds) / len(batch_seconds), "gradient_norm_mean": float(np.mean(gradients)) if gradients else None, "gradient_norm_min": float(np.min(gradients)) if gradients else None}


def save_checkpoint(path, model, config, summary, history, arguments, train_support):
    torch.save({"model_state_dict": model.state_dict(), "fusion_config": config.__dict__, "output_labels": list(NEW_LABELS), "dataset_summary": summary, "history": history, "training_arguments": arguments, "train_positive_support": train_support, "selection_metric": "validation_macro_average_precision", "note": "S1/S2 fusion checkpoint; no text modality and no target-derived model input."}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--mapping", type=Path, default=Path("configs/label_mapping.json")); parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--seed", type=int, default=20261013); parser.add_argument("--patience", type=int, default=3); parser.add_argument("--disable-pos-weight", action="store_true"); parser.add_argument("--max-train-batches", type=int); parser.add_argument("--max-validation-batches", type=int); parser.add_argument("--log-every", type=int, default=0); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); last_path = args.output.with_stem(f"{args.output.stem}-last")
    if (args.output.exists() or last_path.exists()) and not args.overwrite: raise FileExistsError(f"Refusing to overwrite {args.output} or {last_path}; pass --overwrite explicitly.")
    set_seed(args.seed); started = time.perf_counter(); records = build_records(args.root, args.mapping); summary = split_summary(records)
    if summary["group_overlap_across_splits"]: raise RuntimeError("Refusing to train with group overlap across research splits.")
    train = BenGe800FusionDataset(records, list(NEW_LABELS), "train"); validation = BenGe800FusionDataset(records, list(NEW_LABELS), "validation")
    train_targets = target_matrix(train); positives = train_targets.sum(axis=0)
    pos_weight = None if args.disable_pos_weight else torch.tensor((len(train) - positives) / np.maximum(positives, 1), dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); config = FusionConfig(num_labels=len(NEW_LABELS)); model = CrossModalFusionClassifier(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay); loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0); validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    history, best, stalled = [], float("-inf"), 0; arguments = {key: str(value.resolve()) if key == "root" else str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter(); training = run_epoch(model, train_loader, loss_fn, device, optimizer, args.max_train_batches, args.log_every)
        with torch.no_grad(): validation_result = run_epoch(model, validation_loader, loss_fn, device, None, args.max_validation_batches, args.log_every)
        selection = validation_result["metrics"]["macro_average_precision"]
        entry = {"epoch": epoch + 1, "training": training, "validation": validation_result, "selection_metric": selection, "elapsed_seconds": time.perf_counter() - epoch_started}; history.append(entry)
        if selection is not None and selection > best:
            best, stalled = selection, 0; args.output.parent.mkdir(parents=True, exist_ok=True); save_checkpoint(args.output, model, config, summary, history, arguments, positives.astype(int).tolist())
        else: stalled += 1
        print(json.dumps({"epoch": epoch + 1, "train_loss": training["loss"], "validation_loss": validation_result["loss"], "validation_macro_ap": selection, "validation_macro_f1": validation_result["metrics"]["macro_f1"], "gradient_norm_mean": training["gradient_norm_mean"], "elapsed_seconds": entry["elapsed_seconds"]}), flush=True)
        if stalled >= args.patience: break
    save_checkpoint(last_path, model, config, summary, history, arguments, positives.astype(int).tolist())
    resource = {"device": str(device), "cuda_available": torch.cuda.is_available(), "cpu_count": os.cpu_count(), "platform": platform.platform(), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "elapsed_seconds": time.perf_counter() - started, "cuda_max_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None}
    report_path = args.output.with_suffix(".json"); report_path.write_text(json.dumps({"checkpoint": str(args.output), "last_checkpoint": str(last_path), "resource_usage": resource, "history": history, "dataset_summary": summary, "training_arguments": arguments, "train_positive_support": positives.astype(int).tolist(), "train_positive_rate": (positives / len(train)).tolist(), "pos_weight": None if pos_weight is None else pos_weight.tolist()}, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "report": str(report_path), "resource_usage": resource, "completed_epochs": len(history)}, indent=2))


if __name__ == "__main__": main()
