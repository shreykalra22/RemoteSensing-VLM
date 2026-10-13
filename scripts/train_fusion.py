"""Bounded, reproducible S1/S2 fusion training; never replaces the baseline."""
from __future__ import annotations
import argparse, json, random, sys, time
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

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def run_epoch(model, loader, loss_fn, device, optimizer=None, limit=None):
    model.train(optimizer is not None); losses, scores, targets = [], [], []
    for batch_index, batch in enumerate(loader):
        if limit is not None and batch_index >= limit: break
        target = batch["target"].to(device); logits = model(batch["s1"].to(device), batch["s2"].to(device)); loss = loss_fn(logits, target)
        if optimizer is not None: optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu())); scores.append(torch.sigmoid(logits).detach().cpu().numpy()); targets.append(target.detach().cpu().numpy())
    if not losses: raise RuntimeError("No batches were processed.")
    return sum(losses) / len(losses), multilabel_metrics(np.concatenate(scores), np.concatenate(targets))

def save_checkpoint(path, model, config, summary, history, arguments):
    torch.save({"model_state_dict": model.state_dict(), "fusion_config": config.__dict__, "output_labels": list(NEW_LABELS), "dataset_summary": summary, "history": history, "training_arguments": arguments, "note": "S1/S2 fusion checkpoint; does not replace the pretrained BigEarthNet baseline."}, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--mapping", type=Path, default=Path("configs/label_mapping.json")); parser.add_argument("--epochs", type=int, default=3); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--seed", type=int, default=20261013); parser.add_argument("--max-train-batches", type=int); parser.add_argument("--max-validation-batches", type=int); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); last_path = args.output.with_stem(f"{args.output.stem}-last")
    if (args.output.exists() or last_path.exists()) and not args.overwrite: raise FileExistsError(f"Refusing to overwrite {args.output} or {last_path}; pass --overwrite explicitly.")
    set_seed(args.seed); started = time.perf_counter(); records = build_records(args.root, args.mapping); summary = split_summary(records)
    train = BenGe800FusionDataset(records, list(NEW_LABELS), "train"); validation = BenGe800FusionDataset(records, list(NEW_LABELS), "validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); config = FusionConfig(num_labels=len(NEW_LABELS)); model = CrossModalFusionClassifier(config).to(device); optimizer = AdamW(model.parameters(), lr=args.learning_rate); loss_fn = BCEWithLogitsLoss()
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed)); validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)
    history, best = [], float("-inf"); arguments = {k: str(v.resolve()) if k == "root" else str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter(); train_loss, train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer, args.max_train_batches)
        with torch.no_grad(): validation_loss, validation_metrics = run_epoch(model, validation_loader, loss_fn, device, None, args.max_validation_batches)
        entry = {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss, "train_metrics": train_metrics, "validation_metrics": validation_metrics, "elapsed_seconds": time.perf_counter() - epoch_start}; history.append(entry)
        if validation_metrics["macro_f1"] > best: best = validation_metrics["macro_f1"]; args.output.parent.mkdir(parents=True, exist_ok=True); save_checkpoint(args.output, model, config, summary, history, arguments)
        print(json.dumps({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss, "validation_macro_f1": validation_metrics["macro_f1"], "validation_micro_f1": validation_metrics["micro_f1"], "elapsed_seconds": entry["elapsed_seconds"]}), flush=True)
    save_checkpoint(last_path, model, config, summary, history, arguments)
    resource = {"device": str(device), "parameter_count": sum(p.numel() for p in model.parameters()), "elapsed_seconds": time.perf_counter() - started, "cuda_max_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None}
    report_path = args.output.with_suffix(".json"); report_path.write_text(json.dumps({"checkpoint": str(args.output), "last_checkpoint": str(last_path), "resource_usage": resource, "history": history, "dataset_summary": summary, "training_arguments": arguments}, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "report": str(report_path), "resource_usage": resource}, indent=2))
if __name__ == "__main__": main()
