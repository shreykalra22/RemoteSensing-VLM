"""Opt-in training for the new S1/S2 fusion model; never replaces the baseline."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data.benge800 import BenGe800FusionDataset, build_records, split_summary
from app.models.fusion import CrossModalFusionClassifier, FusionConfig
from configilm.extra.BENv2_utils import NEW_LABELS

def run_epoch(model, loader, loss_fn, device, optimizer=None, limit=None):
    model.train(optimizer is not None); losses = []
    for batch_index, batch in enumerate(loader):
        if limit is not None and batch_index >= limit: break
        loss = loss_fn(model(batch["s1"].to(device), batch["s2"].to(device)), batch["target"].to(device))
        if optimizer is not None: optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses: raise RuntimeError("No batches were processed.")
    return sum(losses) / len(losses)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--mapping", type=Path, default=Path("configs/label_mapping.json")); parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--max-train-batches", type=int); parser.add_argument("--max-validation-batches", type=int); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite: raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite explicitly.")
    records = build_records(args.root, args.mapping); train = BenGe800FusionDataset(records, list(NEW_LABELS), "train"); validation = BenGe800FusionDataset(records, list(NEW_LABELS), "validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = CrossModalFusionClassifier(FusionConfig(num_labels=len(NEW_LABELS))).to(device); optimizer = AdamW(model.parameters(), lr=args.learning_rate); loss_fn = BCEWithLogitsLoss(); history = []
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, DataLoader(train, batch_size=args.batch_size, shuffle=True), loss_fn, device, optimizer, args.max_train_batches)
        with torch.no_grad(): val_loss = run_epoch(model, DataLoader(validation, batch_size=args.batch_size), loss_fn, device, None, args.max_validation_batches)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss}); print(history[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "fusion_config": FusionConfig().__dict__, "output_labels": list(NEW_LABELS), "dataset_summary": split_summary(records), "history": history, "note": "New fusion checkpoint; does not replace the pretrained BigEarthNet baseline."}, args.output)
    print(json.dumps({"output": str(args.output), "device": str(device), "history": history}, indent=2))
if __name__ == "__main__": main()
