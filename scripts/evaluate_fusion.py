"""Evaluate a trained S1/S2 fusion checkpoint on the fixed held-out split."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data.benge800 import BenGe800FusionDataset, build_records, split_summary
from app.evaluation.baseline import multilabel_metrics
from app.models.fusion import CrossModalFusionClassifier, FusionConfig
from configilm.extra.BENv2_utils import NEW_LABELS

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--mapping", type=Path, default=Path("configs/label_mapping.json")); parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(); started = time.perf_counter(); saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved["output_labels"] != list(NEW_LABELS): raise ValueError("Checkpoint output-label order differs from the baseline.")
    records = build_records(args.root, args.mapping); summary = split_summary(records)
    if saved["dataset_summary"] != summary: raise ValueError("Checkpoint and current dataset split summaries differ.")
    dataset = BenGe800FusionDataset(records, list(NEW_LABELS), "test"); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModalFusionClassifier(FusionConfig(**saved["fusion_config"])).to(device); model.load_state_dict(saved["model_state_dict"]); model.eval(); scores, targets = [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            scores.append(torch.sigmoid(model(batch["s1"].to(device), batch["s2"].to(device))).cpu().numpy()); targets.append(batch["target"].numpy())
    scores_array, targets_array = np.concatenate(scores), np.concatenate(targets)
    metrics = multilabel_metrics(scores_array, targets_array); metrics["predicted_positive_rate"] = (scores_array >= 0.50).mean(axis=0).tolist()
    metrics.update({"labels": list(NEW_LABELS), "dataset_summary": summary, "evaluation": {"split": "test", "evaluated": len(dataset), "batch_size": args.batch_size, "checkpoint": str(args.checkpoint), "selection_metric": saved.get("selection_metric"), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "elapsed_seconds": time.perf_counter() - started, "device": str(device)}})
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8"); print(json.dumps({k: metrics[k] for k in ("macro_f1", "micro_f1", "macro_average_precision", "micro_average_precision", "evaluation")}, indent=2))
if __name__ == "__main__": main()
