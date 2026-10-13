"""Evaluate the untouched pretrained 19-class baseline on the research test split."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data.benge800 import build_records, split_summary
from app.evaluation.baseline import multilabel_metrics
from app.services.prediction_service import predict_satellite_scene
from configilm.extra.BENv2_utils import NEW_LABELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=Path("configs/label_mapping.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Smoke-test only: cap the test records; never report this as the full split.")
    args = parser.parse_args()
    records = [r for r in build_records(args.root, args.mapping) if r.split == "test" and r.target_labels]
    full_count = len(records)
    if args.limit is not None: records = records[:args.limit]
    label_index = {label: i for i, label in enumerate(NEW_LABELS)}
    scores, targets, skipped = [], [], []
    for number, record in enumerate(records, 1):
        try:
            result = predict_satellite_scene(record.s1_dir, record.s2_dir)
            row_scores = np.zeros(len(NEW_LABELS), dtype=np.float32)
            for prediction in result["predictions"]: row_scores[label_index[prediction["label"]]] = prediction["confidence"]
            row_targets = np.zeros(len(NEW_LABELS), dtype=bool)
            for label in record.target_labels: row_targets[label_index[label]] = True
            scores.append(row_scores); targets.append(row_targets)
        except Exception as exc:  # Preserve a trace of every excluded sample.
            skipped.append({"s1_id": record.s1_id, "s2_id": record.s2_id, "reason": str(exc)})
        print(f"{number}/{len(records)}", flush=True)
    if not scores: raise RuntimeError("No records were successfully evaluated.")
    metrics = multilabel_metrics(np.stack(scores), np.stack(targets))
    metrics["labels"] = list(NEW_LABELS)
    metrics["dataset_summary"] = split_summary(build_records(args.root, args.mapping))
    metrics["evaluation"] = {"split": "test", "eligible_test_records": full_count, "attempted": len(records), "evaluated": len(scores), "skipped": skipped, "smoke_test_limit": args.limit, "complete_test_split": args.limit is None and len(records) == full_count}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in ("macro_f1", "micro_f1", "macro_average_precision", "evaluation")}, indent=2))


if __name__ == "__main__": main()
