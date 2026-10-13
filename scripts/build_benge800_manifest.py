"""Validate ben-ge-800 pairing and write a reproducible research manifest."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data.benge800 import build_records, split_summary

parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--mapping", type=Path, default=Path("configs/label_mapping.json")); parser.add_argument("--output", type=Path, required=True)
if __name__ == "__main__":
    args = parser.parse_args(); records = build_records(args.root, args.mapping)
    payload = {"dataset_root": str(args.root.resolve()), "method": "research-only deterministic split; not an official BigEarthNet or ben-ge benchmark split", "grouping": "metadata scene_source; groups are never split", "summary": split_summary(records), "records": [{"s1_id": r.s1_id, "s2_id": r.s2_id, "group_id": r.group_id, "source_labels": r.source_labels, "target_labels": r.target_labels, "unmapped_labels": r.unmapped_labels, "split": r.split} for r in records]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8"); print(json.dumps(payload["summary"], indent=2))
