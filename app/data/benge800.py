"""Metadata-validated ben-ge-800 records and deterministic research splits."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset
from configilm.extra.BENv2_utils import band_combi_to_mean_std
from torchvision import transforms

from app.services.preprocessing import create_model_input
from app.services.sentinel_loader import load_sentinel_bands

SPLIT_SEED = 20261013
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
_MEAN, _STD = band_combi_to_mean_std(12)
_NORMALIZE = transforms.Normalize(mean=_MEAN, std=_STD)


@dataclass(frozen=True)
class BenGeRecord:
    s1_id: str; s2_id: str; s1_dir: Path; s2_dir: Path; group_id: str
    source_labels: tuple[str, ...]; target_labels: tuple[str, ...]; unmapped_labels: tuple[str, ...]; split: str


def canonical_root(root: str | Path) -> Path:
    root = Path(root).resolve()
    required = (root / "sentinel-1", root / "sentinel-2", root / "ben-ge-800_meta.csv")
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"Expected the outer ben-ge-800 archive root; got {root}")
    return root


def _mapping(mapping_path: str | Path) -> tuple[dict[str, str], set[str]]:
    payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    if payload.get("status") != "reviewed":
        raise ValueError("Taxonomy mapping is not reviewed; refusing to construct targets.")
    return payload["mapping"], set(payload.get("explicitly_unmapped", []))


def _split_for_group(group_id: str, seed: int) -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{group_id}".encode()).digest()[:8], "big") / 2**64
    return "train" if value < .70 else "validation" if value < .85 else "test"


def build_records(root: str | Path, mapping_path: str | Path, seed: int = SPLIT_SEED) -> list[BenGeRecord]:
    """Discover unique pairs, proving S1/S2 correspondence from JSON and CSV metadata.

    `scene_source` groups S1 patches from one SAR scene, which prevents those
    related patches crossing this research-only split. It is not an official split.
    """
    root = canonical_root(root); mapping, explicitly_unmapped = _mapping(mapping_path)
    with (root / "ben-ge-800_meta.csv").open(encoding="utf-8", newline="") as handle:
        csv_index = {row["patch_id_s1"]: row for row in csv.DictReader(handle)}
    records, seen_s2 = [], set()
    for s1_dir in sorted((root / "sentinel-1").iterdir()):
        metadata_path = s1_dir / f"{s1_dir.name}_labels_metadata.json"
        if not s1_dir.is_dir() or not metadata_path.is_file(): continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")); s2_id = metadata.get("corresponding_s2_patch")
        s2_dir = root / "sentinel-2" / str(s2_id)
        if not isinstance(s2_id, str) or not s2_dir.is_dir(): raise ValueError(f"Invalid JSON S1/S2 pair for {s1_dir.name}")
        if s1_dir.name not in csv_index or csv_index[s1_dir.name]["patch_id"] != s2_id: raise ValueError(f"CSV/JSON pair disagreement for {s1_dir.name}")
        if s2_id in seen_s2: raise ValueError(f"Duplicate S2 pair discovered: {s2_id}")
        seen_s2.add(s2_id); labels = tuple(metadata.get("labels", []))
        unsupported = [x for x in labels if x not in mapping and x not in explicitly_unmapped]
        if unsupported: raise ValueError(f"Unsupported source labels for {s1_dir.name}: {unsupported}")
        group_id = metadata.get("scene_source")
        if not isinstance(group_id, str) or not group_id: raise ValueError(f"Missing scene_source for {s1_dir.name}")
        records.append(BenGeRecord(s1_dir.name, s2_id, s1_dir, s2_dir, group_id, labels, tuple(sorted({mapping[x] for x in labels if x in mapping})), tuple(sorted(x for x in labels if x not in mapping)), _split_for_group(group_id, seed)))
    if not records: raise ValueError(f"No S1 metadata records found under {root}")
    return records


def split_summary(records: Iterable[BenGeRecord]) -> dict:
    records = list(records)
    return {"seed": SPLIT_SEED, "fractions": SPLIT_FRACTIONS, "records": len(records), "groups": len({r.group_id for r in records}), "by_split": {s: sum(r.split == s for r in records) for s in SPLIT_FRACTIONS}, "records_with_unmapped_source_labels": sum(bool(r.unmapped_labels) for r in records), "records_with_no_mapped_target": sum(not r.target_labels for r in records)}


class BenGe800FusionDataset(Dataset):
    """S1/S2 tensors and verified targets; it never supplies target-derived text."""
    def __init__(self, records: Iterable[BenGeRecord], output_labels: list[str], split: str):
        self.records = [r for r in records if r.split == split and r.target_labels]; self.output_labels = list(output_labels); self.label_index = {v: i for i, v in enumerate(output_labels)}
        unknown = {x for r in self.records for x in r.target_labels} - self.label_index.keys()
        if unknown: raise ValueError(f"Mapped targets absent from output labels: {sorted(unknown)}")
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        r = self.records[index]; tensor = _NORMALIZE(create_model_input(load_sentinel_bands(r.s1_dir, r.s2_dir))); target = torch.zeros(len(self.output_labels), dtype=torch.float32)
        for label in r.target_labels: target[self.label_index[label]] = 1.0
        return {"s1": tensor[:2], "s2": tensor[2:], "target": target, "s1_id": r.s1_id, "s2_id": r.s2_id}
