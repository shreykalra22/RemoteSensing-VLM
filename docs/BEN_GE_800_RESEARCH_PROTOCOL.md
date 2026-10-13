# ben-ge-800 research protocol

Canonical local root: `data/raw/ben-ge-800` (the outer directory containing
`README`, `ben-ge-800_meta.csv`, `sentinel-1`, and `sentinel-2`). The nested
`ben-ge-800/ben-ge-800` extraction is not scanned.

Pair validation follows each S1 metadata JSON's `corresponding_s2_patch` and
then requires the same S1/S2 relationship in `ben-ge-800_meta.csv`. Duplicate
S2 partners are rejected. Source labels are mapped only by
`configs/label_mapping.json`, whose source is Table 1 of the
[BigEarthNet v2.0 dataset description](https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf).
The table's *Unlabeled* source classes remain explicit unmapped annotations.

There is no official ben-ge-800 benchmark split in the local archive. The
manifest therefore uses a research-only deterministic 70/15/15 split, seed
`20261013`, by SHA-256 of `scene_source`. All patches from the same S1 scene
are assigned together. It does not prove spatial independence across different
scenes and must not be called an official split.

Build the ignored local manifest:

```powershell
.\.venv\Scripts\python.exe scripts\build_benge800_manifest.py --root data\raw\ben-ge-800 --output data\manifests\ben-ge-800-research-split.json
```

Smoke-test one record, then run the complete fixed test split:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_benge800_baseline.py --root data\raw\ben-ge-800 --output results\baseline-smoke.json --limit 1
.\.venv\Scripts\python.exe scripts\evaluate_benge800_baseline.py --root data\raw\ben-ge-800 --output results\baseline-test.json
```

The adapter uses the baseline's existing 12-channel order, nearest-neighbour
120x120 resize, and normalization only where the pretrained baseline does.
The fusion adapter preserves that same channel order, resize, and 12-channel
normalization before splitting S1 `[VV,VH]` and S2 `[B02...B12]` tensors. It
does not pass target labels or descriptions as inputs.
