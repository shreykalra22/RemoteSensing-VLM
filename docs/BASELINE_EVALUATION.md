# Baseline evaluation

The pretrained baseline was evaluated on the local ben-ge-800 research-only
test split defined in `docs/BEN_GE_800_RESEARCH_PROTOCOL.md`: 161 evaluated
records, zero skips, threshold 0.50, macro F1 0.7408, micro F1 0.7725, macro
average precision 0.8330, and micro average precision 0.8585. These replace
the earlier result after the split was corrected to prevent shared-S2-scene
leakage. They are not official BigEarthNet benchmark
results and must not be compared directly with an official split.

`app.evaluation.baseline` supplies thresholded metric calculation and refuses a taxonomy mapping unless `configs/label_mapping.json` is explicitly marked `reviewed`. Before evaluation, construct a manifest from each S1 metadata JSON and its `corresponding_s2_patch`, use an official split, and record skipped, missing, and unmapped examples. Do not use source labels directly as 19-class targets without the reviewed crosswalk.
