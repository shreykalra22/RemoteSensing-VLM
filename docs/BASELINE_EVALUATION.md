# Baseline evaluation

The pretrained baseline was evaluated on the local ben-ge-800 research-only
test split defined in `docs/BEN_GE_800_RESEARCH_PROTOCOL.md`: 100 evaluated
records, zero skips, threshold 0.50, macro F1 0.7561, micro F1 0.8034, and
macro average precision 0.8354. These are not official BigEarthNet benchmark
results and must not be compared directly with an official split.

`app.evaluation.baseline` supplies thresholded metric calculation and refuses a taxonomy mapping unless `configs/label_mapping.json` is explicitly marked `reviewed`. Before evaluation, construct a manifest from each S1 metadata JSON and its `corresponding_s2_patch`, use an official split, and record skipped, missing, and unmapped examples. Do not use source labels directly as 19-class targets without the reviewed crosswalk.
