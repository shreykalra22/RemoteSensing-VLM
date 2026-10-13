# Baseline evaluation

No project-owned baseline metrics have been computed yet. The checkpoint card's published metrics are not application results.

`app.evaluation.baseline` supplies thresholded metric calculation and refuses a taxonomy mapping unless `configs/label_mapping.json` is explicitly marked `reviewed`. Before evaluation, construct a manifest from each S1 metadata JSON and its `corresponding_s2_patch`, use an official split, and record skipped, missing, and unmapped examples. Do not use source labels directly as 19-class targets without the reviewed crosswalk.
