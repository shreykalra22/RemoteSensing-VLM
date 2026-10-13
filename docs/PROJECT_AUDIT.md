# Project audit

## Verified baseline

The repository baseline is commit `0138f65` (`chore: save working baseline`).
The application imports `app.main`, which exposes `/`, `/health`, `/predict`,
and `/predict/upload`.  The React entry point is `frontend/src/main.jsx`.

The active request path is: upload two ZIP archives -> safe temporary
extraction -> `run_scene_pipeline` -> validated Sentinel band loading ->
BigEarthNet preprocessing -> pretrained ResNet-50 inference -> spectral/radar
measurements -> rule-based evidence/explanation -> JSON -> React display.

## Model and input contract

`models/bigearthnet/model.safetensors` is the local checkpoint used through
the upstream `reben-training-scripts` implementation.  The checkpoint model
is a 12-channel ResNet-50 with 19 sigmoid outputs; it is the baseline, not a
Transformer or VLM.  Its required input order is `VV, VH, B02, B03, B04,
B05, B06, B07, B08, B8A, B11, B12`, resized to 120 x 120 and normalized with
`band_combi_to_mean_std(12)`.

## Dataset availability

The ignored local `data/` directory contains ben-ge-800 material, including
paired Sentinel folders and BigEarthNet label metadata.  A Sentinel-1 metadata
file identifies its exact S2 partner through `corresponding_s2_patch` and has
raw source labels.  The repository does not contain `DEMO_PAIRS/Demo_01` to
`Demo_05`; root-level `S1.zip` and `S2.zip` are unrelated patches and must not
be used as an evaluation pair.

The local annotations are source taxonomy labels, not automatically the
checkpoint's 19-label taxonomy.  A reviewable mapping is therefore required
before metrics can be considered valid.

## Technical debt and risks

- Historical diagnostic scripts in `app/services` are not imported by the API.
- There is no root-level Python dependency manifest for the application;
  upstream reBEN has its own Poetry configuration.
- `/predict` references legacy local sample directories, which may fail the
  intentional S1/S2 pairing validation.
- Heuristic evidence and deterministic text are supplementary analysis, not
  calibrated probability or VLM output.

## Implementation order

1. Preserve the baseline and test input/model contracts.
2. Add an explicit data manifest, taxonomy-mapping gate, and baseline evaluator.
3. Add a separately named trainable fusion model with tests; do not replace the
   ResNet baseline.
4. Add training/checkpoint interfaces only after a reviewed split and mapping
   are available.
5. Surface only real model/evaluation state in the API and frontend.
