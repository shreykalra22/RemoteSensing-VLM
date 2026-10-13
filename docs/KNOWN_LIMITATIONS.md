# Known limitations

- The production model is a pretrained 12-channel ResNet-50 baseline, not the experimental fusion Transformer.
- The experimental fusion model has not been trained or evaluated.
- No reviewed source-label to 19-class mapping or official local split is configured, so project evaluation metrics are intentionally unavailable.
- Rule-based explanations and evidence scores are not VLM outputs, calibrated probabilities, or correctness measures.
- The upload contract supports one matched BigEarthNet-compatible S1/S2 patch; arbitrary scene mosaics are unsupported.
