# Architecture

The production baseline accepts a matched BigEarthNet-style S1/S2 patch, loads VV/VH plus ten S2 bands, constructs the verified 12-channel tensor, applies official reBEN normalization, and invokes the pretrained ResNet-50. Spectral/radar services and deterministic rule-based explanations are downstream of model inference.

`app.models.fusion` is isolated experimental code: independent S1 and S2 encoders emit tokens, optional non-target metadata tokens are encoded separately, and a Transformer encoder fuses the tokens for a 19-logit multilabel head.
