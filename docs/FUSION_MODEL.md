# Fusion model

`app.models.fusion.CrossModalFusionClassifier` is an **implemented, not yet trained** experimental architecture.

It uses separate trainable S1 (2-channel) and S2 (10-channel) encoders, a trainable metadata-token encoder, modality markers, and `torch.nn.TransformerEncoder` self-attention before a 19-logit multilabel head. It is not connected to the production inference endpoint.

Metadata text must exclude target land-cover labels. The local ben-ge metadata provides non-target fields such as timestamps, season, climate zone and coarse location; a future dataset adapter must document exactly which fields become text tokens and prove they are available at inference time.
