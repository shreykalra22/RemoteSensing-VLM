# Training guide

Do not train until a reviewed label crosswalk and non-overlapping split manifest exist. The fusion model is CPU-compatible for smoke tests; use `BCEWithLogitsLoss` on its raw logits. Store experimental checkpoints outside the baseline model directory and never overwrite `models/bigearthnet/model.safetensors`.
