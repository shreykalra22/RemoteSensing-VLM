# Training guide

Do not train until a reviewed label crosswalk and non-overlapping split manifest exist. The fusion model is CPU-compatible for smoke tests; use `BCEWithLogitsLoss` on its raw logits. Store experimental checkpoints outside the baseline model directory and never overwrite `models/bigearthnet/model.safetensors`.

## Fusion training (research-only)

The fusion model is independent of the production baseline. It consumes only
normalized S1/S2 pixels and uses no target labels, annotations, or
target-derived text as inputs. The command refuses to overwrite its output.

```powershell
.\.venv\Scripts\python.exe scripts\train_fusion.py --root data\raw\ben-ge-800 --output results\fusion-experiment.pt --epochs 10 --batch-size 8
```
One-batch wiring test only (not a training result):

```powershell
.\.venv\Scripts\python.exe scripts\train_fusion.py --root data\raw\ben-ge-800 --output results\fusion-smoke.pt --epochs 1 --batch-size 2 --max-train-batches 1 --max-validation-batches 1
```

