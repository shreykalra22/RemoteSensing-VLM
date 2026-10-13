import sys
from pathlib import Path

import torch


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

REBEN_DIR = BASE_DIR / "reben-training-scripts"
MODEL_DIR = BASE_DIR / "models" / "bigearthnet"

sys.path.insert(0, str(REBEN_DIR))


# ============================================================
# IMPORT OFFICIAL MODEL
# ============================================================

from reben_publication.BigEarthNetv2_0_ImageClassifier import (
    BigEarthNetv2_0_ImageClassifier,
)

from configilm.extra.BENv2_utils import NEW_LABELS


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading BigEarthNet model...")

model = BigEarthNetv2_0_ImageClassifier.from_pretrained(
    str(MODEL_DIR)
)

model = model.to(DEVICE)
model.eval()

print(f"BigEarthNet model loaded on: {DEVICE}")


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(tensor: torch.Tensor):
    """
    Run BigEarthNet inference.

    Expected input:
        [batch, 12, 120, 120]

    Returns:
        List of predictions for each image.
    """

    tensor = tensor.to(DEVICE)

    logits = model(tensor)

    probabilities = torch.sigmoid(logits)

    results = []

    for probs in probabilities:

        predictions = []

        for index, probability in enumerate(probs):

            predictions.append(
                {
                    "label": NEW_LABELS[index],
                    "confidence": float(probability.item()),
                }
            )

        predictions.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        results.append(predictions)

    return results