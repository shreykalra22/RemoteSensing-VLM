from pathlib import Path

import torch

from configilm.extra.BENv2_utils import (
    NEW_LABELS,
    band_combi_to_mean_std,
)
from torchvision import transforms

from app.services.bigearthnet_service import model, DEVICE
from app.services.preprocessing import load_model_input


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

S1_DIR = BASE_DIR / "reben-training-scripts" / "scripts" / "data" / "S1"
S2_DIR = BASE_DIR / "reben-training-scripts" / "scripts" / "data" / "S2"


# ============================================================
# OFFICIAL BIGEARTHNET NORMALIZATION
# ============================================================

MEAN, STD = band_combi_to_mean_std(12)

NORMALIZE = transforms.Normalize(
    mean=MEAN,
    std=STD,
)


# ============================================================
# INFERENCE
# ============================================================

def predict():

    print("\nLoading and preprocessing satellite data...")

    # --------------------------------------------------------
    # Load Sentinel-1 + Sentinel-2
    #
    # Output:
    # [12, 120, 120]
    # --------------------------------------------------------

    image = load_model_input(
        s1_dir=S1_DIR,
        s2_dir=S2_DIR,
    )

    print("\nBefore normalization:")
    print("Shape:", image.shape)
    print("Dtype:", image.dtype)

    # --------------------------------------------------------
    # Official BigEarthNet normalization
    # --------------------------------------------------------

    image = NORMALIZE(image)

    print("\nAfter normalization:")
    print("Shape:", image.shape)
    print("Dtype:", image.dtype)

    # --------------------------------------------------------
    # Add batch dimension
    #
    # [12,120,120]
    #       ↓
    # [1,12,120,120]
    # --------------------------------------------------------

    image = image.unsqueeze(0)

    print("\nModel input:")
    print("Shape:", image.shape)

    # --------------------------------------------------------
    # Move to model device
    # --------------------------------------------------------

    image = image.to(DEVICE)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():
        logits = model(image)

    print("\nRaw model output:")
    print("Shape:", logits.shape)

    # --------------------------------------------------------
    # Multi-label classification
    #
    # Use sigmoid, NOT softmax.
    # --------------------------------------------------------

    probabilities = torch.sigmoid(logits)[0]

    # --------------------------------------------------------
    # Sort predictions
    # --------------------------------------------------------

    sorted_indices = torch.argsort(
        probabilities,
        descending=True,
    )

    predictions = []

    for index in sorted_indices:

        index = index.item()

        predictions.append(
            {
                "label": NEW_LABELS[index],
                "confidence": float(
                    probabilities[index].item()
                ),
            }
        )

    return predictions


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    predictions = predict()

    print("\n" + "=" * 70)
    print("BIGEARTHNET PREDICTIONS")
    print("=" * 70)

    for rank, prediction in enumerate(
        predictions,
        start=1,
    ):

        print(
            f"{rank:2}. "
            f"{prediction['label']:<70} "
            f"{prediction['confidence'] * 100:6.2f}%"
        )