import torch

from configilm.extra.BENv2_utils import (
    NEW_LABELS,
    band_combi_to_mean_std,
)
from torchvision import transforms

from app.services.bigearthnet_service import model, DEVICE
from app.services.preprocessing import load_model_input

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

S1_DIR = BASE_DIR / "reben-training-scripts" / "scripts" / "data" / "S1"
S2_DIR = BASE_DIR / "reben-training-scripts" / "scripts" / "data" / "S2"


# ============================================================
# NORMALIZATION
# ============================================================

MEAN, STD = band_combi_to_mean_std(12)

NORMALIZE = transforms.Normalize(
    mean=MEAN,
    std=STD,
)


# ============================================================
# LOAD
# ============================================================

image = load_model_input(
    s1_dir=S1_DIR,
    s2_dir=S2_DIR,
)

print("\n========== RAW INPUT ==========")

for i in range(12):
    print(
        f"{i:02d} | "
        f"mean={image[i].mean():10.4f} | "
        f"std={image[i].std():10.4f} | "
        f"min={image[i].min():10.4f} | "
        f"max={image[i].max():10.4f}"
    )


# ============================================================
# NORMALIZE
# ============================================================

image = NORMALIZE(image)

print("\n========== NORMALIZED INPUT ==========")

for i in range(12):
    print(
        f"{i:02d} | "
        f"mean={image[i].mean():10.4f} | "
        f"std={image[i].std():10.4f} | "
        f"min={image[i].min():10.4f} | "
        f"max={image[i].max():10.4f}"
    )


# ============================================================
# MODEL
# ============================================================

image = image.unsqueeze(0).to(DEVICE)

model.eval()

with torch.no_grad():
    logits = model(image)

probabilities = torch.sigmoid(logits)[0]


# ============================================================
# RAW LOGITS
# ============================================================

print("\n========== RAW LOGITS ==========")

for i, logit in enumerate(logits[0]):
    print(
        f"{i:02d} | "
        f"{NEW_LABELS[i]:<70} | "
        f"logit={logit.item():12.6f} | "
        f"prob={probabilities[i].item():.10f}"
    )


# ============================================================
# TOP PREDICTIONS
# ============================================================

print("\n========== TOP PREDICTIONS ==========")

indices = torch.argsort(
    probabilities,
    descending=True,
)

for rank, index in enumerate(indices, start=1):

    index = index.item()

    print(
        f"{rank:02d}. "
        f"{NEW_LABELS[index]:<70} "
        f"{probabilities[index].item() * 100:10.6f}%"
    )
