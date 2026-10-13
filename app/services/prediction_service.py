"""BigEarthNet-v2 prediction service for the stable baseline."""

from typing import Mapping

import numpy as np
import torch
from configilm.extra.BENv2_utils import NEW_LABELS, band_combi_to_mean_std
from torchvision import transforms

from app.services.bigearthnet_service import DEVICE, model
from app.services.preprocessing import create_model_input
from app.services.sentinel_loader import load_sentinel_bands

MEAN, STD = band_combi_to_mean_std(12)
NORMALIZE = transforms.Normalize(mean=MEAN, std=STD)
DEFAULT_THRESHOLD = 0.50


def predict_satellite_scene(s1_dir, s2_dir, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Load a pair and return the 19-label BigEarthNet prediction."""
    return predict_bands(load_sentinel_bands(s1_dir, s2_dir), threshold)


def predict_bands(bands: Mapping[str, np.ndarray], threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Predict from already-loaded bands, avoiding duplicate raster reads."""
    image = NORMALIZE(create_model_input(bands)).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(image))[0]

    predictions = []
    for index in torch.argsort(probabilities, descending=True).tolist():
        probability = float(probabilities[index].item())
        predictions.append({
            "label": NEW_LABELS[index],
            "confidence": probability,
            "confidence_percent": round(probability * 100, 4),
            "detected": probability >= threshold,
        })

    return {
        "model": "BigEarthNet-v2 ResNet-50",
        "task": "multi-label land-cover classification",
        "input_channels": 12,
        "input_size": [120, 120],
        "threshold": threshold,
        "top_prediction": predictions[0],
        "detected_labels": [item for item in predictions if item["detected"]],
        "predictions": predictions,
    }
