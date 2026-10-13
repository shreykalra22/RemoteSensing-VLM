from pathlib import Path
from typing import Dict, Any

import torch

from app.services.preprocessing import load_model_input


# ============================================================
# SCENE EVIDENCE SERVICE
# ============================================================

def extract_scene_evidence(
    s1_dir: Path,
    s2_dir: Path,
) -> Dict[str, Any]:
    """
    Extract numerical remote-sensing evidence from
    Sentinel-1 and Sentinel-2 imagery.

    Current version:
        - Loads the official 12-channel model input
        - Computes per-channel statistics
        - Separates optical and SAR channels

    Returns structured evidence for downstream
    scene reasoning / VLM explanation.
    """

    # --------------------------------------------------------
    # Load the same 12-channel input used by BigEarthNet
    # --------------------------------------------------------

    image = load_model_input(
        s1_dir=s1_dir,
        s2_dir=s2_dir,
    )

    if image.ndim != 3:
        raise ValueError(
            f"Expected [channels, height, width], got {image.shape}"
        )

    # --------------------------------------------------------
    # Channel definitions
    # --------------------------------------------------------

    channel_order = [
        "VV",
        "VH",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
    ]

    # --------------------------------------------------------
    # Channel statistics
    # --------------------------------------------------------

    channel_statistics = {}

    for index, channel in enumerate(channel_order):

        channel_data = image[index].float()

        channel_statistics[channel] = {
            "mean": float(channel_data.mean().item()),
            "std": float(channel_data.std().item()),
            "min": float(channel_data.min().item()),
            "max": float(channel_data.max().item()),
        }

    # --------------------------------------------------------
    # Separate SAR and optical channels
    # --------------------------------------------------------

    sar_channels = ["VV", "VH"]

    optical_channels = [
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
    ]

    # --------------------------------------------------------
    # Return evidence
    # --------------------------------------------------------

    return {
        "input_shape": list(image.shape),
        "channel_order": channel_order,
        "sar_channels": sar_channels,
        "optical_channels": optical_channels,
        "channel_statistics": channel_statistics,
    }