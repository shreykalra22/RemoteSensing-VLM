from pathlib import Path

import torch
import torch.nn.functional as F

from app.services.sentinel_loader import load_sentinel_bands


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 120

# Exact channel order used by our 12-channel pipeline.
CHANNEL_ORDER = [
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


# ============================================================
# RESIZE ONE BAND
# ============================================================

def resize_band(
    image: torch.Tensor,
    size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """
    Resize one satellite band to 120 x 120.

    The official repository's example preprocessing uses
    nearest-neighbour interpolation when constructing the
    model input.
    """

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D band, got shape {tuple(image.shape)}"
        )

    image = image.unsqueeze(0).unsqueeze(0)

    image = F.interpolate(
        image,
        size=(size, size),
        mode="nearest",
    )

    return image.squeeze(0).squeeze(0)


# ============================================================
# CREATE 12-CHANNEL TENSOR
# ============================================================

def create_model_input(
    bands: dict,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """
    Convert loaded Sentinel bands into:

        [12, 120, 120]

    tensor.
    """

    processed = []

    for band_name in CHANNEL_ORDER:

        if band_name not in bands:
            raise ValueError(
                f"Missing required band: {band_name}"
            )

        image = torch.from_numpy(
            bands[band_name]
        ).float()

        image = resize_band(
            image,
            size=image_size,
        )

        processed.append(image)

    tensor = torch.stack(processed, dim=0)

    expected_shape = (
        len(CHANNEL_ORDER),
        image_size,
        image_size,
    )

    if tensor.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected model input shape: "
            f"{tuple(tensor.shape)}. "
            f"Expected {expected_shape}."
        )

    return tensor


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def load_model_input(
    s1_dir: str | Path,
    s2_dir: str | Path,
) -> torch.Tensor:
    """
    Load Sentinel-1/Sentinel-2 data and convert it into
    the 12-channel BigEarthNet model input.
    """

    bands = load_sentinel_bands(
        s1_dir=s1_dir,
        s2_dir=s2_dir,
    )

    tensor = create_model_input(bands)

    return tensor


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    S1_DIR = Path(
        "reben-training-scripts/scripts/data/S1"
    )

    S2_DIR = Path(
        "reben-training-scripts/scripts/data/S2"
    )

    print("\nLoading satellite data...")

    tensor = load_model_input(
        s1_dir=S1_DIR,
        s2_dir=S2_DIR,
    )

    print("\n========== MODEL INPUT ==========")

    print("Shape:", tensor.shape)
    print("Dtype:", tensor.dtype)

    print("\nChannel statistics:")

    for index, band_name in enumerate(CHANNEL_ORDER):

        channel = tensor[index]

        print(
            f"{index:02d} | "
            f"{band_name:>3} | "
            f"shape={tuple(channel.shape)} | "
            f"min={channel.min().item():.4f} | "
            f"max={channel.max().item():.4f} | "
            f"mean={channel.mean().item():.4f}"
        )
