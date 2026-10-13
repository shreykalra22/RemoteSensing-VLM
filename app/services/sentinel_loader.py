"""Validated loading of the BigEarthNet Sentinel-1/Sentinel-2 inputs."""

from pathlib import Path
from typing import Dict

import numpy as np
import rasterio

S1_BANDS = ("VV", "VH")
S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
ALL_BANDS = S2_BANDS + S1_BANDS
SUPPORTED_SUFFIXES = {".tif", ".tiff"}


def _raster_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def _matches_band(path: Path, band: str) -> bool:
    return path.stem.upper().endswith(f"_{band}")


def _scene_key(path: Path, band: str) -> str:
    suffix = f"_{band}"
    stem = path.stem
    return stem[: -len(suffix)] if stem.upper().endswith(suffix) else stem


def _patch_coordinates(scene_key: str) -> tuple[str, str] | None:
    """Extract BigEarthNet's trailing ``row_column`` patch coordinates."""
    parts = scene_key.rsplit("_", 2)
    if len(parts) == 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return parts[-2], parts[-1]
    return None


def find_band_files(s1_dir: str | Path, s2_dir: str | Path) -> Dict[str, Path]:
    """Find exactly one file per required band, rejecting ambiguous uploads."""
    s1_files = _raster_files(Path(s1_dir))
    s2_files = _raster_files(Path(s2_dir))
    band_files: Dict[str, Path] = {}

    for satellite, files, bands in (("Sentinel-1", s1_files, S1_BANDS), ("Sentinel-2", s2_files, S2_BANDS)):
        for band in bands:
            matches = [path for path in files if _matches_band(path, band)]
            if not matches:
                raise FileNotFoundError(f"Missing required {satellite} band: {band}.")
            if len(matches) != 1:
                names = ", ".join(path.name for path in matches[:3])
                raise ValueError(f"Ambiguous {satellite} band {band}: found {len(matches)} files ({names}). Upload one paired patch per archive.")
            band_files[band] = matches[0]

    s1_keys = {_scene_key(band_files[band], band) for band in S1_BANDS}
    s2_keys = {_scene_key(band_files[band], band) for band in S2_BANDS}
    if len(s1_keys) != 1 or len(s2_keys) != 1:
        raise ValueError("Each upload must contain bands from one acquisition/patch only.")

    s1_key, s2_key = next(iter(s1_keys)), next(iter(s2_keys))
    s1_coords, s2_coords = _patch_coordinates(s1_key), _patch_coordinates(s2_key)
    if s1_coords and s2_coords and s1_coords != s2_coords:
        raise ValueError(f"Sentinel-1 and Sentinel-2 patches do not match: {s1_coords} versus {s2_coords}.")
    return band_files


def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        image = source.read(1).astype(np.float32)
    if image.ndim != 2 or image.size == 0:
        raise ValueError(f"Raster is not a non-empty single band: {path.name}")
    if not np.isfinite(image).all():
        raise ValueError(f"Raster contains NaN or infinite values: {path.name}")
    return image


def load_sentinel_bands(s1_dir: str | Path, s2_dir: str | Path) -> Dict[str, np.ndarray]:
    band_files = find_band_files(s1_dir, s2_dir)
    return {band: read_band(band_files[band]) for band in ALL_BANDS}
