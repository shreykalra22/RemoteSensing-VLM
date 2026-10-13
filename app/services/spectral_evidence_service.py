
from typing import Dict, Any

import numpy as np
import cv2


# ============================================================
# STATISTICS HELPER
# ============================================================

def _statistics(array: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistics for a satellite band/index.
    """

    array = np.asarray(array, dtype=np.float32)

    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


# ============================================================
# RESIZE HELPER
# ============================================================

def _match_shape(
    array: np.ndarray,
    target_shape: tuple,
) -> np.ndarray:
    """
    Resize a satellite band to match the target spatial shape.

    This is required because Sentinel-2 bands have different
    native spatial resolutions.

    Example:
        B08 -> 120 x 120
        B11 -> 60 x 60

    B11 is resized to 120 x 120 before calculating NDBI.
    """

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    if array.shape == target_shape:
        return array

    target_height, target_width = target_shape

    return cv2.resize(
        array,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )


# ============================================================
# NORMALIZED DIFFERENCE HELPER
# ============================================================

def _normalized_difference(
    band_a: np.ndarray,
    band_b: np.ndarray,
) -> np.ndarray:
    """
    Calculate normalized difference:

        (A - B) / (A + B)

    Small epsilon prevents division by zero.
    """

    # --------------------------------------------------------
    # Make both bands the same spatial resolution
    # --------------------------------------------------------

    if band_a.shape != band_b.shape:

        band_b = _match_shape(
            band_b,
            band_a.shape,
        )

    # --------------------------------------------------------
    # Calculate normalized difference
    # --------------------------------------------------------

    denominator = band_a + band_b

    return (
        (band_a - band_b)
        / (denominator + 1e-8)
    )


# ============================================================
# SPECTRAL EVIDENCE
# ============================================================

def extract_spectral_evidence(
    bands: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Extract spectral and radar evidence from Sentinel-1
    and Sentinel-2 bands.

    Expected input:

        {
            "B02": ...,
            "B03": ...,
            "B04": ...,
            "B05": ...,
            "B06": ...,
            "B07": ...,
            "B08": ...,
            "B8A": ...,
            "B11": ...,
            "B12": ...,
            "VV": ...,
            "VH": ...
        }
    """

    # ========================================================
    # VALIDATE REQUIRED BANDS
    # ========================================================

    required_bands = [
        "B04",
        "B08",
        "B03",
        "B11",
        "VV",
        "VH",
    ]

    for band in required_bands:

        if band not in bands:

            raise KeyError(
                f"Required band '{band}' is missing."
            )

    # ========================================================
    # SENTINEL-2 BANDS
    # ========================================================

    B03 = np.asarray(
        bands["B03"],
        dtype=np.float32,
    )

    B04 = np.asarray(
        bands["B04"],
        dtype=np.float32,
    )

    B08 = np.asarray(
        bands["B08"],
        dtype=np.float32,
    )

    B11 = np.asarray(
        bands["B11"],
        dtype=np.float32,
    )

    # ========================================================
    # SENTINEL-1 BANDS
    # ========================================================

    VV = np.asarray(
        bands["VV"],
        dtype=np.float32,
    )

    VH = np.asarray(
        bands["VH"],
        dtype=np.float32,
    )

    # ========================================================
    # MATCH SENTINEL-2 RESOLUTIONS
    # ========================================================
    #
    # B03, B04 and B08 are normally 10 m resolution.
    #
    # B11 is normally 20 m resolution and therefore may be
    # 60 x 60 while B08 is 120 x 120.
    #
    # Resize B11 to B08 resolution before calculating
    # NDBI.
    # ========================================================

    B03 = _match_shape(
        B03,
        B08.shape,
    )

    B04 = _match_shape(
        B04,
        B08.shape,
    )

    B11 = _match_shape(
        B11,
        B08.shape,
    )

    # ========================================================
    # SPECTRAL INDICES
    # ========================================================

    # --------------------------------------------------------
    # NDVI
    #
    # Vegetation:
    #
    # (NIR - Red)
    # ------------
    # (NIR + Red)
    # --------------------------------------------------------

    NDVI = _normalized_difference(
        B08,
        B04,
    )

    # --------------------------------------------------------
    # NDWI
    #
    # Water:
    #
    # (Green - NIR)
    # -------------
    # (Green + NIR)
    # --------------------------------------------------------

    NDWI = _normalized_difference(
        B03,
        B08,
    )

    # --------------------------------------------------------
    # NDBI
    #
    # Built-up:
    #
    # (SWIR - NIR)
    # ------------
    # (SWIR + NIR)
    # --------------------------------------------------------

    NDBI = _normalized_difference(
        B11,
        B08,
    )

    # NDMI is intentionally not emitted. With this pair of bands it is
    # exactly the negative of NDBI, so presenting both as independent
    # evidence would double-count the same measurement.

    # ========================================================
    # RETURN STRUCTURED EVIDENCE
    # ========================================================

    return {

        "indices": {

            "NDVI": _statistics(
                NDVI
            ),

            "NDWI": _statistics(
                NDWI
            ),

            "NDBI": _statistics(
                NDBI
            ),

        },

        "radar": {

            "VV": _statistics(
                VV
            ),

            "VH": _statistics(
                VH
            ),
        },
    }
