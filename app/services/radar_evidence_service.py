from typing import Dict, Any


# ============================================================
# RADAR EVIDENCE SERVICE
# ============================================================

"""
Sentinel-1 radar evidence interpretation service.

This service interprets VV and VH backscatter measurements.

IMPORTANT:
    This is NOT a land-cover classifier.

    Radar evidence is used as supporting physical evidence
    and consistency information.

The service uses:
    VV
    VH
    VV-VH polarization separation

All thresholds are heuristic interpretation thresholds.
They should not be treated as universal scientific
land-cover boundaries.
"""


# ============================================================
# HEURISTIC THRESHOLDS
# ============================================================

# These thresholds are deliberately conservative.

VV_LOW = -15.0
VV_HIGH = -5.0

VH_LOW = -20.0
VH_HIGH = -8.0

POLARIZATION_GAP_LOW = 2.0
POLARIZATION_GAP_HIGH = 8.0


# ============================================================
# SAFE NUMBER
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return default


# ============================================================
# RADAR DATA QUALITY
# ============================================================

def _calculate_data_quality(
    vv: Dict[str, Any],
    vh: Dict[str, Any],
) -> float:
    """
    Calculate radar measurement availability/quality.

    This does NOT represent land-cover confidence.

    It only checks whether VV/VH contain usable statistics.
    """

    required_keys = [
        "mean",
        "std",
        "min",
        "max",
    ]

    values = []

    for source in (
        vv,
        vh,
    ):

        for key in required_keys:

            value = source.get(
                key,
                None,
            )

            try:

                value = float(value)

                if value != value:  # NaN

                    return 0.0

                values.append(value)

            except (
                TypeError,
                ValueError,
            ):

                return 0.0

    if not values:

        return 0.0

    return 1.0


# ============================================================
# RADAR INTERPRETATION
# ============================================================

def interpret_radar_evidence(
    radar: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Interpret Sentinel-1 VV/VH evidence.

    Parameters
    ----------
    radar:
        Expected structure:

        {
            "VV": {
                "mean": ...,
                "std": ...,
                "min": ...,
                "max": ...
            },
            "VH": {
                "mean": ...,
                "std": ...,
                "min": ...,
                "max": ...
            }
        }

    Returns
    -------
    dict
        Structured radar evidence.
    """

    if "VV" not in radar:

        raise KeyError(
            "Sentinel-1 VV radar statistics are missing."
        )

    if "VH" not in radar:

        raise KeyError(
            "Sentinel-1 VH radar statistics are missing."
        )

    vv = radar["VV"]
    vh = radar["VH"]

    # ========================================================
    # EXTRACT MEANS
    # ========================================================

    vv_mean = _safe_float(
        vv.get("mean")
    )

    vh_mean = _safe_float(
        vh.get("mean")
    )

    # ========================================================
    # POLARIZATION SEPARATION
    # ========================================================

    polarization_gap = (
        vv_mean - vh_mean
    )

    # ========================================================
    # RELATIVE CROSS-POLARIZATION
    # ========================================================
    #
    # VH - VV is represented in dB.
    #
    # This describes the relative strength of VH
    # compared with VV.
    #
    # It is NOT a classification probability.
    # ========================================================

    vh_minus_vv = (
        vh_mean - vv_mean
    )

    # ========================================================
    # VV INTERPRETATION
    # ========================================================

    if vv_mean <= VV_LOW:

        vv_signal = "low_backscatter"

    elif vv_mean >= VV_HIGH:

        vv_signal = "high_backscatter"

    else:

        vv_signal = "moderate_backscatter"

    # ========================================================
    # VH INTERPRETATION
    # ========================================================

    if vh_mean <= VH_LOW:

        vh_signal = "low_cross_polarized_backscatter"

    elif vh_mean >= VH_HIGH:

        vh_signal = "strong_cross_polarized_backscatter"

    else:

        vh_signal = "moderate_cross_polarized_backscatter"

    # ========================================================
    # POLARIZATION RELATIONSHIP
    # ========================================================

    if (
        POLARIZATION_GAP_LOW
        <= polarization_gap
        <= POLARIZATION_GAP_HIGH
    ):

        polarization_signal = (
            "moderate_vv_vh_separation"
        )

    elif polarization_gap < POLARIZATION_GAP_LOW:

        polarization_signal = (
            "low_vv_vh_separation"
        )

    else:

        polarization_signal = (
            "high_vv_vh_separation"
        )

    # ========================================================
    # OVERALL RADAR CHARACTER
    # ========================================================

    if (
        vv_signal == "moderate_backscatter"
        and vh_signal
        in {
            "moderate_cross_polarized_backscatter",
            "strong_cross_polarized_backscatter",
        }
        and polarization_signal
        == "moderate_vv_vh_separation"
    ):

        radar_character = (
            "mixed_surface_with_structural_backscatter"
        )

        interpretation = (
            "The Sentinel-1 measurements show moderate "
            "co-polarized backscatter together with a "
            "meaningful cross-polarized response. This is "
            "compatible with a heterogeneous surface containing "
            "structural scattering elements. Radar evidence "
            "supports the scene interpretation as an additional "
            "observation, but does not independently determine "
            "land-cover class."
        )

    elif vv_signal == "low_backscatter":

        radar_character = (
            "low_radar_backscatter"
        )

        interpretation = (
            "The Sentinel-1 measurements show relatively low "
            "radar backscatter. This provides useful physical "
            "context, but radar alone is insufficient to assign "
            "a land-cover class."
        )

    elif vv_signal == "high_backscatter":

        radar_character = (
            "high_radar_backscatter"
        )

        interpretation = (
            "The Sentinel-1 measurements show relatively high "
            "co-polarized radar backscatter. This indicates "
            "stronger surface or structural scattering, but "
            "does not independently determine land cover."
        )

    else:

        radar_character = (
            "moderate_radar_backscatter"
        )

        interpretation = (
            "The Sentinel-1 measurements provide moderate "
            "radar backscatter evidence. The measurements are "
            "retained as supporting physical observations."
        )

    # ========================================================
    # RADAR EVIDENCE SCORE
    # ========================================================
    #
    # This score represents how useful/consistent the radar
    # measurements are as supporting evidence.
    #
    # It is NOT land-cover probability.
    # ========================================================

    data_quality = _calculate_data_quality(
        vv,
        vh,
    )

    if data_quality == 0.0:

        radar_evidence_score = 0.0

    else:

        score = 0.50

        if (
            vv_signal
            != "low_backscatter"
        ):

            score += 0.15

        if (
            vh_signal
            in {
                "moderate_cross_polarized_backscatter",
                "strong_cross_polarized_backscatter",
            }
        ):

            score += 0.15

        if (
            polarization_signal
            == "moderate_vv_vh_separation"
        ):

            score += 0.20

        radar_evidence_score = min(
            score,
            1.0,
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "VV": {
            "mean": vv_mean,
            "std": _safe_float(
                vv.get("std")
            ),
            "min": _safe_float(
                vv.get("min")
            ),
            "max": _safe_float(
                vv.get("max")
            ),
            "signal": vv_signal,
        },

        "VH": {
            "mean": vh_mean,
            "std": _safe_float(
                vh.get("std")
            ),
            "min": _safe_float(
                vh.get("min")
            ),
            "max": _safe_float(
                vh.get("max")
            ),
            "signal": vh_signal,
        },

        "polarization": {

            "vv_vh_gap_db": round(
                polarization_gap,
                4,
            ),

            "vh_minus_vv_db": round(
                vh_minus_vv,
                4,
            ),

            "signal": polarization_signal,
        },

        "radar_character":
            radar_character,

        "radar_evidence_score":
            round(
                radar_evidence_score * 100,
                2,
            ),

        "data_quality_score":
            round(
                data_quality * 100,
                2,
            ),

        "interpretation":
            interpretation,

        "note": (
            "Radar evidence is a heuristic interpretation "
            "of Sentinel-1 backscatter measurements and should "
            "not be treated as an independent land-cover "
            "classifier."
        ),
    }