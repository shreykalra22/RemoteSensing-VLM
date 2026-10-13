from typing import Dict, Any
from app.services.radar_evidence_service import (
    interpret_radar_evidence,
)

# ============================================================
# HEURISTIC THRESHOLDS
# ============================================================
#
# These are interpretation thresholds, NOT ML thresholds.
#
# They are used only to describe whether the calculated
# spectral indices provide supporting or opposing evidence.
#
# They should not be treated as scientifically absolute
# land-cover classification boundaries.
# ============================================================

NDVI_STRONG_VEGETATION = 0.40
NDVI_MODERATE_VEGETATION = 0.20

NDWI_WATER_SIGNAL = 0.10

NDBI_BUILT_UP_SIGNAL = 0.10


# ============================================================
# SUPPORT LEVEL
# ============================================================

def _support_level(
    value: float,
    positive_threshold: float,
    moderate_threshold: float = 0.0,
) -> str:
    """
    Convert an index value into a qualitative support level.

    Returns:
        strong
        moderate
        weak
        not_supported
    """

    if value >= positive_threshold:
        return "strong"

    if value >= moderate_threshold:
        return "moderate"

    if value > -moderate_threshold:
        return "weak"

    return "not_supported"


# ============================================================
# VEGETATION INTERPRETATION
# ============================================================

def _interpret_ndvi(
    ndvi_mean: float,
) -> Dict[str, Any]:
    """
    Interpret mean NDVI as vegetation evidence.
    """

    if ndvi_mean >= NDVI_STRONG_VEGETATION:

        return {
            "value": ndvi_mean,
            "signal": "vegetation",
            "support_level": "strong",
            "interpretation": (
                "The mean NDVI indicates a strong vegetation signal."
            ),
        }

    if ndvi_mean >= NDVI_MODERATE_VEGETATION:

        return {
            "value": ndvi_mean,
            "signal": "vegetation",
            "support_level": "moderate",
            "interpretation": (
                "The mean NDVI indicates a moderate vegetation signal."
            ),
        }

    if ndvi_mean >= 0:

        return {
            "value": ndvi_mean,
            "signal": "vegetation",
            "support_level": "weak",
            "interpretation": (
                "The mean NDVI indicates a weak vegetation signal."
            ),
        }

    return {
        "value": ndvi_mean,
        "signal": "low_vegetation",
        "support_level": "not_supported",
        "interpretation": (
            "The mean NDVI does not provide strong evidence "
            "of vegetation dominance."
        ),
    }


# ============================================================
# WATER INTERPRETATION
# ============================================================

def _interpret_ndwi(
    ndwi_mean: float,
) -> Dict[str, Any]:
    """
    Interpret mean NDWI as water evidence.
    """

    if ndwi_mean >= NDWI_WATER_SIGNAL:

        return {
            "value": ndwi_mean,
            "signal": "water",
            "support_level": "strong",
            "interpretation": (
                "The mean NDWI provides a positive water signal."
            ),
        }

    if ndwi_mean >= 0:

        return {
            "value": ndwi_mean,
            "signal": "possible_water",
            "support_level": "weak",
            "interpretation": (
                "The mean NDWI is near neutral and provides "
                "limited evidence of water."
            ),
        }

    return {
        "value": ndwi_mean,
        "signal": "low_water_signal",
        "support_level": "not_supported",
        "interpretation": (
            "The negative mean NDWI does not support water "
            "as the dominant scene characteristic."
        ),
    }


# ============================================================
# BUILT-UP INTERPRETATION
# ============================================================

def _interpret_ndbi(
    ndbi_mean: float,
) -> Dict[str, Any]:
    """
    Interpret mean NDBI as built-up evidence.
    """

    if ndbi_mean >= NDBI_BUILT_UP_SIGNAL:

        return {
            "value": ndbi_mean,
            "signal": "built_up",
            "support_level": "strong",
            "interpretation": (
                "The mean NDBI provides a positive built-up signal."
            ),
        }

    if ndbi_mean >= 0:

        return {
            "value": ndbi_mean,
            "signal": "possible_built_up",
            "support_level": "weak",
            "interpretation": (
                "The mean NDBI is near neutral and provides "
                "limited built-up evidence."
            ),
        }

    return {
        "value": ndbi_mean,
        "signal": "low_built_up_signal",
        "support_level": "not_supported",
        "interpretation": (
            "The mean NDBI does not provide strong evidence "
            "of built-up land."
        ),
    }


# ============================================================
# CATEGORY EVIDENCE
# ============================================================

def _build_category_evidence(
    scene_analysis: Dict[str, Any],
    ndvi: Dict[str, Any],
    ndwi: Dict[str, Any],
    ndbi: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combine BigEarthNet category predictions with spectral
    evidence.

    This does NOT replace the model prediction.

    It determines whether the physical measurements provide
    supporting, neutral, or conflicting evidence.
    """

    categories = scene_analysis.get(
        "categories",
        {},
    )

    # --------------------------------------------------------
    # Agriculture
    # --------------------------------------------------------

    agriculture_detected = categories.get(
        "agriculture",
        {},
    ).get("detected", False)

    if agriculture_detected:

        if ndvi["support_level"] in {
            "strong",
            "moderate",
        }:

            agriculture_support = "supported"

            agriculture_reason = (
                "The model detects agricultural land and the "
                "NDVI provides supporting vegetation evidence."
            )

        else:

            agriculture_support = "weak_support"

            agriculture_reason = (
                "The model detects agricultural land, but the "
                "mean NDVI provides limited vegetation support."
            )

    else:

        agriculture_support = "not_detected"

        agriculture_reason = (
            "Agricultural land was not detected by BigEarthNet."
        )

    # --------------------------------------------------------
    # Forest
    # --------------------------------------------------------

    forest_detected = categories.get(
        "forest",
        {},
    ).get("detected", False)

    if forest_detected:

        if ndvi["support_level"] in {
            "strong",
            "moderate",
        }:

            forest_support = "supported"

            forest_reason = (
                "The model detects forest and the NDVI provides "
                "supporting vegetation evidence."
            )

        else:

            forest_support = "weak_support"

            forest_reason = (
                "The model detects forest, but the available "
                "mean NDVI provides limited vegetation support."
            )

    else:

        forest_support = "not_detected"

        forest_reason = (
            "Forest was not detected by BigEarthNet."
        )

    # --------------------------------------------------------
    # Woodland / shrub
    # --------------------------------------------------------

    woodland_detected = categories.get(
        "woodland_and_shrub",
        {},
    ).get("detected", False)

    if woodland_detected:

        if ndvi["support_level"] in {
            "strong",
            "moderate",
        }:

            woodland_support = "supported"

            woodland_reason = (
                "The model detects woodland/shrub and the "
                "NDVI provides supporting vegetation evidence."
            )

        else:

            woodland_support = "weak_support"

            woodland_reason = (
                "The model detects woodland/shrub, but the "
                "vegetation signal is limited."
            )

    else:

        woodland_support = "not_detected"

        woodland_reason = (
            "Woodland/shrub was not detected by BigEarthNet."
        )

    # --------------------------------------------------------
    # Water
    # --------------------------------------------------------

    water_detected = categories.get(
        "water",
        {},
    ).get("detected", False)

    if water_detected:

        if ndwi["support_level"] in {
            "strong",
            "moderate",
        }:

            water_support = "supported"

            water_reason = (
                "The model detects water and NDWI provides "
                "supporting spectral evidence."
            )

        else:

            water_support = "weak_support"

            water_reason = (
                "The model detects water, but the mean NDWI "
                "does not provide strong supporting evidence."
            )

    else:

        if ndwi["support_level"] == "not_supported":

            water_support = "consistent_absence"

            water_reason = (
                "Water was not detected by the model and the "
                "negative NDWI provides consistent evidence "
                "against water dominance."
            )

        else:

            water_support = "not_detected"

            water_reason = (
                "Water was not detected by BigEarthNet."
            )

    # --------------------------------------------------------
    # Urban
    # --------------------------------------------------------

    urban_detected = categories.get(
        "urban",
        {},
    ).get("detected", False)

    if urban_detected:

        if ndbi["support_level"] in {
            "strong",
            "moderate",
        }:

            urban_support = "supported"

            urban_reason = (
                "The model detects urban/built-up land and "
                "NDBI provides supporting built-up evidence."
            )

        else:

            urban_support = "weak_support"

            urban_reason = (
                "The model detects urban/built-up land, but "
                "the mean NDBI provides limited support."
            )

    else:

        if ndbi["support_level"] == "not_supported":

            urban_support = "consistent_absence"

            urban_reason = (
                "Urban land was not detected by the model and "
                "the NDBI does not provide strong built-up evidence."
            )

        else:

            urban_support = "not_detected"

            urban_reason = (
                "Urban/built-up land was not detected by BigEarthNet."
            )

    # --------------------------------------------------------
    # Return category evidence
    # --------------------------------------------------------

    return {
        "agriculture": {
            "model_detected": agriculture_detected,
            "evidence_status": agriculture_support,
            "reason": agriculture_reason,
        },

        "forest": {
            "model_detected": forest_detected,
            "evidence_status": forest_support,
            "reason": forest_reason,
        },

        "woodland_and_shrub": {
            "model_detected": woodland_detected,
            "evidence_status": woodland_support,
            "reason": woodland_reason,
        },

        "water": {
            "model_detected": water_detected,
            "evidence_status": water_support,
            "reason": water_reason,
        },

        "urban": {
            "model_detected": urban_detected,
            "evidence_status": urban_support,
            "reason": urban_reason,
        },
    }


# ============================================================
# MAIN FUNCTION
# ============================================================

def interpret_scene_evidence(
    prediction: Dict[str, Any],
    scene_analysis: Dict[str, Any],
    spectral_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combine model predictions, scene analysis and spectral
    evidence into a structured evidence interpretation.

    Parameters
    ----------
    prediction:
        Output from predict_satellite_scene().

    scene_analysis:
        Output from analyze_scene().

    spectral_evidence:
        Output from extract_spectral_evidence().

    Returns
    -------
    dict
        Structured evidence interpretation.
    """

    # ========================================================
    # Extract spectral statistics
    # ========================================================

    indices = spectral_evidence["indices"]

    ndvi = _interpret_ndvi(
        indices["NDVI"]["mean"]
    )

    ndwi = _interpret_ndwi(
        indices["NDWI"]["mean"]
    )

    ndbi = _interpret_ndbi(
        indices["NDBI"]["mean"]
    )

    # ========================================================
    # Radar
    # ========================================================

    radar = interpret_radar_evidence(
    spectral_evidence["radar"]
)

    # ========================================================
    # Category-level interpretation
    # ========================================================

    categories = _build_category_evidence(
        scene_analysis=scene_analysis,
        ndvi=ndvi,
        ndwi=ndwi,
        ndbi=ndbi,
    )

    # ========================================================
    # Dominant prediction
    # ========================================================

    dominant = scene_analysis[
        "dominant_land_cover"
    ]

    # ========================================================
    # Return
    # ========================================================

    return {
        "dominant_prediction": {
            "label": dominant["label"],
            "confidence": dominant["confidence"],
            "confidence_percent": dominant[
                "confidence_percent"
            ],
        },

        "spectral_interpretation": {
            "NDVI": ndvi,
            "NDWI": ndwi,
            "NDBI": ndbi,
        },

        "radar_interpretation": radar,

        "category_evidence": categories,

        "methodology": {
            "type": "rule-based evidence interpretation",
            "model_predictions_preserved": True,
            "spectral_indices_used": [
                "NDVI",
                "NDWI",
                "NDBI",
            ],
            "radar_channels_used": [
                "VV",
                "VH",
            ],
            "note": (
                "Evidence status is a heuristic interpretation "
                "of the calculated measurements and should not "
                "be treated as an independent land-cover classifier."
            ),
        },
    }
