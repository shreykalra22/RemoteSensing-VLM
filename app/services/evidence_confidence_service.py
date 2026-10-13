"""
Evidence Consistency Service

Calculates a HEURISTIC consistency score between:

1. BigEarthNet model prediction
2. Sentinel-2 spectral indicators
3. Category-level evidence

This is NOT a calibrated confidence score.
It is NOT an independent classifier.
It is only a heuristic interpretation score.
"""

from typing import Any, Dict


STATUS_SCORES = {
    "strong": 1.00,
    "moderate": 0.75,
    "weak": 0.25,
    "not_supported": 0.00,
    "unknown": 0.00,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)

        if value != value:
            return default

        return value

    except (TypeError, ValueError):
        return default


def _normalize_probability(value: Any) -> float:
    """
    Convert probability into 0-1.

    0.735 -> 0.735
    73.5  -> 0.735
    """

    value = _safe_float(value)

    if value > 1.0:
        value = value / 100.0

    return max(0.0, min(1.0, value))


def _extract_prediction_label(
    prediction: Dict[str, Any],
    scene_analysis: Dict[str, Any],
) -> str:

    # --------------------------------------------------------
    # Scene analysis
    # --------------------------------------------------------

    value = (
        scene_analysis.get("dominant_land_cover")
        or scene_analysis.get("dominant_category")
        or scene_analysis.get("top_prediction")
    )

    # If object was returned
    if isinstance(value, dict):
        return str(
            value.get("label")
            or value.get("name")
            or value.get("category")
            or "Unknown"
        )

    if value:
        return str(value)

    # --------------------------------------------------------
    # Raw prediction
    # --------------------------------------------------------

    value = (
        prediction.get("dominant_land_cover")
        or prediction.get("dominant_category")
        or prediction.get("top_prediction")
    )

    if isinstance(value, dict):
        return str(
            value.get("label")
            or value.get("name")
            or value.get("category")
            or "Unknown"
        )

    if value:
        return str(value)

    return "Unknown"


def _extract_prediction_probability(
    prediction: Dict[str, Any],
    scene_analysis: Dict[str, Any],
) -> float:

    # --------------------------------------------------------
    # First check nested top_prediction object.
    # This is the important fix.
    # --------------------------------------------------------

    top_prediction = (
        prediction.get("top_prediction")
        or scene_analysis.get("top_prediction")
    )

    if isinstance(top_prediction, dict):

        value = (
            top_prediction.get("confidence")
            or top_prediction.get("probability")
            or top_prediction.get("score")
            or top_prediction.get("confidence_percent")
        )

        if value is not None:
            return _normalize_probability(value)

    # --------------------------------------------------------
    # Flat fields
    # --------------------------------------------------------

    value = (
        scene_analysis.get("model_confidence")
        or scene_analysis.get("dominant_confidence")
        or scene_analysis.get("confidence")
        or prediction.get("model_confidence")
        or prediction.get("dominant_confidence")
        or prediction.get("confidence")
    )

    if value is not None:
        return _normalize_probability(value)

    return 0.0


def _status_score(status: Any) -> float:

    status = str(
        status or ""
    ).lower().strip()

    return STATUS_SCORES.get(
        status,
        0.0,
    )


def _category_score(item: Any) -> float:

    if not isinstance(item, dict):
        return 0.0

    status = str(
        item.get("evidence_status")
        or item.get("status")
        or item.get("verdict")
        or ""
    ).lower().strip()

    if "consistent_absence" in status:
        return 1.0

    if "consistent absence" in status:
        return 1.0

    if status == "supported":
        return 1.0

    if "supported" in status:
        return 1.0

    if "weak" in status:
        return 0.5

    if "contradicted" in status:
        return 0.0

    if "not_detected" in status:
        return 0.0

    if "not detected" in status:
        return 0.0

    return 0.0


def calculate_evidence_confidence(
    prediction: Dict[str, Any] = None,
    scene_analysis: Dict[str, Any] = None,
    evidence_interpretation: Dict[str, Any] = None,
) -> Dict[str, Any]:

    prediction = prediction or {}
    scene_analysis = scene_analysis or {}
    evidence_interpretation = evidence_interpretation or {}

    # ========================================================
    # MODEL
    # ========================================================

    highest_scoring_label = _extract_prediction_label(
        prediction,
        scene_analysis,
    )

    model_probability = _extract_prediction_probability(
        prediction,
        scene_analysis,
    )

    # ========================================================
    # SPECTRAL EVIDENCE
    # ========================================================

    spectral_interpretation = (
        evidence_interpretation.get(
            "spectral_interpretation"
        )
        or {}
    )

    spectral_scores = []

    for index_name in [
        "ndvi",
        "ndwi",
        "ndbi",
    ]:

        index_info = spectral_interpretation.get(
            index_name
        )

        if isinstance(index_info, dict):

            status = index_info.get(
                "status"
            )

            spectral_scores.append(
                _status_score(status)
            )

    if spectral_scores:

        spectral_consistency = (
            sum(spectral_scores)
            / len(spectral_scores)
        )

    else:

        spectral_consistency = 0.0

    # ========================================================
    # CATEGORY EVIDENCE
    # ========================================================

    category_evidence = (
        evidence_interpretation.get(
            "category_evidence"
        )
        or {}
    )

    category_scores = []

    supported_categories = 0
    consistent_absences = 0
    weak_categories = 0

    if isinstance(
        category_evidence,
        dict,
    ):

        for _, item in category_evidence.items():

            if not isinstance(
                item,
                dict,
            ):
                continue

            category_scores.append(
                _category_score(item)
            )

            status = str(
                item.get("evidence_status")
                or item.get("status")
                or item.get("verdict")
                or ""
            ).lower().strip()

            if status == "supported" or (
                "supported" in status
                and "not_supported" not in status
            ):

                supported_categories += 1

            elif (
                "consistent_absence"
                in status
                or "consistent absence"
                in status
            ):

                consistent_absences += 1

            elif "weak" in status:

                weak_categories += 1

    if category_scores:

        category_consistency = (
            sum(category_scores)
            / len(category_scores)
        )

    else:

        category_consistency = 0.0

    total_categories = len(
        category_scores
    )

    # ========================================================
    # HEURISTIC SCORE
    # ========================================================
    #
    # 50% model probability
    # 30% spectral consistency
    # 20% category consistency
    #
    # This is a heuristic score, NOT probability.
    # ========================================================

    heuristic_score = (
        0.50 * model_probability
        + 0.30 * spectral_consistency
        + 0.20 * category_consistency
    )

    heuristic_score = max(
        0.0,
        min(
            1.0,
            heuristic_score,
        ),
    )

    # ========================================================
    # LEVEL
    # ========================================================

    if heuristic_score >= 0.80:
        consistency_level = "very_high"

    elif heuristic_score >= 0.65:
        consistency_level = "high"

    elif heuristic_score >= 0.50:
        consistency_level = "moderate"

    elif heuristic_score >= 0.35:
        consistency_level = "low"

    else:
        consistency_level = "very_low"

    # ========================================================
    # AGREEMENT
    # ========================================================

    if (
        spectral_consistency >= 0.65
        and category_consistency >= 0.65
    ):

        evidence_agreement = "strong"

    elif (
        spectral_consistency >= 0.40
        or category_consistency >= 0.40
    ):

        evidence_agreement = "moderate"

    else:

        evidence_agreement = "limited"

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "highest_scoring_label":
            highest_scoring_label,

        "top_label_probability":
            round(
                model_probability * 100.0,
                2,
            ),

        "spectral_consistency_score":
            round(
                spectral_consistency * 100.0,
                2,
            ),

        "category_consistency_score":
            round(
                category_consistency * 100.0,
                2,
            ),

        "heuristic_evidence_score":
            round(
                heuristic_score * 100.0,
                2,
            ),

        "consistency_level":
            consistency_level,

        "evidence_agreement":
            evidence_agreement,

        "supported_categories":
            supported_categories,

        "consistent_absences":
            consistent_absences,

        "weak_categories":
            weak_categories,

        "total_categories_evaluated":
            total_categories,

        "formula":
            "0.50 × model probability + "
            "0.30 × spectral consistency + "
            "0.20 × category consistency",

        "note":
            "Heuristic consistency score for "
            "interpretation only. It is not a "
            "calibrated probability, independent "
            "classifier, or scientifically validated "
            "confidence measure.",
    }