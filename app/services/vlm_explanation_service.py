"""
Rule-Based Scene Explanation Service

The current implementation uses deterministic rules.
It does NOT use an actual VLM or LLM.
"""

from typing import Any, Dict


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        value = float(value)

        if value != value:
            return default

        return value

    except (TypeError, ValueError):
        return default


def _get_label(
    prediction: Dict[str, Any],
    scene_analysis: Dict[str, Any],
) -> str:

    # --------------------------------------------------------
    # IMPORTANT:
    # top_prediction is an OBJECT in your API response.
    # Extract only its label.
    # --------------------------------------------------------

    top_prediction = (
        prediction.get("top_prediction")
        or scene_analysis.get("top_prediction")
    )

    if isinstance(
        top_prediction,
        dict,
    ):

        return str(
            top_prediction.get("label")
            or top_prediction.get("name")
            or top_prediction.get("category")
            or "Unknown"
        )

    # --------------------------------------------------------
    # Flat values
    # --------------------------------------------------------

    value = (
        scene_analysis.get("dominant_land_cover")
        or scene_analysis.get("dominant_category")
        or prediction.get("dominant_land_cover")
        or prediction.get("dominant_category")
    )

    if isinstance(value, dict):

        return str(
            value.get("label")
            or value.get("name")
            or value.get("category")
            or "Unknown"
        )

    return str(
        value or "Unknown"
    )


def _get_probability(
    prediction: Dict[str, Any],
    scene_analysis: Dict[str, Any],
) -> float:

    top_prediction = (
        prediction.get("top_prediction")
        or scene_analysis.get("top_prediction")
    )

    if isinstance(
        top_prediction,
        dict,
    ):

        value = (
            top_prediction.get("confidence")
            or top_prediction.get("probability")
            or top_prediction.get("score")
            or top_prediction.get("confidence_percent")
        )

        if value is not None:

            value = _safe_float(value)

            if value <= 1:
                value *= 100

            return value

    value = (
        scene_analysis.get("model_confidence")
        or scene_analysis.get("dominant_confidence")
        or prediction.get("model_confidence")
        or prediction.get("dominant_confidence")
    )

    value = _safe_float(value)

    if value <= 1:
        value *= 100

    return value


def generate_scene_explanation(
    prediction: Dict[str, Any] = None,
    scene_analysis: Dict[str, Any] = None,
    evidence_interpretation: Dict[str, Any] = None,
    evidence_confidence: Dict[str, Any] = None,
) -> Dict[str, Any]:

    prediction = prediction or {}
    scene_analysis = scene_analysis or {}
    evidence_interpretation = (
        evidence_interpretation or {}
    )
    evidence_confidence = (
        evidence_confidence or {}
    )

    # ========================================================
    # MODEL
    # ========================================================

    label = _get_label(
        prediction,
        scene_analysis,
    )

    probability = _get_probability(
        prediction,
        scene_analysis,
    )

    # ========================================================
    # CATEGORY EVIDENCE
    # ========================================================

    category_evidence = (
        evidence_interpretation.get(
            "category_evidence"
        )
        or {}
    )

    supported = []
    absences = []
    weak = []

    if isinstance(
        category_evidence,
        dict,
    ):

        for category, item in category_evidence.items():

            if not isinstance(item, dict):
                continue

            status = str(
                item.get("evidence_status")
                or item.get("status")
                or item.get("verdict")
                or ""
            ).lower()

            category_name = str(
                category
            ).replace(
                "_",
                " ",
            )

            if (
                status == "supported"
                or "supported" in status
                and "not_supported" not in status
            ):

                supported.append(
                    category_name
                )

            elif (
                "consistent_absence"
                in status
                or "consistent absence"
                in status
            ):

                absences.append(
                    category_name
                )

            elif "weak" in status:

                weak.append(
                    category_name
                )

    # ========================================================
    # EVIDENCE SCORE
    # ========================================================

    heuristic_score = _safe_float(
        evidence_confidence.get(
            "heuristic_evidence_score"
        )
    )

    consistency_level = str(
        evidence_confidence.get(
            "consistency_level"
        )
        or "unknown"
    ).replace(
        "_",
        " ",
    )

    agreement = str(
        evidence_confidence.get(
            "evidence_agreement"
        )
        or "limited"
    ).replace(
        "_",
        " ",
    )

    # ========================================================
    # EXPLANATION
    # ========================================================

    sentences = []

    sentences.append(
        f'The BigEarthNet model assigns the highest score '
        f'to "{label}" with a top-label probability of '
        f'{probability:.2f}%.'
    )

    if supported:

        sentences.append(
            "The supplementary category analysis supports "
            + ", ".join(supported)
            + "."
        )

    if absences:

        sentences.append(
            "The evidence also indicates consistent absence "
            + "of "
            + ", ".join(absences)
            + "."
        )

    if weak:

        sentences.append(
            "Weak evidence was observed for "
            + ", ".join(weak)
            + "."
        )

    sentences.append(
        f"The resulting heuristic evidence score is "
        f"{heuristic_score:.2f}%, classified as "
        f"{consistency_level}, with {agreement} "
        f"agreement between the model prediction and "
        f"supplementary evidence."
    )

    summary = " ".join(
        sentences
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "summary": summary,

        "highest_scoring_label":
            label,

        "top_label_probability":
            round(
                probability,
                2,
            ),

        "supporting_land_covers":
            supported,

        "consistent_absence_categories":
            absences,

        "weak_categories":
            weak,

        "category_evidence":
            category_evidence,

        "evidence_consistency": {

            "heuristic_score":
                heuristic_score,

            "level":
                consistency_level,

            "agreement":
                agreement,
        },

        "explanation_type":
            "rule_based",

        "note":
            "This explanation is generated "
            "using deterministic rules from "
            "model predictions and satellite "
            "measurements. No LLM or VLM is used.",
    }