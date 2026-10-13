from typing import Dict, Any


# ============================================================
# SCENE RESULT SERVICE
# ============================================================

"""
Final result construction service for RemoteSensing-VLM.

This service combines:

    BigEarthNet prediction
            +
    Evidence confidence
            +
    Spectral evidence
            +
    Radar evidence
            +
    Category evidence
            +
    VLM explanation
            +
    Methodology

into one stable API response.

This file does NOT perform ML inference.

It only structures the final result.
"""


# ============================================================
# BUILD FINAL SCENE RESULT
# ============================================================

def build_scene_result(
    prediction: Dict[str, Any],
    scene_analysis: Dict[str, Any],
    spectral_evidence: Dict[str, Any],
    evidence_interpretation: Dict[str, Any],
    evidence_confidence: Dict[str, Any],
    explanation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the final structured scene analysis result.

    Parameters
    ----------
    prediction:
        Raw BigEarthNet prediction.

    scene_analysis:
        Scene-level interpretation.

    spectral_evidence:
        Raw spectral and radar measurements.

    evidence_interpretation:
        Interpreted spectral, radar and category evidence.

    evidence_confidence:
        Evidence-grounded confidence calculation.

    explanation:
        Evidence-grounded natural language explanation.

    Returns
    -------
    dict
        Complete structured scene result.
    """

    # ========================================================
    # SAFE EXTRACTION
    # ========================================================

    prediction = prediction or {}
    scene_analysis = scene_analysis or {}
    spectral_evidence = spectral_evidence or {}
    evidence_interpretation = (
        evidence_interpretation or {}
    )
    evidence_confidence = (
        evidence_confidence or {}
    )
    explanation = explanation or {}

    # ========================================================
    # SPECTRAL EVIDENCE
    # ========================================================

    spectral_interpretation = (
        evidence_interpretation.get(
            "spectral_interpretation",
            {},
        )
    )

    # ========================================================
    # RADAR EVIDENCE
    # ========================================================
    #
    # IMPORTANT:
    #
    # The evidence interpretation service stores
    # Sentinel-1 interpretation under:
    #
    #     radar_interpretation
    #
    # The final API exposes it as:
    #
    #     radar_evidence
    #
    # This explicit mapping prevents radar_evidence
    # from becoming {}.
    # ========================================================

    radar_evidence = (
        evidence_interpretation.get(
            "radar_interpretation",
            {},
        )
    )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------
    #
    # If radar interpretation is unavailable for any reason,
    # preserve the raw radar measurements instead of
    # returning an empty object.
    # --------------------------------------------------------

    if not radar_evidence:

        radar_evidence = (
            spectral_evidence.get(
                "radar",
                {},
            )
        )

    # ========================================================
    # CATEGORY EVIDENCE
    # ========================================================

    category_evidence = (
        evidence_interpretation.get(
            "category_evidence",
            {},
        )
    )

    # ========================================================
    # METHODOLOGY
    # ========================================================

    methodology = (
        evidence_interpretation.get(
            "methodology",
            {},
        )
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        "prediction": prediction,

        # ----------------------------------------------------
        # EVIDENCE CONFIDENCE
        # ----------------------------------------------------

        "evidence_confidence": evidence_confidence,

        # ----------------------------------------------------
        # SPECTRAL EVIDENCE
        # ----------------------------------------------------
        #
        # Preserve the original raw measurements.
        #
        # This contains:
        #
        #     indices
        #     radar
        #
        # ----------------------------------------------------

        "spectral_evidence": spectral_evidence,

        # ----------------------------------------------------
        # RADAR EVIDENCE
        # ----------------------------------------------------
        #
        # Expose interpreted Sentinel-1 evidence separately
        # for the frontend and downstream consumers.
        #
        # ----------------------------------------------------

        "radar_evidence": radar_evidence,

        # ----------------------------------------------------
        # CATEGORY EVIDENCE
        # ----------------------------------------------------

        "category_evidence": category_evidence,

        # ----------------------------------------------------
        # VLM / NATURAL LANGUAGE EXPLANATION
        # ----------------------------------------------------

        "explanation": explanation,

        # ----------------------------------------------------
        # METHODOLOGY
        # ----------------------------------------------------

        "methodology": methodology,
    }