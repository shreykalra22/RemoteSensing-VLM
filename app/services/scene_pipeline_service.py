# ============================================================
# SCENE PIPELINE SERVICE
# ============================================================

"""
Production orchestration layer for the remote sensing scene
analysis pipeline.

This service coordinates:

    Sentinel-1 / Sentinel-2 loading
            ↓
    BigEarthNet prediction
            ↓
    Scene analysis
            ↓
    Spectral evidence extraction
            ↓
    Evidence interpretation
            ↓
    Evidence confidence
            ↓
    VLM explanation
            ↓
    Final scene result

This file does NOT implement ML inference or evidence logic.

It only orchestrates the existing services.
"""

from pathlib import Path
from typing import Dict, Any


# ============================================================
# SERVICE IMPORTS
# ============================================================

from app.services.prediction_service import predict_bands

from app.services.scene_analysis_service import (
    analyze_scene,
)

from app.services.sentinel_loader import (
    load_sentinel_bands,
)

from app.services.spectral_evidence_service import (
    extract_spectral_evidence,
)

from app.services.evidence_interpretation_service import (
    interpret_scene_evidence,
)

from app.services.evidence_confidence_service import (
    calculate_evidence_confidence,
)

from app.services.vlm_explanation_service import (
    generate_scene_explanation,
)

from app.services.scene_result_service import (
    build_scene_result,
)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_scene_pipeline(
    s1_dir: Path,
    s2_dir: Path,
) -> Dict[str, Any]:
    """
    Run the complete remote sensing scene analysis pipeline.

    Parameters
    ----------
    s1_dir : Path
        Directory containing Sentinel-1 data.

    s2_dir : Path
        Directory containing Sentinel-2 data.

    Returns
    -------
    dict
        Complete structured scene analysis result.

    Pipeline
    --------
    1. BigEarthNet prediction
    2. Scene analysis
    3. Satellite band loading
    4. Spectral evidence extraction
    5. Evidence interpretation
    6. Evidence confidence calculation
    7. VLM explanation
    8. Final result construction
    """

    # ========================================================
    # VALIDATE INPUT DIRECTORIES
    # ========================================================

    s1_dir = Path(s1_dir)
    s2_dir = Path(s2_dir)

    if not s1_dir.exists():
        raise FileNotFoundError(
            f"Sentinel-1 directory does not exist: {s1_dir}"
        )

    if not s2_dir.exists():
        raise FileNotFoundError(
            f"Sentinel-2 directory does not exist: {s2_dir}"
        )

    # ========================================================
    # STEP 1 — LOAD AND VALIDATE BANDS ONCE
    # ========================================================

    bands = load_sentinel_bands(
        s1_dir=s1_dir,
        s2_dir=s2_dir,
    )

    # STEP 2 — BIGEARTHNET PREDICTION
    prediction = predict_bands(bands)

    # ========================================================
    # STEP 3 — SCENE ANALYSIS
    # ========================================================

    scene_analysis = analyze_scene(
        prediction
    )

    # ========================================================
    # STEP 4 — SPECTRAL EVIDENCE
    # ========================================================

    spectral_evidence = extract_spectral_evidence(
        bands
    )

    # ========================================================
    # STEP 5 — EVIDENCE INTERPRETATION
    # ========================================================

    evidence_interpretation = interpret_scene_evidence(
        prediction=prediction,
        scene_analysis=scene_analysis,
        spectral_evidence=spectral_evidence,
    )

    # ========================================================
    # STEP 6 — EVIDENCE CONFIDENCE
    # ========================================================

    evidence_confidence = calculate_evidence_confidence(
        prediction=prediction,
        scene_analysis=scene_analysis,
        evidence_interpretation=evidence_interpretation,
    )

    # ========================================================
    # STEP 7 — VLM EXPLANATION
    # ========================================================

    explanation = generate_scene_explanation(
        prediction=prediction,
        scene_analysis=scene_analysis,
        evidence_interpretation=evidence_interpretation,
        evidence_confidence=evidence_confidence,
    )

    # ========================================================
    # STEP 8 — BUILD FINAL RESULT
    # ========================================================

    final_result = build_scene_result(
        prediction=prediction,
        scene_analysis=scene_analysis,
        spectral_evidence=spectral_evidence,
        evidence_interpretation=evidence_interpretation,
        evidence_confidence=evidence_confidence,
        explanation=explanation,
    )

    # ========================================================
    # RETURN FINAL RESULT
    # ========================================================

    return final_result
