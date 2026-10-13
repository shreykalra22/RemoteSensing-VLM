
from pathlib import Path
from pprint import pprint


from app.services.prediction_service import (
    predict_satellite_scene,
)

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

from app.services.vlm_explanation_service import (
    generate_scene_explanation,
)

from app.services.evidence_confidence_service import (
    calculate_evidence_confidence,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

S1_DIR = (
    BASE_DIR
    / "reben-training-scripts"
    / "scripts"
    / "data"
    / "S1"
)

S2_DIR = (
    BASE_DIR
    / "reben-training-scripts"
    / "scripts"
    / "data"
    / "S2"
)


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning BigEarthNet prediction...")

prediction = predict_satellite_scene(
    s1_dir=S1_DIR,
    s2_dir=S2_DIR,
)


# ============================================================
# SCENE ANALYSIS
# ============================================================

print("\nRunning scene analysis...")

scene_analysis = analyze_scene(
    prediction
)


# ============================================================
# LOAD BANDS
# ============================================================

print("\nLoading satellite bands...")

bands = load_sentinel_bands(
    s1_dir=S1_DIR,
    s2_dir=S2_DIR,
)


# ============================================================
# SPECTRAL EVIDENCE
# ============================================================

print("\nExtracting spectral evidence...")

spectral_evidence = extract_spectral_evidence(
    bands
)


# ============================================================
# EVIDENCE INTERPRETATION
# ============================================================

print("\nInterpreting scene evidence...")

evidence_interpretation = interpret_scene_evidence(
    prediction=prediction,
    scene_analysis=scene_analysis,
    spectral_evidence=spectral_evidence,
)


# ============================================================
# DEBUG EVIDENCE STRUCTURE
# ============================================================

print("\n")
print("=" * 70)
print("RAW EVIDENCE INTERPRETATION")
print("=" * 70)

pprint(
    evidence_interpretation,
    sort_dicts=False,
)


# ============================================================
# VLM EXPLANATION
# ============================================================

print("\nGenerating explanation...")

explanation = generate_scene_explanation(
    prediction=prediction,
    scene_analysis=scene_analysis,
    evidence_interpretation=evidence_interpretation,
)


# ============================================================
# CONFIDENCE
# ============================================================

print("\nCalculating evidence confidence...")

confidence = calculate_evidence_confidence(
    prediction=prediction,
    scene_analysis=scene_analysis,
    evidence_interpretation=evidence_interpretation,
)


# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 70)
print("EVIDENCE CONFIDENCE")
print("=" * 70)


print("\nDOMINANT LAND COVER")
print("-" * 70)

print(
    scene_analysis[
        "dominant_land_cover"
    ]["label"]
)

print(
    f"Model confidence: "
    f"{scene_analysis['dominant_land_cover']['confidence_percent']:.2f}%"
)


print("\nEVIDENCE CONFIDENCE")
print("-" * 70)

for key, value in confidence.items():

    print(
        f"{key}: {value}"
    )


print("\n")
print("=" * 70)
print("EVIDENCE CONFIDENCE COMPLETE")
print("=" * 70)
