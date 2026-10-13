from pathlib import Path
import pprint


from app.services.prediction_service import (
    predict_satellite_scene,
)

from app.services.scene_result_service import (
    build_scene_result,
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
# PATH DEBUG
# ============================================================

print("\n")
print("=" * 70)
print("PATH DEBUG")
print("=" * 70)

print("\nBASE_DIR:")
print(BASE_DIR)

print("\nS1_DIR:")
print(S1_DIR)
print("Exists:", S1_DIR.exists())

print("\nS2_DIR:")
print(S2_DIR)
print("Exists:", S2_DIR.exists())


# ============================================================
# FIND B02
# ============================================================

print("\n")
print("Searching for B02...")

b02_matches = list(
    BASE_DIR.rglob("*_B02.tiff")
)

if b02_matches:

    print("\nB02 files found:")

    for file in b02_matches:
        print(file)

else:

    print("\nNO B02 FILE FOUND.")


# ============================================================
# SHOW S2 FILES
# ============================================================

print("\n")
print("S2 DIRECTORY CONTENTS:")

if S2_DIR.exists():

    files = list(
        S2_DIR.rglob("*")
    )

    if files:

        for file in files:
            print(file)

    else:

        print("S2 directory is empty.")

else:

    print("S2 directory does not exist.")


print("\n")
print("=" * 70)
print("END PATH DEBUG")
print("=" * 70)


# ============================================================
# STEP 1 — BIGEARTHNET PREDICTION
# ============================================================

print("\nRunning BigEarthNet prediction...")

prediction = predict_satellite_scene(
    s1_dir=S1_DIR,
    s2_dir=S2_DIR,
)


# ============================================================
# STEP 2 — SCENE ANALYSIS
# ============================================================

print("\nRunning scene analysis...")

scene_analysis = analyze_scene(
    prediction
)


# ============================================================
# STEP 3 — LOAD SATELLITE BANDS
# ============================================================

print("\nLoading satellite bands...")

bands = load_sentinel_bands(
    s1_dir=S1_DIR,
    s2_dir=S2_DIR,
)


# ============================================================
# STEP 4 — SPECTRAL EVIDENCE
# ============================================================

print("\nExtracting spectral evidence...")

spectral_evidence = extract_spectral_evidence(
    bands
)


# ============================================================
# STEP 5 — EVIDENCE INTERPRETATION
# ============================================================

print("\nInterpreting scene evidence...")

evidence_interpretation = interpret_scene_evidence(
    prediction=prediction,
    scene_analysis=scene_analysis,
    spectral_evidence=spectral_evidence,
)


# ============================================================
# STEP 6 — EVIDENCE CONFIDENCE
# ============================================================

print("\nCalculating evidence confidence...")

evidence_confidence = calculate_evidence_confidence(
    prediction=prediction,
    scene_analysis=scene_analysis,
    evidence_interpretation=evidence_interpretation,
)


# ============================================================
# STEP 7 — VLM EXPLANATION
# ============================================================

print("Generating explanation...")

explanation = generate_scene_explanation(
    prediction=prediction,
    scene_analysis=scene_analysis,
    evidence_interpretation=evidence_interpretation,
    evidence_confidence=evidence_confidence,
)


# ============================================================
# STEP 8 — BUILD FINAL RESULT
# ============================================================

print("Building final scene result...")

final_result = build_scene_result(
    prediction=prediction,
    scene_analysis=scene_analysis,
    spectral_evidence=spectral_evidence,
    evidence_interpretation=evidence_interpretation,
    evidence_confidence=evidence_confidence,
    explanation=explanation,
)


# ============================================================
# FINAL STRUCTURED RESULT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL STRUCTURED SCENE RESULT")
print("=" * 70)

pprint.pprint(
    final_result,
    sort_dicts=False
)

print("=" * 70)
print("FINAL RESULT COMPLETE")
print("=" * 70)


# ============================================================
# HUMAN-READABLE DISPLAY
# ============================================================

print("\n")
print("=" * 70)
print("EVIDENCE-GROUNDED SCENE EXPLANATION")
print("=" * 70)


# ============================================================
# SUMMARY
# ============================================================

print("\nSUMMARY")
print("-" * 70)

print(
    explanation.get(
        "summary",
        "No summary available."
    )
)


# ============================================================
# DOMINANT LAND COVER
# ============================================================

print("\nDOMINANT LAND COVER")
print("-" * 70)

print(
    explanation.get(
        "dominant_land_cover",
        "Unknown"
    )
)

dominant_confidence = explanation.get(
    "dominant_confidence"
)

if dominant_confidence is not None:

    print(
        f"Confidence: "
        f"{dominant_confidence:.2f}%"
    )


# ============================================================
# SUPPORTING LAND COVERS
# ============================================================

print("\nSUPPORTING LAND COVERS")
print("-" * 70)

supporting_land_covers = explanation.get(
    "supporting_land_covers",
    []
)

if supporting_land_covers:

    for label in supporting_land_covers:

        print(
            f"- {label}"
        )

else:

    print("None")


# ============================================================
# SPECTRAL EVIDENCE
# ============================================================

print("\nSPECTRAL EVIDENCE")
print("-" * 70)

explanation_spectral = explanation.get(
    "spectral_evidence",
    {}
)

for name, value in explanation_spectral.items():

    if isinstance(value, (int, float)):

        print(
            f"{name}: {value:.4f}"
        )

    else:

        print(
            f"{name}: {value}"
        )


# ============================================================
# EVIDENCE CONFIDENCE
# ============================================================

print("\nEVIDENCE CONFIDENCE")
print("-" * 70)

overall_confidence = evidence_confidence.get(
    "overall_evidence_confidence"
)

confidence_level = evidence_confidence.get(
    "confidence_level"
)

evidence_agreement = evidence_confidence.get(
    "evidence_agreement"
)

model_confidence = evidence_confidence.get(
    "model_confidence_percent"
)

spectral_score = evidence_confidence.get(
    "spectral_evidence_score"
)

category_score = evidence_confidence.get(
    "category_agreement_score"
)


if overall_confidence is not None:

    print(
        f"Overall Evidence Confidence: "
        f"{overall_confidence:.2f}%"
    )

if confidence_level:

    print(
        f"Confidence Level: "
        f"{confidence_level}"
    )

if evidence_agreement:

    print(
        f"Evidence Agreement: "
        f"{evidence_agreement}"
    )

if model_confidence is not None:

    print(
        f"Model Confidence: "
        f"{model_confidence:.2f}%"
    )

if spectral_score is not None:

    print(
        f"Spectral Evidence Score: "
        f"{spectral_score:.2f}"
    )

if category_score is not None:

    print(
        f"Category Agreement Score: "
        f"{category_score:.2f}"
    )


# ============================================================
# CATEGORY EVIDENCE
# ============================================================

print("\nCATEGORY EVIDENCE")
print("-" * 70)

category_evidence = final_result.get(
    "category_evidence",
    {}
)

for category, details in category_evidence.items():

    status = details.get(
        "evidence_status",
        "unknown"
    )

    print(
        f"{category}: {status}"
    )


# ============================================================
# METHODOLOGY
# ============================================================

print("\nMETHODOLOGY")
print("-" * 70)

methodology = final_result.get(
    "methodology",
    {}
)

print(
    f"Type: "
    f"{methodology.get('type', 'Unknown')}"
)

print(
    f"Model predictions preserved: "
    f"{methodology.get('model_predictions_preserved', False)}"
)

print(
    "Spectral indices used: "
    f"{methodology.get('spectral_indices_used', [])}"
)

print(
    "Radar channels used: "
    f"{methodology.get('radar_channels_used', [])}"
)

print(
    f"Note: "
    f"{methodology.get('note', '')}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("EXPLANATION COMPLETE")
print("=" * 70)