from pathlib import Path
import pprint

from app.services.scene_pipeline_service import (
    run_scene_pipeline,
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
print("PHASE 7 — SCENE PIPELINE TEST")
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
# RUN COMPLETE PIPELINE
# ============================================================

print("\n")
print("=" * 70)
print("RUNNING COMPLETE SATELLITE SCENE PIPELINE")
print("=" * 70)

final_result = run_scene_pipeline(
    s1_dir=S1_DIR,
    s2_dir=S2_DIR,
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SCENE RESULT")
print("=" * 70)

pprint.pprint(
    final_result,
    sort_dicts=False,
)


# ============================================================
# HUMAN-READABLE SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)

prediction = final_result.get(
    "prediction",
    {},
)

explanation = final_result.get(
    "explanation",
    {},
)

evidence_confidence = final_result.get(
    "evidence_confidence",
    {},
)


print("\nDominant Land Cover:")
print(
    prediction.get(
        "label",
        "Unknown",
    )
)


print("\nModel Confidence:")

confidence = prediction.get(
    "confidence_percent",
)

if confidence is not None:
    print(f"{confidence:.2f}%")
else:
    print("Unknown")


print("\nOverall Evidence Confidence:")

overall_confidence = evidence_confidence.get(
    "overall_evidence_confidence",
)

if overall_confidence is not None:
    print(f"{overall_confidence:.2f}%")
else:
    print("Unknown")


print("\nConfidence Level:")

print(
    evidence_confidence.get(
        "confidence_level",
        "Unknown",
    )
)


print("\nEvidence Agreement:")

print(
    evidence_confidence.get(
        "evidence_agreement",
        "Unknown",
    )
)


print("\nExplanation:")

print(
    explanation.get(
        "summary",
        "No explanation available.",
    )
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 7 PIPELINE TEST COMPLETE")
print("=" * 70)