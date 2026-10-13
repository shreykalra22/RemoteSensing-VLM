from pathlib import Path

from app.services.prediction_service import predict_satellite_scene
from app.services.scene_analysis_service import analyze_scene


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

S1_DIR = BASE_DIR / "reben-training-scripts" / "scripts" / "data" / "S1"
S2_DIR = BASE_DIR / "reben-training-scripts" / "scripts" / "data" / "S2"


# ============================================================
# RUN PREDICTION
# ============================================================

print("\nRunning BigEarthNet prediction...\n")

prediction_result = predict_satellite_scene(
    s1_dir=S1_DIR,
    s2_dir=S2_DIR,
)


# ============================================================
# RUN SCENE ANALYSIS
# ============================================================

print("\nAnalyzing scene...\n")

analysis = analyze_scene(
    prediction_result
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 70)
print("SCENE ANALYSIS")
print("=" * 70)

print("\nDOMINANT LAND COVER")
print("-" * 70)

dominant = analysis["dominant_land_cover"]

print(
    f"{dominant['label']} "
    f"({dominant['confidence_percent']:.2f}%)"
)


print("\nCATEGORIES")
print("-" * 70)

for category, information in analysis["categories"].items():

    status = "DETECTED" if information["detected"] else "NOT DETECTED"

    print(f"\n{category}: {status}")

    for label in information["labels"]:

        print(
            f"  - {label['label']} "
            f"({label['confidence_percent']:.2f}%)"
        )


print("\n" + "=" * 70)