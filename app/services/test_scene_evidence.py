from pathlib import Path

from app.services.scene_evidence_service import (
    extract_scene_evidence,
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
# TEST
# ============================================================

print("Extracting scene evidence...")

evidence = extract_scene_evidence(
    s1_dir=S1_DIR,
    s2_dir=S2_DIR,
)


print()
print("=" * 70)
print("SCENE EVIDENCE")
print("=" * 70)

print("\nINPUT SHAPE")
print("-" * 70)
print(evidence["input_shape"])

print("\nCHANNEL STATISTICS")
print("-" * 70)

for channel, statistics in evidence["channel_statistics"].items():

    print(
        f"{channel:>4} | "
        f"mean={statistics['mean']:.4f} | "
        f"std={statistics['std']:.4f} | "
        f"min={statistics['min']:.4f} | "
        f"max={statistics['max']:.4f}"
    )

print("\n" + "=" * 70)