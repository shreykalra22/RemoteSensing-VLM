from pathlib import Path

from app.services.sentinel_loader import load_sentinel_bands
from app.services.spectral_evidence_service import (
    extract_spectral_evidence,
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
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load Sentinel-1 and Sentinel-2 bands
    # --------------------------------------------------------

    print("Loading satellite scene...")

    bands = load_sentinel_bands(
        s1_dir=S1_DIR,
        s2_dir=S2_DIR,
    )

    # --------------------------------------------------------
    # Extract spectral evidence
    # --------------------------------------------------------

    print("\nExtracting spectral evidence...")

    evidence = extract_spectral_evidence(
        bands
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("SPECTRAL EVIDENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Spectral indices
    # --------------------------------------------------------

    print("\nNDVI")
    print("-" * 70)

    for key, value in evidence["indices"]["NDVI"].items():
        print(f"{key}: {value}")

    print("\nNDWI")
    print("-" * 70)

    for key, value in evidence["indices"]["NDWI"].items():
        print(f"{key}: {value}")

    print("\nNDBI")
    print("-" * 70)

    for key, value in evidence["indices"]["NDBI"].items():
        print(f"{key}: {value}")

    print("\nNDMI")
    print("-" * 70)

    for key, value in evidence["indices"]["NDMI"].items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # Radar
    # --------------------------------------------------------

    print("\nRADAR")
    print("-" * 70)

    print("VV:", evidence["radar"]["VV"])
    print("VH:", evidence["radar"]["VH"])

    print("\n" + "=" * 70)