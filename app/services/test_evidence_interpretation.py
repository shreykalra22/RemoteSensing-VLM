from pathlib import Path

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

    # ========================================================
    # STEP 1 — BIGEARTHNET PREDICTION
    # ========================================================

    print("\nRunning BigEarthNet prediction...")

    prediction = predict_satellite_scene(
        s1_dir=S1_DIR,
        s2_dir=S2_DIR,
    )

    # ========================================================
    # STEP 2 — SCENE ANALYSIS
    # ========================================================

    print("\nRunning scene analysis...")

    scene_analysis = analyze_scene(
        prediction
    )

    # ========================================================
    # STEP 3 — LOAD RAW BANDS
    # ========================================================

    print("\nLoading satellite bands...")

    bands = load_sentinel_bands(
        s1_dir=S1_DIR,
        s2_dir=S2_DIR,
    )

    # ========================================================
    # STEP 4 — SPECTRAL EVIDENCE
    # ========================================================

    print("\nExtracting spectral evidence...")

    spectral_evidence = extract_spectral_evidence(
        bands
    )

    # ========================================================
    # STEP 5 — INTERPRET EVIDENCE
    # ========================================================

    print("\nInterpreting scene evidence...")

    evidence = interpret_scene_evidence(
        prediction=prediction,
        scene_analysis=scene_analysis,
        spectral_evidence=spectral_evidence,
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n" + "=" * 70)
    print("EVIDENCE INTERPRETATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Dominant prediction
    # --------------------------------------------------------

    dominant = evidence[
        "dominant_prediction"
    ]

    print("\nDOMINANT PREDICTION")
    print("-" * 70)

    print(
        f"Label: {dominant['label']}"
    )

    print(
        f"Confidence: "
        f"{dominant['confidence_percent']:.2f}%"
    )

    # --------------------------------------------------------
    # Spectral interpretation
    # --------------------------------------------------------

    print("\nSPECTRAL INTERPRETATION")
    print("-" * 70)

    spectral = evidence[
        "spectral_interpretation"
    ]

    for index_name, result in spectral.items():

        print(
            f"{index_name}: "
            f"value={result['value']:.4f} | "
            f"signal={result['signal']} | "
            f"support={result['support_level']}"
        )

        print(
            f"  {result['interpretation']}"
        )

    # --------------------------------------------------------
    # Radar
    # --------------------------------------------------------

    print("\nRADAR INTERPRETATION")
    print("-" * 70)

    radar = evidence[
        "radar_interpretation"
    ]

    print(
        f"VV mean: "
        f"{radar['VV']['mean']:.4f}"
    )

    print(
        f"VH mean: "
        f"{radar['VH']['mean']:.4f}"
    )

    print(
        f"Interpretation: "
        f"{radar['interpretation']}"
    )

    # --------------------------------------------------------
    # Category evidence
    # --------------------------------------------------------

    print("\nCATEGORY EVIDENCE")
    print("-" * 70)

    categories = evidence[
        "category_evidence"
    ]

    for category, result in categories.items():

        print(
            f"\n{category.upper()}"
        )

        print(
            f"Model detected: "
            f"{result['model_detected']}"
        )

        print(
            f"Evidence status: "
            f"{result['evidence_status']}"
        )

        print(
            f"Reason: "
            f"{result['reason']}"
        )

    # --------------------------------------------------------
    # Methodology
    # --------------------------------------------------------

    print("\nMETHODOLOGY")
    print("-" * 70)

    print(
        evidence["methodology"]["note"]
    )

    print("\n" + "=" * 70)