from pathlib import Path

from app.services.prediction_service import (
    predict_satellite_scene,
)


S1_DIR = Path(
    "reben-training-scripts/scripts/data/S1"
)

S2_DIR = Path(
    "reben-training-scripts/scripts/data/S2"
)


if __name__ == "__main__":

    print("\nRunning satellite prediction...\n")

    result = predict_satellite_scene(
        s1_dir=S1_DIR,
        s2_dir=S2_DIR,
    )

    print("\n" + "=" * 70)
    print("SATELLITE SCENE ANALYSIS")
    print("=" * 70)

    print("\nModel:")
    print(result["model"])

    print("\nTask:")
    print(result["task"])

    print("\nInput:")
    print(
        f"{result['input_channels']} channels "
        f"x {result['input_size'][0]} x {result['input_size'][1]}"
    )

    print("\nThreshold:")
    print(f"{result['threshold']:.2f}")

    print("\nTOP PREDICTION")
    print("-" * 70)

    top = result["top_prediction"]

    print(
        f"{top['label']}\n"
        f"Confidence: {top['confidence_percent']:.2f}%"
    )

    print("\nDETECTED LABELS")
    print("-" * 70)

    for prediction in result["detected_labels"]:

        print(
            f"[DETECTED] {prediction['label']}"
            f" - {prediction['confidence_percent']:.2f}%"
        )

    print("\nALL PREDICTIONS")
    print("-" * 70)

    for rank, prediction in enumerate(
        result["predictions"],
        start=1,
    ):

        print(
            f"{rank:02d}. "
            f"{prediction['label']:<70} "
            f"{prediction['confidence_percent']:7.2f}%"
        )
