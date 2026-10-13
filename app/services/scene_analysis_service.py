# ============================================================
# SCENE ANALYSIS SERVICE
# ============================================================

"""
Converts BigEarthNet predictions and satellite evidence into
higher-level scene information.

This layer does NOT perform ML inference.
It interprets BigEarthNet predictions and spectral evidence.
"""



# ============================================================
# LAND-COVER GROUPS
# ============================================================

AGRICULTURE_LABELS = {
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Complex cultivation patterns",
    "Permanent crops",
    "Arable land",
    "Pastures",
    "Agro-forestry areas",
}


FOREST_LABELS = {
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
}


WOODLAND_LABELS = {
    "Transitional woodland, shrub",
}


GRASSLAND_LABELS = {
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
}


WATER_LABELS = {
    "Inland waters",
    "Inland wetlands",
    "Marine waters",
    "Coastal wetlands",
}


URBAN_LABELS = {
    "Urban fabric",
    "Industrial or commercial units",
}


BARE_LAND_LABELS = {
    "Beaches, dunes, sands",
}


# ============================================================
# HELPERS
# ============================================================

def _get_detected_labels(predictions):
    """
    Extract only labels detected by the model.
    """

    return [
        prediction
        for prediction in predictions
        if prediction.get("detected", False)
    ]


def _find_category_labels(predictions, category_labels):
    """
    Find detected predictions belonging to a category.
    """

    return [
        prediction
        for prediction in predictions
        if (
            prediction["label"] in category_labels
            and prediction.get("detected", False)
        )
    ]


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_scene(prediction_result, image=None):
    """
    Analyze BigEarthNet predictions and optional
    satellite spectral evidence.

    Parameters
    ----------
    prediction_result : dict
        Output returned by predict_satellite_scene().

    image : torch.Tensor, optional
        Raw satellite image with shape:

            [12, H, W]

        Channel order:

            VV, VH, B02, B03, B04, B05,
            B06, B07, B08, B8A, B11, B12

    Returns
    -------
    dict
        Structured scene analysis.
    """

    predictions = prediction_result["predictions"]

    detected = _get_detected_labels(predictions)

    agriculture = _find_category_labels(
        predictions,
        AGRICULTURE_LABELS,
    )

    forest = _find_category_labels(
        predictions,
        FOREST_LABELS,
    )

    woodland = _find_category_labels(
        predictions,
        WOODLAND_LABELS,
    )

    grassland = _find_category_labels(
        predictions,
        GRASSLAND_LABELS,
    )

    water = _find_category_labels(
        predictions,
        WATER_LABELS,
    )

    urban = _find_category_labels(
        predictions,
        URBAN_LABELS,
    )

    bare_land = _find_category_labels(
        predictions,
        BARE_LAND_LABELS,
    )

    # --------------------------------------------------------
    # Dominant land cover
    # --------------------------------------------------------

    top_prediction = prediction_result["top_prediction"]

    # --------------------------------------------------------
    # Spectral evidence
    # --------------------------------------------------------

    # Spectral evidence is calculated once by the pipeline after raw bands
    # have been validated. The optional ``image`` argument remains only for
    # backwards compatibility with early local scripts.
    spectral_evidence = None

    # --------------------------------------------------------
    # Return structured analysis
    # --------------------------------------------------------

    return {
        "dominant_land_cover": top_prediction,

        "detected_land_covers": detected,

        "categories": {
            "agriculture": {
                "detected": len(agriculture) > 0,
                "labels": agriculture,
            },

            "forest": {
                "detected": len(forest) > 0,
                "labels": forest,
            },

            "woodland_and_shrub": {
                "detected": len(woodland) > 0,
                "labels": woodland,
            },

            "grassland_and_sparse_vegetation": {
                "detected": len(grassland) > 0,
                "labels": grassland,
            },

            "water": {
                "detected": len(water) > 0,
                "labels": water,
            },

            "urban": {
                "detected": len(urban) > 0,
                "labels": urban,
            },

            "bare_land": {
                "detected": len(bare_land) > 0,
                "labels": bare_land,
            },
        },

        "spectral_evidence": spectral_evidence,
    }
