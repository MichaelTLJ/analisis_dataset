from .config import (
    CLASS_COLORS,
    FEATURE_LABELS,
    FEATURE_COLUMNS,
    MAX_EMBEDDING_VALUES_IN_PANEL,
    MAX_PER_CLASS,
)
from .data_loader import load_sdss_sample
from .embedding import build_embedding_projection


def build_visualization_payload():
    df, class_counts = load_sdss_sample(MAX_PER_CLASS)
    embedding_payload = build_embedding_projection(
        df,
        FEATURE_COLUMNS,
        MAX_EMBEDDING_VALUES_IN_PANEL,
    )
    return {
        **embedding_payload,
        "classColors": CLASS_COLORS,
        "classCounts": class_counts,
        "featureLabels": FEATURE_LABELS,
        "source": {
            "photo": "data/photoPosPlate-dr17.fits",
            "spec": "data/specObj-dr17.fits",
        },
    }

