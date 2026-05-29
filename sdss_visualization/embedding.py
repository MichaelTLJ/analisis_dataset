import numpy as np

from .grouping import add_science_groups
from .projection import pca_2d, standardize_matrix


def _safe_float(value):
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def build_embedding_projection(df, feature_columns, max_vector_values):
    coordinate_columns = [
        "RA",
        "DEC",
        "RAERR",
        "DECERR",
        "CX",
        "CY",
        "CZ",
        "L",
        "B",
        "OBJC_ROWC",
        "OBJC_COLC",
        "ROWVDEG",
        "COLVDEG",
    ]
    work = df[["ROW_INDEX", "CLASS", "OBJID", "BESTOBJID"] + coordinate_columns + feature_columns].copy()
    work = work.dropna(subset=feature_columns)
    if work.empty:
        raise ValueError("No hay filas completas para construir el embedding.")
    work = add_science_groups(work)

    embedding = standardize_matrix(work, feature_columns)
    coordinates, explained = pca_2d(embedding)

    points = []
    for pos, (_, row) in enumerate(work.iterrows()):
        vector = {
            feature: _safe_float(embedding[pos, feature_pos])
            for feature_pos, feature in enumerate(feature_columns[:max_vector_values])
        }
        raw_features = {
            feature: _safe_float(row[feature])
            for feature in feature_columns[:max_vector_values]
        }
        points.append(
            {
                "id": int(row["ROW_INDEX"]),
                "class": row["CLASS"],
                "redshiftGroup": row["REDSHIFT_GROUP"],
                "groupId": row["GROUP_ID"],
                "objid": str(row["OBJID"]),
                "bestobjid": str(row["BESTOBJID"]),
                "ra": _safe_float(row["RA"]),
                "dec": _safe_float(row["DEC"]),
                "raerr": _safe_float(row["RAERR"]),
                "decerr": _safe_float(row["DECERR"]),
                "cx": _safe_float(row["CX"]),
                "cy": _safe_float(row["CY"]),
                "cz": _safe_float(row["CZ"]),
                "galactic_l": _safe_float(row["L"]),
                "galactic_b": _safe_float(row["B"]),
                "object_row": _safe_float(row["OBJC_ROWC"]),
                "object_col": _safe_float(row["OBJC_COLC"]),
                "row_motion": _safe_float(row["ROWVDEG"]),
                "col_motion": _safe_float(row["COLVDEG"]),
                "redshift": _safe_float(row["Z"]),
                "shape": {
                    "petrotheta": _safe_float(row["PETROTHETA_r"]),
                    "m_e1": _safe_float(row["M_E1_r"]),
                    "m_e2": _safe_float(row["M_E2_r"]),
                    "fracdev": _safe_float(row["FRACDEV_r"]),
                    "ellipticity": _safe_float(row["ELLIPTICITY_R"]),
                },
                "x": _safe_float(coordinates[pos, 0]),
                "y": _safe_float(coordinates[pos, 1]),
                "embedding": vector,
                "raw": raw_features,
            }
        )

    return {
        "points": points,
        "features": feature_columns,
        "projection": {
            "method": "PCA por SVD sobre embedding estandarizado",
            "pc1_explained": _safe_float(explained[0]) if len(explained) else None,
            "pc2_explained": _safe_float(explained[1]) if len(explained) > 1 else None,
            "rows_used": len(points),
            "dimensions": len(feature_columns),
        },
    }
