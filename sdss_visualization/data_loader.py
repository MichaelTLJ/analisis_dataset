from astropy.io import fits
import numpy as np
import pandas as pd

from .config import BANDS, CLASS_ORDER, PHOTO_FITS, SPEC_FITS, RANDOM_SEED


def _sample_indices_by_class(spec_data, max_per_class):
    rng = np.random.default_rng(RANDOM_SEED)
    classes = np.char.upper(np.char.strip(spec_data["CLASS"].astype(str)))
    selected = []
    counts = {}

    for class_name in CLASS_ORDER:
        class_indices = np.where(classes == class_name)[0]
        counts[class_name] = int(len(class_indices))
        take = min(max_per_class, len(class_indices))
        if take:
            selected.append(rng.choice(class_indices, size=take, replace=False))

    if not selected:
        raise ValueError("No se encontraron clases STAR, GALAXY o QSO en specObj.")

    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return indices, counts


def _scalar_col(data, name, indices):
    return np.asarray(data[name][indices], dtype=np.float64)


def _vector_cols(data, name, indices):
    values = np.asarray(data[name][indices], dtype=np.float64)
    return {f"{name}_{band}": values[:, pos] for pos, band in enumerate(BANDS)}


def _clean_values(df):
    non_numeric_cols = {"CLASS", "OBJID", "BESTOBJID"}
    numeric_cols = [col for col in df.columns if col not in non_numeric_cols]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] <= -9000, col] = np.nan
        if "MAG_" in col:
            df.loc[df[col] <= 0, col] = np.nan
        if col.startswith(("AIRMASS_", "PETROTHETA_")):
            df.loc[df[col] <= 0, col] = np.nan
        df.loc[~np.isfinite(df[col]), col] = np.nan
    return df


def load_sdss_sample(max_per_class):
    with fits.open(PHOTO_FITS, memmap=True) as photo_hdul, fits.open(SPEC_FITS, memmap=True) as spec_hdul:
        photo = photo_hdul[1].data
        spec = spec_hdul[1].data
        indices, class_counts = _sample_indices_by_class(spec, max_per_class)

        rows = {
            "ROW_INDEX": indices.astype(int),
            "CLASS": np.char.upper(np.char.strip(spec["CLASS"][indices].astype(str))),
            "OBJID": photo["OBJID"][indices].astype(str),
            "BESTOBJID": spec["BESTOBJID"][indices].astype(str),
            "RA": _scalar_col(photo, "RA", indices),
            "DEC": _scalar_col(photo, "DEC", indices),
            "RAERR": _scalar_col(photo, "RAERR", indices),
            "DECERR": _scalar_col(photo, "DECERR", indices),
            "CX": _scalar_col(photo, "CX", indices),
            "CY": _scalar_col(photo, "CY", indices),
            "CZ": _scalar_col(photo, "CZ", indices),
            "L": _scalar_col(photo, "L", indices),
            "B": _scalar_col(photo, "B", indices),
            "OBJC_ROWC": _scalar_col(photo, "OBJC_ROWC", indices),
            "OBJC_COLC": _scalar_col(photo, "OBJC_COLC", indices),
            "ROWVDEG": _scalar_col(photo, "ROWVDEG", indices),
            "COLVDEG": _scalar_col(photo, "COLVDEG", indices),
            "Z": _scalar_col(spec, "Z", indices),
            "SN_MEDIAN_ALL": _scalar_col(spec, "SN_MEDIAN_ALL", indices),
        }

        for name in ["OFFSETRA", "OFFSETDEC", "ROWC", "COLC"]:
            rows.update(_vector_cols(photo, name, indices))

        for name in [
            "PSFMAG",
            "CMODELMAG",
            "PSFFLUX",
            "CMODELFLUX",
            "PETROTHETA",
            "M_E1",
            "M_E2",
            "FRACDEV",
            "AIRMASS",
            "EXTINCTION",
        ]:
            rows.update(_vector_cols(photo, name, indices))

    df = _clean_values(pd.DataFrame(rows))

    for prefix in ["PSFMAG", "CMODELMAG"]:
        df[f"{prefix}_ug"] = df[f"{prefix}_u"] - df[f"{prefix}_g"]
        df[f"{prefix}_gr"] = df[f"{prefix}_g"] - df[f"{prefix}_r"]
        df[f"{prefix}_ri"] = df[f"{prefix}_r"] - df[f"{prefix}_i"]
        df[f"{prefix}_iz"] = df[f"{prefix}_i"] - df[f"{prefix}_z"]

    df["ELLIPTICITY_R"] = np.sqrt(df["M_E1_r"] ** 2 + df["M_E2_r"] ** 2)
    return df, class_counts
