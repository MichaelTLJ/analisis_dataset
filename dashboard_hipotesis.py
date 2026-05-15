from pathlib import Path
import json

from astropy.io import fits
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DASHBOARD_PATH = OUTPUT_DIR / "dashboard.html"
HYPOTHESIS_DIR = OUTPUT_DIR / "hipotesis_csv"
ANALYSIS_DIR = OUTPUT_DIR / "analisis"

PHOTO_FITS = BASE_DIR / "data" / "photoPosPlate-dr17.fits"
SPEC_FITS = BASE_DIR / "data" / "specObj-dr17.fits"

CLASS_ORDER = ["STAR", "GALAXY", "QSO"]
CLASS_COLORS = {"STAR": "#2878b5", "GALAXY": "#2f9e44", "QSO": "#d64545"}
BANDS = ["u", "g", "r", "i", "z"]
MAX_PER_CLASS = 4500
RANDOM_SEED = 42


STEP_TABLES = {
    "Paso 0": [
        ("Indice de analisis", ANALYSIS_DIR / "indice_analisis.csv"),
    ],
    "Paso 1": [
        ("Resumen PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "tablas" / "resumen_general.csv"),
        ("Resumen SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "resumen_general.csv"),
        ("Metadata expandida PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "tablas" / "metadata_expandida.csv"),
        ("Metadata expandida SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "metadata_expandida.csv"),
    ],
    "Paso 2": [
        ("Tipos y rangos PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "tablas" / "tipos_y_rangos.csv"),
        ("Tipos y rangos SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "tipos_y_rangos.csv"),
        ("Nulos PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "tablas" / "nulos.csv"),
        ("Nulos SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "nulos.csv"),
        ("Duplicados PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "tablas" / "duplicados.csv"),
        ("Duplicados SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "duplicados.csv"),
    ],
    "Paso 3": [
        ("Estadisticas PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "tablas" / "estadisticas.csv"),
        ("Estadisticas SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "estadisticas.csv"),
        ("Histogramas PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "graficas" / "histogramas.csv"),
        ("Histogramas SpecObj", ANALYSIS_DIR / "specObj-dr17" / "graficas" / "histogramas.csv"),
        ("Barras categoricas SpecObj", ANALYSIS_DIR / "specObj-dr17" / "graficas" / "barras_categoricas.csv"),
    ],
    "Paso 4": [
        ("Outliers PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "tablas" / "outliers.csv"),
        ("Outliers SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "outliers.csv"),
        ("Top outliers PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "tablas" / "top_outliers.csv"),
        ("Top outliers SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "top_outliers.csv"),
    ],
    "Paso 5": [
        ("Correlacion PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "tablas" / "correlacion.csv"),
        ("Correlacion SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "correlacion.csv"),
        ("Top correlaciones PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "tablas" / "top_correlaciones.csv"),
        ("Top correlaciones SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "top_correlaciones.csv"),
        ("Scatterplots PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "graficas" / "scatterplots.csv"),
        ("Scatterplots SpecObj", ANALYSIS_DIR / "specObj-dr17" / "graficas" / "scatterplots.csv"),
    ],
    "Paso 6": [
        ("Covarianza PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "tablas" / "covarianza.csv"),
        ("Covarianza SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "covarianza.csv"),
        ("Heatmap PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "graficas" / "heatmap_correlacion.csv"),
        ("Heatmap SpecObj", ANALYSIS_DIR / "specObj-dr17" / "graficas" / "heatmap_correlacion.csv"),
    ],
    "Paso 7": [
        ("Tipos y rangos PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "tablas" / "tipos_y_rangos.csv"),
        ("Tipos y rangos SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "tipos_y_rangos.csv"),
    ],
    "Paso 8": [
        ("Distribucion de clases SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "distribucion_clases.csv"),
        ("Value counts CLASS", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "value_counts_top_CLASS.csv"),
    ],
    "Paso 9": [
        ("Errores estadisticas PhotoObj", ANALYSIS_DIR / "photoPosPlate-dr17" / "tablas" / "errores_estadisticas.csv"),
        ("Errores estadisticas SpecObj", ANALYSIS_DIR / "specObj-dr17" / "tablas" / "errores_estadisticas.csv"),
    ],
    "Paso 10": [
        ("Resultados H1", HYPOTHESIS_DIR / "h1_redshift_modelos.csv"),
        ("Resultados H2", HYPOTHESIS_DIR / "h2_accuracy_grupos.csv"),
        ("Resultados H3", HYPOTHESIS_DIR / "h3_correlaciones_condicionadas.csv"),
        ("Resultados H4", HYPOTHESIS_DIR / "h4_correlaciones_sesgo.csv"),
        ("Resultados H5", HYPOTHESIS_DIR / "h5_histogramas_colores.csv"),
    ],
    "Paso 11": [
        ("Indice CSV hipotesis", HYPOTHESIS_DIR / "indice_resultados_hipotesis.csv"),
    ],
}


def safe_float(value):
    if value is None or pd.isna(value) or not np.isfinite(value):
        return None
    return float(value)


def sample_indices_by_class(spec_data):
    rng = np.random.default_rng(RANDOM_SEED)
    classes = np.char.upper(np.char.strip(spec_data["CLASS"].astype(str)))
    indices = []
    counts = {}

    for cls in CLASS_ORDER:
        cls_idx = np.where(classes == cls)[0]
        counts[cls] = int(len(cls_idx))
        take = min(MAX_PER_CLASS, len(cls_idx))
        if take > 0:
            indices.append(rng.choice(cls_idx, size=take, replace=False))

    if not indices:
        raise ValueError("No se encontraron clases STAR/GALAXY/QSO en specObj.")

    idx = np.concatenate(indices)
    rng.shuffle(idx)
    return idx, counts


def vector_col(data, name, idx):
    arr = np.asarray(data[name][idx], dtype=np.float64)
    return {f"{name}_{band}": arr[:, pos] for pos, band in enumerate(BANDS)}


def scalar_col(data, name, idx):
    return np.asarray(data[name][idx], dtype=np.float64)


def clean_values(df):
    numeric_cols = [col for col in df.columns if col != "CLASS"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] <= -9000, col] = np.nan
        if "MAG_" in col:
            df.loc[df[col] <= 0, col] = np.nan
        if col.startswith("AIRMASS_") or col.startswith("PETROTHETA_"):
            df.loc[df[col] <= 0, col] = np.nan
        df.loc[~np.isfinite(df[col]), col] = np.nan
    return df


def load_project_sample():
    with fits.open(PHOTO_FITS, memmap=True) as photo_hdul, fits.open(SPEC_FITS, memmap=True) as spec_hdul:
        photo = photo_hdul[1].data
        spec = spec_hdul[1].data

        idx, class_counts = sample_indices_by_class(spec)
        rows = {
            "CLASS": np.char.upper(np.char.strip(spec["CLASS"][idx].astype(str))),
            "Z": scalar_col(spec, "Z", idx),
            "SN_MEDIAN_ALL": scalar_col(spec, "SN_MEDIAN_ALL", idx),
        }

        for name in ["PSFMAG", "CMODELMAG", "MODELMAG", "PETROMAG", "PSFFLUX", "CMODELFLUX", "MODELFLUX", "PETROFLUX", "DEVFLUX", "EXPFLUX", "PETROTHETA", "M_E1", "M_E2", "FRACDEV", "AIRMASS", "EXTINCTION"]:
            rows.update(vector_col(photo, name, idx))

        df = pd.DataFrame(rows)
        df = clean_values(df)

        for prefix in ["PSFMAG", "CMODELMAG", "MODELMAG", "PETROMAG"]:
            df[f"{prefix}_ug"] = df[f"{prefix}_u"] - df[f"{prefix}_g"]
            df[f"{prefix}_gr"] = df[f"{prefix}_g"] - df[f"{prefix}_r"]
            df[f"{prefix}_ri"] = df[f"{prefix}_r"] - df[f"{prefix}_i"]
            df[f"{prefix}_iz"] = df[f"{prefix}_i"] - df[f"{prefix}_z"]

        df["ELLIPTICITY_R"] = np.sqrt(df["M_E1_r"] ** 2 + df["M_E2_r"] ** 2)
        df["PHOTO_ERROR_FLAG"] = df[[f"PSFMAG_{b}" for b in BANDS]].isna().any(axis=1).astype(int)

        match_check = pd.DataFrame({
            "photo_OBJID": photo["OBJID"][idx[:20]].astype(str),
            "spec_BESTOBJID": spec["BESTOBJID"][idx[:20]].astype(str),
        })
        aligned = bool((match_check["photo_OBJID"] == match_check["spec_BESTOBJID"]).mean() >= 0.95)

    return df, class_counts, aligned


def histogram(df, feature, by_class=True, bins=34):
    payload = {}
    values_all = df[feature].dropna()
    if values_all.empty:
        return payload
    lo, hi = values_all.quantile(0.01), values_all.quantile(0.99)
    if lo == hi:
        lo, hi = values_all.min(), values_all.max()
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5

    groups = CLASS_ORDER if by_class else ["ALL"]
    for group in groups:
        values = df.loc[df["CLASS"] == group, feature].dropna() if by_class else values_all
        if values.empty:
            continue
        counts, edges = np.histogram(values.clip(lo, hi), bins=bins, range=(lo, hi))
        payload[group] = {
            "edges": [float(x) for x in edges],
            "counts": [int(x) for x in counts],
        }
    return payload


def corr(a, b):
    pair = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(pair) < 3:
        return None
    return safe_float(pair["a"].corr(pair["b"]))


def eta_squared(df, feature, target="CLASS"):
    pair = df[[target, feature]].dropna()
    if pair.empty:
        return None
    overall = pair[feature].mean()
    ss_total = ((pair[feature] - overall) ** 2).sum()
    if ss_total == 0:
        return None
    ss_between = 0
    for _, group in pair.groupby(target):
        ss_between += len(group) * (group[feature].mean() - overall) ** 2
    return safe_float(ss_between / ss_total)


def linear_regression_r2(df, features, target="Z"):
    work = df[features + [target]].dropna()
    if len(work) < 50:
        return None
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(work))
    split = int(len(work) * 0.75)
    train = work.iloc[order[:split]]
    test = work.iloc[order[split:]]
    x_train = train[features].to_numpy(dtype=float)
    y_train = train[target].to_numpy(dtype=float)
    x_test = test[features].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)
    mean = x_train.mean(axis=0)
    std = np.where(x_train.std(axis=0) == 0, 1, x_train.std(axis=0))
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_test = np.c_[np.ones(len(x_test)), x_test]
    beta = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    pred = x_test @ beta
    ss_res = np.sum((y_test - pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    return safe_float(1 - ss_res / ss_tot) if ss_tot else None


def centroid_accuracy(df, features):
    work = df[["CLASS"] + features].dropna()
    if len(work) < 50:
        return None
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(work))
    split = int(len(work) * 0.75)
    train = work.iloc[order[:split]]
    test = work.iloc[order[split:]]
    mean = train[features].mean()
    std = train[features].std().replace(0, 1)
    train_x = (train[features] - mean) / std
    test_x = (test[features] - mean) / std
    centroids = {cls: train_x[train["CLASS"] == cls].mean().to_numpy(dtype=float) for cls in CLASS_ORDER if (train["CLASS"] == cls).any()}
    correct = 0
    for (_, row), true_cls in zip(test_x.iterrows(), test["CLASS"]):
        x = row.to_numpy(dtype=float)
        pred = min(centroids, key=lambda cls: np.linalg.norm(x - centroids[cls]))
        correct += int(pred == true_cls)
    return safe_float(correct / len(test))


def pca_projection(df, features, max_points=5000):
    work = df[["CLASS"] + features].dropna()
    if len(work) > max_points:
        work = work.sample(max_points, random_state=RANDOM_SEED)
    if work.empty:
        return []
    x = work[features].to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) == 0, 1, x.std(axis=0))
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:2].T
    return [{"CLASS": cls, "PC1": safe_float(proj[i, 0]), "PC2": safe_float(proj[i, 1])} for i, cls in enumerate(work["CLASS"])]


def binned_quality(df, feature, bins=8):
    needed = list(dict.fromkeys([feature, "PHOTO_ERROR_FLAG", "SN_MEDIAN_ALL", "Z"]))
    work = df[needed].dropna()
    if work.empty:
        return []
    work["bin"] = pd.qcut(work[feature], q=bins, duplicates="drop")
    rows = []
    for interval, group in work.groupby("bin", observed=False):
        rows.append({
            "bin": str(interval),
            "x": safe_float(group[feature].median()),
            "error_rate": safe_float(group["PHOTO_ERROR_FLAG"].mean()),
            "sn_median": safe_float(group["SN_MEDIAN_ALL"].median()),
            "z_median": safe_float(group["Z"].median()),
            "count": int(len(group)),
        })
    return rows


def relpath(path):
    try:
        return str(path.relative_to(OUTPUT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def table_preview(path, max_rows=8, max_cols=8):
    if not path.exists():
        return {
            "path": relpath(path),
            "exists": False,
            "columns": [],
            "rows": [],
            "total_rows": 0,
            "total_cols": 0,
        }
    try:
        preview = pd.read_csv(path, nrows=max_rows)
        total_rows = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
    except Exception:
        return {
            "path": relpath(path),
            "exists": False,
            "columns": [],
            "rows": [],
            "total_rows": 0,
            "total_cols": 0,
        }
    preview = preview.iloc[:, :max_cols].where(pd.notnull(preview), None)
    return {
        "path": relpath(path),
        "exists": True,
        "columns": [str(col) for col in preview.columns],
        "rows": preview.to_dict(orient="records"),
        "total_rows": int(max(total_rows, 0)),
        "total_cols": int(len(pd.read_csv(path, nrows=0).columns)),
    }


def build_step_tables():
    payload = {}
    for step, tables in STEP_TABLES.items():
        payload[step] = [
            {
                "title": title,
                **table_preview(path),
            }
            for title, path in tables
        ]
    return payload


def flatten_histograms(histograms):
    rows = []
    for feature, groups in histograms.items():
        for cls, info in groups.items():
            edges = info["edges"]
            counts = info["counts"]
            for i, count in enumerate(counts):
                rows.append({
                    "feature": feature,
                    "class": cls,
                    "bin_start": edges[i],
                    "bin_end": edges[i + 1],
                    "bin_center": (edges[i] + edges[i + 1]) / 2,
                    "count": count,
                })
    return pd.DataFrame(rows)


def export_hypothesis_csv(df, payload):
    HYPOTHESIS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = payload["metrics"]
    exported = []

    def save(name, frame, title, hypothesis, description):
        path = HYPOTHESIS_DIR / name
        frame.to_csv(path, index=False)
        exported.append({
            "hipotesis": hypothesis,
            "titulo": title,
            "archivo": relpath(path),
            "filas": int(len(frame)),
            "descripcion": description,
        })

    def sample_frame(frame, max_rows=8000):
        if len(frame) <= max_rows:
            return frame
        return frame.sample(max_rows, random_state=RANDOM_SEED)

    h1_models = pd.DataFrame([
        {"modelo": "Regresion lineal con colores", "target": "Z", "r2": metrics["photoz_r2_colors"], "features": "PSFMAG/CMODELMAG colores"},
        {"modelo": "Regresion lineal con magnitudes y colores", "target": "Z", "r2": metrics["photoz_r2_mags_colors"], "features": "PSFMAG, CMODELMAG y colores"},
    ])
    save("h1_redshift_modelos.csv", h1_models, "R2 de modelos exploratorios de redshift", "H1", "Resultados numericos de la hipotesis de redshift fotometrico.")
    save(
        "h1_scatter_color_z.csv",
        sample_frame(df[["CLASS", "Z", "PSFMAG_ug", "PSFMAG_gr", "PSFMAG_ri", "PSFMAG_iz", "CMODELMAG_ug", "CMODELMAG_gr"]].dropna()),
        "Datos para color vs redshift",
        "H1",
        "Puntos limpios usados para graficar relaciones entre colores fotometricos y Z.",
    )

    h2_acc = pd.DataFrame([
        {"grupo": "luz", "accuracy_centroides": metrics["class_acc_light"], "features": "*MAG y *FLUX"},
        {"grupo": "morfologia", "accuracy_centroides": metrics["class_acc_morph"], "features": "PETROTHETA, M_E1, M_E2, FRACDEV"},
        {"grupo": "colores", "accuracy_centroides": metrics["class_acc_colors"], "features": "indices de color"},
        {"grupo": "multivariado", "accuracy_centroides": metrics["class_acc_multi"], "features": "colores, Z, magnitudes y morfologia"},
    ])
    save("h2_accuracy_grupos.csv", h2_acc, "Comparacion luz vs morfologia", "H2", "Accuracy exploratoria por centroides para cada familia de variables.")
    save("h2_ranking_variables.csv", pd.DataFrame(payload["ranking"]), "Ranking de variables", "H2", "Eta cuadrado contra CLASS y correlacion con Z.")
    save("h2_pca_projection.csv", pd.DataFrame(payload["pca"]), "Proyeccion PCA", "H2", "Componentes principales para visualizar separabilidad multivariada.")

    corr_rows = []
    for key, value in metrics["corr_global"].items():
        corr_rows.append({"grupo": "global", "par": key, "correlacion": value})
    for key, value in metrics["corr_conditioned"].items():
        corr_rows.append({"grupo": "condicionado", "par": key, "correlacion": value})
    save("h3_correlaciones_condicionadas.csv", pd.DataFrame(corr_rows), "Correlaciones globales y condicionadas", "H3", "Contraste de colinealidad global vs subpoblaciones complejas.")
    save(
        "h3_scatter_flujos.csv",
        sample_frame(df[["CLASS", "PETROFLUX_r", "DEVFLUX_r", "EXPFLUX_r", "CMODELFLUX_r", "PETROTHETA_r", "ELLIPTICITY_R"]].dropna()),
        "Datos de flujos por morfologia",
        "H3",
        "Datos para comparar metodos de flujo luminoso y condicionamiento morfologico.",
    )

    quality_rows = []
    for key, rows in payload["quality"].items():
        for row in rows:
            quality_rows.append({"variable": key, **row})
    save("h4_calidad_bins.csv", pd.DataFrame(quality_rows), "Calidad por bins", "H4", "Tasa de error fisico, SN y Z por bins de AIRMASS, EXTINCTION y redshift.")
    h4_corr = pd.DataFrame([
        {"variable": "AIRMASS_r", "correlacion_error_fisico": metrics["quality_corr_airmass_error"]},
        {"variable": "EXTINCTION_r", "correlacion_error_fisico": metrics["quality_corr_extinction_error"]},
        {"variable": "Z", "correlacion_error_fisico": metrics["quality_corr_z_error"]},
        {"variable": "SN_MEDIAN_ALL", "correlacion_error_fisico": metrics["quality_corr_sn_error"]},
    ])
    save("h4_correlaciones_sesgo.csv", h4_corr, "Correlaciones de sesgo/calidad", "H4", "Dependencia estadistica de errores fisicos con condiciones instrumentales.")

    save("h5_histogramas_colores.csv", flatten_histograms(payload["histograms"]), "Histogramas por clase", "H5", "Bins de magnitudes, flujos y colores usados por el dashboard.")
    save(
        "h5_color_color_scatter.csv",
        sample_frame(df[["CLASS", "PSFMAG_r", "PSFFLUX_r", "PSFMAG_ug", "PSFMAG_gr", "PSFMAG_ri", "PSFMAG_iz"]].dropna()),
        "Diagrama color-color",
        "H5",
        "Puntos limpios para evaluar separabilidad por indices de color.",
    )

    clean_cols = ["CLASS", "Z", "SN_MEDIAN_ALL", "PHOTO_ERROR_FLAG", "PSFMAG_u", "PSFMAG_g", "PSFMAG_r", "PSFMAG_i", "PSFMAG_z", "PSFMAG_ug", "PSFMAG_gr", "PETROTHETA_r", "ELLIPTICITY_R", "AIRMASS_r", "EXTINCTION_r"]
    save("muestra_limpia_dashboard.csv", sample_frame(df[clean_cols], 10000), "Muestra limpia del dashboard", "General", "Muestra usada para graficas interactivas y resultados de hipotesis.")

    index = pd.DataFrame(exported)
    index_path = HYPOTHESIS_DIR / "indice_resultados_hipotesis.csv"
    index.to_csv(index_path, index=False)
    return index.to_dict(orient="records")


def build_payload():
    df, class_counts, aligned = load_project_sample()

    color_features = ["PSFMAG_ug", "PSFMAG_gr", "PSFMAG_ri", "PSFMAG_iz", "CMODELMAG_ug", "CMODELMAG_gr", "CMODELMAG_ri"]
    mag_features = [f"PSFMAG_{b}" for b in BANDS] + [f"CMODELMAG_{b}" for b in BANDS]
    flux_features = [f"PSFFLUX_{b}" for b in BANDS] + [f"CMODELFLUX_{b}" for b in BANDS] + [f"PETROFLUX_{b}" for b in BANDS]
    morph_features = [f"PETROTHETA_{b}" for b in BANDS] + [f"M_E1_{b}" for b in BANDS] + [f"M_E2_{b}" for b in BANDS] + [f"FRACDEV_{b}" for b in BANDS]
    multi_features = color_features + ["Z"] + mag_features[:5] + ["ELLIPTICITY_R", "PETROTHETA_r"]

    corr_global = {
        "PETRO_vs_DEV_r": corr(df["PETROFLUX_r"], df["DEVFLUX_r"]),
        "PETRO_vs_EXP_r": corr(df["PETROFLUX_r"], df["EXPFLUX_r"]),
        "PETRO_vs_CMODEL_r": corr(df["PETROFLUX_r"], df["CMODELFLUX_r"]),
    }
    extended = df[df["PETROTHETA_r"] >= df["PETROTHETA_r"].quantile(0.80)]
    elliptical = df[df["ELLIPTICITY_R"] >= df["ELLIPTICITY_R"].quantile(0.80)]
    corr_conditioned = {
        "extended_PETRO_vs_DEV_r": corr(extended["PETROFLUX_r"], extended["DEVFLUX_r"]),
        "extended_PETRO_vs_EXP_r": corr(extended["PETROFLUX_r"], extended["EXPFLUX_r"]),
        "elliptical_PETRO_vs_DEV_r": corr(elliptical["PETROFLUX_r"], elliptical["DEVFLUX_r"]),
        "elliptical_PETRO_vs_EXP_r": corr(elliptical["PETROFLUX_r"], elliptical["EXPFLUX_r"]),
    }

    ranking = []
    for feature in color_features + mag_features + flux_features[:8] + morph_features[:12] + ["Z", "SN_MEDIAN_ALL", "AIRMASS_r", "EXTINCTION_r"]:
        ranking.append({
            "feature": feature,
            "eta2_class": eta_squared(df, feature),
            "corr_z": corr(df[feature], df["Z"]) if feature != "Z" else 1.0,
        })
    ranking.sort(key=lambda row: -1 if row["eta2_class"] is None else row["eta2_class"], reverse=True)

    scatter_cols = ["CLASS", "Z", "PSFMAG_ug", "PSFMAG_gr", "PSFMAG_ri", "PSFMAG_r", "PSFFLUX_r", "CMODELFLUX_r", "PETROFLUX_r", "DEVFLUX_r", "EXPFLUX_r", "PETROTHETA_r", "ELLIPTICITY_R", "AIRMASS_r", "EXTINCTION_r", "SN_MEDIAN_ALL", "PHOTO_ERROR_FLAG"]
    scatter_df = df[scatter_cols].dropna(subset=["CLASS"]).sample(min(len(df), 8000), random_state=RANDOM_SEED)

    payload = {
        "source": {
            "photo": str(PHOTO_FITS),
            "spec": str(SPEC_FITS),
            "aligned_by_row": aligned,
        },
        "sampleRows": int(len(df)),
        "classCounts": class_counts,
        "classColors": CLASS_COLORS,
        "scatter": scatter_df.where(pd.notnull(scatter_df), None).to_dict(orient="records"),
        "pca": pca_projection(df, multi_features),
        "histograms": {
            "PSFMAG_r": histogram(df, "PSFMAG_r"),
            "PSFFLUX_r": histogram(df, "PSFFLUX_r"),
            "PSFMAG_ug": histogram(df, "PSFMAG_ug"),
            "PSFMAG_gr": histogram(df, "PSFMAG_gr"),
            "Z": histogram(df, "Z"),
        },
        "metrics": {
            "photoz_r2_colors": linear_regression_r2(df, color_features),
            "photoz_r2_mags_colors": linear_regression_r2(df, mag_features + color_features),
            "class_acc_light": centroid_accuracy(df, mag_features + flux_features),
            "class_acc_morph": centroid_accuracy(df, morph_features),
            "class_acc_colors": centroid_accuracy(df, color_features),
            "class_acc_multi": centroid_accuracy(df, multi_features),
            "corr_global": corr_global,
            "corr_conditioned": corr_conditioned,
            "quality_corr_airmass_error": corr(df["AIRMASS_r"], df["PHOTO_ERROR_FLAG"]),
            "quality_corr_extinction_error": corr(df["EXTINCTION_r"], df["PHOTO_ERROR_FLAG"]),
            "quality_corr_z_error": corr(df["Z"], df["PHOTO_ERROR_FLAG"]),
            "quality_corr_sn_error": corr(df["SN_MEDIAN_ALL"], df["PHOTO_ERROR_FLAG"]),
        },
        "quality": {
            "airmass": binned_quality(df, "AIRMASS_r"),
            "extinction": binned_quality(df, "EXTINCTION_r"),
            "redshift": binned_quality(df, "Z"),
        },
        "ranking": ranking[:35],
        "steps": [
            ["Paso 0", "Contexto y preparación", "Problema: evaluar si fotometría, morfología, calidad instrumental y redshift explican clasificación y mediciones espectroscópicas. Fuente: FITS locales SDSS DR17 del proyecto."],
            ["Paso 1", "Estructura", "Cada fila representa un objeto astronómico alineado entre photoPosPlate y specObj. Se combinan columnas fotométricas, morfológicas, calidad, CLASS y Z."],
            ["Paso 2", "Calidad y limpieza", "El data wrangling ya fue realizado. En esta etapa se tratan nulos físicos como -9999, duplicados, rangos, tipos y variables de calidad."],
            ["Paso 3", "Distribuciones univariadas", "Histogramas de magnitudes y flujos muestran solapamiento entre clases y colas largas."],
            ["Paso 4", "Outliers", "Los outliers se analizan como posibles objetos raros o fallas instrumentales; no se eliminan sin justificación astrofísica."],
            ["Paso 5", "Relaciones bivariadas", "Scatterplots y correlaciones evalúan color vs Z, métodos de flujo y sesgo de calidad."],
            ["Paso 6", "Relaciones multivariadas", "PCA y comparación de grupos de variables muestran estructuras no evidentes en una sola variable."],
            ["Paso 7", "Temporalidad", "El dataset contiene MJD/TAI en los análisis procesados; aquí se usa como contexto observacional, no como serie temporal principal."],
            ["Paso 8", "Balance", "Se revisa la representatividad de STAR, GALAXY y QSO para clasificación supervisada."],
            ["Paso 9", "Preguntas emergentes", "¿Qué errores son sesgo instrumental? ¿Qué variables son redundantes solo globalmente? ¿Qué features conviene conservar para modelos?"],
            ["Paso 10", "Conclusiones preliminares", "Se responde cada hipótesis con evidencia visual y métricas exploratorias."],
            ["Paso 11", "Comunicación", "El dashboard presenta más de 5 visualizaciones con título, descripción e interpretación."],
        ],
    }
    payload["csvExports"] = export_hypothesis_csv(df, payload)
    payload["stepTables"] = build_step_tables()
    return payload


def build_html(payload):
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard SDSS - 5 Hipótesis</title>
<style>
body{{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f6f7fb;color:#182033;line-height:1.5}}
header{{background:#fff;border-bottom:1px solid #dce2ec;padding:24px clamp(14px,4vw,56px);position:sticky;top:0;z-index:5}}
h1{{margin:0;font-size:clamp(25px,4vw,42px)}} h2{{margin:0 0 8px}} h3{{margin:14px 0 6px}}
main{{max-width:1500px;margin:auto;padding:18px clamp(10px,3vw,36px) 60px}}
.tabs{{display:flex;gap:8px;overflow:auto;margin-top:16px}}button,.chip{{border:1px solid #cfd8ea;background:#fff;border-radius:7px;padding:9px 11px;cursor:pointer}}
.tab.active{{background:#3156d4;color:#fff}}.panel{{background:#fff;border:1px solid #dce2ec;border-radius:10px;padding:16px;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(320px,.85fr);gap:16px}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.metric{{background:#fff;border:1px solid #dce2ec;border-radius:10px;padding:13px}}.metric span{{color:#667085;font-size:13px;display:block}}.metric strong{{font-size:24px;display:block}}
.filters{{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}}select{{padding:8px;border:1px solid #cfd8ea;border-radius:7px;background:#fff}}
svg{{width:100%;height:min(62vh,620px);min-height:380px;background:#fbfcff;border:1px solid #dce2ec;border-radius:8px}}
.hyp{{display:none}}.hyp.active{{display:block}}.finding{{background:#fff7df;border-left:5px solid #9b6500}}.step-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}}
.step{{border:1px solid #dce2ec;border-radius:8px;padding:12px;background:#fbfcff}}.legend{{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;color:#667085}}.dot{{width:11px;height:11px;border-radius:50%;display:inline-block}}
.table{{max-height:360px;overflow:auto;border:1px solid #dce2ec;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #edf0f5;text-align:left}}th{{background:#f1f4fa}}
.export-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}}.csv-card{{border:1px solid #dce2ec;border-radius:8px;padding:10px;background:#fbfcff}}.csv-card a{{font-weight:700;color:#3156d4;text-decoration:none}}.mini{{font-size:12px;color:#667085}}.step-table{{margin-top:10px;border-top:1px solid #dce2ec;padding-top:10px}}.step-table h4{{margin:6px 0}}.step details{{margin-top:10px}}.step summary{{cursor:pointer;color:#3156d4;font-weight:700}}
.tooltip{{position:fixed;pointer-events:none;background:#182033;color:white;padding:7px 9px;border-radius:6px;font-size:12px;opacity:0;z-index:20}}
@media(max-width:950px){{.grid{{grid-template-columns:1fr}}svg{{height:460px}}}}
</style>
</head>
<body>
<header>
<h1>Dashboard EDA SDSS: 5 hipótesis del proyecto</h1>
<p>Fuente exclusiva del proyecto: <code>data/photoPosPlate-dr17.fits</code>, <code>data/specObj-dr17.fits</code> y análisis en <code>output/analisis</code>. El data wrangling ya está realizado; aquí se enfatiza limpieza, evidencia visual y conclusiones.</p>
<div class="tabs">
<button class="tab active" data-tab="h1">H1 Redshift fotométrico</button>
<button class="tab" data-tab="h2">H2 Luz vs morfología</button>
<button class="tab" data-tab="h3">H3 Correlación condicionada</button>
<button class="tab" data-tab="h4">H4 Sesgo/calidad</button>
<button class="tab" data-tab="h5">H5 Feature engineering</button>
<button class="tab" data-tab="steps">Pasos 0-11</button>
</div>
</header>
<main>
<section class="panel"><div class="metrics" id="metrics"></div><div class="filters"><label>Clases visibles <span id="classToggles"></span></label><label>X <select id="xFeature"></select></label><label>Y <select id="yFeature"></select></label></div></section>
<section class="panel"><h2>Resultados exportados por hipótesis</h2><p>Las gráficas y métricas principales también quedan guardadas como CSV para documentar el informe o reutilizarlas en otro notebook.</p><div class="export-grid" id="csvExports"></div></section>
<section id="h1" class="hyp active"><div class="grid"><div class="panel"><h2>H1. Estimación de Redshift Fotométrico</h2><p>¿Se puede aproximar Z usando magnitudes y colores, prescindiendo de espectroscopía?</p><svg id="plotH1"></svg><div class="legend" id="legendH1"></div></div><aside><div class="panel finding" id="findingH1"></div><div class="panel"><h3>Distribución de Z</h3><svg id="histZ"></svg></div></aside></div></section>
<section id="h2" class="hyp"><div class="grid"><div class="panel"><h2>H2. Clasificación Morfológica vs Fotométrica</h2><p>Compara poder predictivo exploratorio de variables de luz contra variables de forma.</p><svg id="plotH2"></svg><div class="legend" id="legendH2"></div></div><aside><div class="panel finding" id="findingH2"></div><div class="panel"><h3>Vista</h3><select id="h2Mode"><option value="scatter">Scatter de variables</option><option value="pca">PCA multivariado</option></select></div><div class="panel"><h3>Ranking de variables</h3><div class="table" id="rankTable"></div></div></aside></div></section>
<section id="h3" class="hyp"><div class="grid"><div class="panel"><h2>H3. Divergencia de algoritmos en morfologías complejas</h2><p>Evalúa si la correlación global entre métodos de flujo se rompe al condicionar por tamaño o elipticidad.</p><svg id="plotH3"></svg></div><aside><div class="panel finding" id="findingH3"></div><div class="panel"><h3>Metodo de flujo comparado</h3><select id="h3Flux"><option value="DEVFLUX_r">DEVFLUX_r</option><option value="EXPFLUX_r">EXPFLUX_r</option><option value="CMODELFLUX_r">CMODELFLUX_r</option></select></div><div class="panel"><h3>Correlaciones globales vs condicionadas</h3><div class="table" id="corrCondTable"></div></div></aside></div></section>
<section id="h4" class="hyp"><div class="grid"><div class="panel"><h2>H4. Sesgo de selección y degradación de calidad</h2><p>Investiga si errores físicos y señal/ruido dependen de AIRMASS, EXTINCTION o redshift.</p><svg id="plotH4"></svg></div><aside><div class="panel finding" id="findingH4"></div><div class="panel"><h3>Variable de calidad</h3><select id="qualityFeature"><option value="airmass">AIRMASS_r</option><option value="extinction">EXTINCTION_r</option><option value="redshift">Z</option></select></div></aside></div></section>
<section id="h5" class="hyp"><div class="grid"><div class="panel"><h2>H5. Separabilidad por ingeniería de características</h2><p>Compara mirar luminosidad cruda vs transformar a índices de color.</p><svg id="plotH5"></svg><div class="legend" id="legendH5"></div></div><aside><div class="panel finding" id="findingH5"></div><div class="panel"><h3>Histograma</h3><select id="histFeature"><option value="PSFMAG_r">PSFMAG_r</option><option value="PSFFLUX_r">PSFFLUX_r</option><option value="PSFMAG_ug">PSFMAG_u - PSFMAG_g</option><option value="PSFMAG_gr">PSFMAG_g - PSFMAG_r</option></select><svg id="histGeneric"></svg></div></aside></div></section>
<section id="steps" class="hyp"><div class="panel"><h2>Paso a paso EDA 0-11</h2><div class="step-grid" id="stepsGrid"></div></div></section>
</main><div class="tooltip" id="tooltip"></div>
<script>
const DATA={data_json};
const classes=["STAR","GALAXY","QSO"]; const colors=DATA.classColors;
const labels={{Z:"redshift Z",PSFMAG_ug:"PSFMAG u-g",PSFMAG_gr:"PSFMAG g-r",PSFMAG_ri:"PSFMAG r-i",PSFMAG_r:"PSFMAG r",PSFFLUX_r:"PSFFLUX r",CMODELFLUX_r:"CMODELFLUX r",PETROFLUX_r:"PETROFLUX r",DEVFLUX_r:"DEVFLUX r",EXPFLUX_r:"EXPFLUX r",PETROTHETA_r:"PETROTHETA r",ELLIPTICITY_R:"Elipticidad r",AIRMASS_r:"AIRMASS r",EXTINCTION_r:"EXTINCTION r",SN_MEDIAN_ALL:"SN median all",PC1:"PC1",PC2:"PC2"}};
const state={{visible:new Set(classes),x:"PSFMAG_ug",y:"PSFMAG_gr",quality:"airmass",hist:"PSFMAG_r",h2Mode:"scatter",h3Flux:"DEVFLUX_r"}};
function fmt(v,d=3){{if(v==null||Number.isNaN(v))return"n/d";return Number(v).toFixed(d).replace(/0+$/,"").replace(/\\.$/,"")}}
function init(){{document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>{{document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("active",x==b));document.querySelectorAll(".hyp").forEach(x=>x.classList.toggle("active",x.id==b.dataset.tab));render();}});document.getElementById("classToggles").innerHTML=classes.map(c=>`<label class="chip"><input type="checkbox" checked data-c="${{c}}"><span class="dot" style="background:${{colors[c]}}"></span>${{c}}</label>`).join("");document.querySelectorAll("#classToggles input").forEach(i=>i.onchange=()=>{{i.checked?state.visible.add(i.dataset.c):state.visible.delete(i.dataset.c);render();}});let opts=Object.keys(labels).filter(k=>k!="PC1"&&k!="PC2").map(k=>`<option value="${{k}}">${{labels[k]}}</option>`).join("");xFeature.innerHTML=opts;yFeature.innerHTML=opts;xFeature.value=state.x;yFeature.value=state.y;xFeature.onchange=e=>{{state.x=e.target.value;render()}};yFeature.onchange=e=>{{state.y=e.target.value;render()}};qualityFeature.onchange=e=>{{state.quality=e.target.value;render()}};histFeature.onchange=e=>{{state.hist=e.target.value;render()}};h2Mode.onchange=e=>{{state.h2Mode=e.target.value;render()}};h3Flux.onchange=e=>{{state.h3Flux=e.target.value;render()}};renderExports()}}
function metric(n,v,s){{return`<div class="metric"><span>${{n}}</span><strong>${{v}}</strong><small>${{s||""}}</small></div>`}}
function renderMetrics(){{let total=Object.values(DATA.classCounts).reduce((a,b)=>a+b,0);metrics.innerHTML=metric("Muestra dashboard",DATA.sampleRows.toLocaleString(),"filas FITS locales")+classes.map(c=>metric(c,(DATA.classCounts[c]||0).toLocaleString(),fmt((DATA.classCounts[c]||0)/total*100,1)+"% dataset")).join("")+metric("Alineación FITS",DATA.source.aligned_by_row?"OK":"revisar","OBJID vs BESTOBJID")}}
function dim(svg){{let r=svg.getBoundingClientRect();return{{w:Math.max(r.width,320),h:Math.max(r.height,340),m:{{l:58,r:18,t:24,b:50}}}}}}
function vals(data,k){{return data.map(d=>d[k]).filter(v=>v!=null&&Number.isFinite(v))}} function pct(a,p){{a=a.slice().sort((x,y)=>x-y);return a[Math.max(0,Math.min(a.length-1,Math.floor((a.length-1)*p)))]}}
function extent(data,k){{let a=vals(data,k);if(!a.length)return[0,1];let lo=pct(a,.01),hi=pct(a,.99);if(lo==hi){{lo-=1;hi+=1}}return[lo,hi]}} function sc(v,d,r){{return r[0]+(v-d[0])/(d[1]-d[0])*(r[1]-r[0])}}
function axes(d,xd,yd,xl,yl){{let{{w,h,m}}=d,pw=w-m.l-m.r,ph=h-m.t-m.b,s=`<line x1="${{m.l}}" y1="${{h-m.b}}" x2="${{w-m.r}}" y2="${{h-m.b}}" stroke="#98a2b3"/><line x1="${{m.l}}" y1="${{m.t}}" x2="${{m.l}}" y2="${{h-m.b}}" stroke="#98a2b3"/>`;for(let i=0;i<=5;i++){{let x=m.l+pw*i/5,vx=xd[0]+(xd[1]-xd[0])*i/5,y=h-m.b-ph*i/5,vy=yd[0]+(yd[1]-yd[0])*i/5;s+=`<text x="${{x}}" y="${{h-m.b+20}}" text-anchor="middle" font-size="11" fill="#667085">${{fmt(vx,2)}}</text><text x="${{m.l-8}}" y="${{y+4}}" text-anchor="end" font-size="11" fill="#667085">${{fmt(vy,2)}}</text>`}}return s+`<text x="${{m.l+pw/2}}" y="${{h-9}}" text-anchor="middle" font-size="13">${{xl}}</text><text transform="translate(16 ${{m.t+ph/2}}) rotate(-90)" text-anchor="middle" font-size="13">${{yl}}</text>`}}
function scatter(id,xk,yk,data=DATA.scatter){{let svg=document.getElementById(id),d=dim(svg),dat=data.filter(r=>state.visible.has(r.CLASS)&&Number.isFinite(r[xk])&&Number.isFinite(r[yk])),xd=extent(dat,xk),yd=extent(dat,yk),{{w,h,m}}=d;let s=axes(d,xd,yd,labels[xk]||xk,labels[yk]||yk);s+=dat.map(r=>`<circle cx="${{sc(r[xk],xd,[m.l,w-m.r])}}" cy="${{sc(r[yk],yd,[h-m.b,m.t])}}" r="3" fill="${{colors[r.CLASS]}}" opacity=".62" data-tip="${{r.CLASS}}<br>${{labels[xk]}}: ${{fmt(r[xk])}}<br>${{labels[yk]}}: ${{fmt(r[yk])}}"/>`).join("");svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`);svg.innerHTML=s;tips(svg)}}
function hist(id,feature){{let svg=document.getElementById(id),d=dim(svg),{{w,h,m}}=d,obj=DATA.histograms[feature];if(!obj)return;let edges=Object.values(obj).flatMap(o=>o.edges),counts=Object.entries(obj).filter(([c])=>state.visible.has(c)).flatMap(([,o])=>o.counts),xd=[Math.min(...edges),Math.max(...edges)],yd=[0,Math.max(...counts,1)];let s=axes(d,xd,yd,labels[feature]||feature,"frecuencia");classes.forEach(c=>{{if(!state.visible.has(c)||!obj[c])return;let pts=obj[c].counts.map((v,i)=>{{let mid=(obj[c].edges[i]+obj[c].edges[i+1])/2;return`${{sc(mid,xd,[m.l,w-m.r])}},${{sc(v,yd,[h-m.b,m.t])}}`}}).join(" ");s+=`<polyline points="${{pts}}" fill="none" stroke="${{colors[c]}}" stroke-width="2.4"/>`}});svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`);svg.innerHTML=s}}
function lineQuality(){{let svg=plotH4,d=dim(svg),dat=DATA.quality[state.quality],xd=extent(dat,"x"),yd=[0,Math.max(...dat.map(r=>r.error_rate||0),.01)],{{w,h,m}}=d;let s=axes(d,xd,yd,state.quality,"tasa de error físico");let pts=dat.map(r=>`${{sc(r.x,xd,[m.l,w-m.r])}},${{sc(r.error_rate,yd,[h-m.b,m.t])}}`).join(" ");s+=`<polyline points="${{pts}}" fill="none" stroke="#d64545" stroke-width="3"/>`+dat.map(r=>`<circle cx="${{sc(r.x,xd,[m.l,w-m.r])}}" cy="${{sc(r.error_rate,yd,[h-m.b,m.t])}}" r="4" fill="#d64545" data-tip="bin: ${{r.bin}}<br>error: ${{fmt(r.error_rate)}}<br>SN mediana: ${{fmt(r.sn_median)}}"/>`).join("");svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`);svg.innerHTML=s;tips(svg)}}
function table(id,rows,cols){{document.getElementById(id).innerHTML=`<table><thead><tr>${{cols.map(c=>`<th>${{c[1]}}</th>`).join("")}}</tr></thead><tbody>${{rows.map(r=>`<tr>${{cols.map(c=>`<td>${{c[2]?c[2](r[c[0]],r):fmt(r[c[0]])}}</td>`).join("")}}</tr>`).join("")}}</tbody></table>`}}
function renderExports(){{csvExports.innerHTML=DATA.csvExports.map(e=>`<article class="csv-card"><a href="${{e.archivo}}">${{e.titulo}}</a><div class="mini">${{e.hipotesis}} · ${{e.filas.toLocaleString()}} filas</div><p>${{e.descripcion}}</p><code>${{e.archivo}}</code></article>`).join("")}}
function previewTable(t){{if(!t.exists)return`<p class="mini">No encontrado: <code>${{t.path}}</code></p>`;let head=t.columns.map(c=>`<th>${{c}}</th>`).join("");let body=t.rows.map(r=>`<tr>${{t.columns.map(c=>`<td>${{r[c]??""}}</td>`).join("")}}</tr>`).join("");return`<div class="step-table"><h4>${{t.title}}</h4><div class="mini">${{t.total_rows.toLocaleString()}} filas · ${{t.total_cols}} columnas · <a href="${{t.path}}">abrir CSV</a></div><div class="table"><table><thead><tr>${{head}}</tr></thead><tbody>${{body}}</tbody></table></div></div>`}}
function renderSteps(){{stepsGrid.innerHTML=DATA.steps.map(s=>{{let tabs=(DATA.stepTables[s[0]]||[]).map(previewTable).join("");return`<article class="step"><h3>${{s[0]}}: ${{s[1]}}</h3><p>${{s[2]}}</p><details><summary>Ver tablas y CSV usados en este paso</summary>${{tabs}}</details></article>`}}).join("")}}
function tips(svg){{let t=tooltip;svg.querySelectorAll("[data-tip]").forEach(e=>{{e.onmousemove=ev=>{{t.innerHTML=e.dataset.tip;t.style.opacity=1;t.style.left=ev.clientX+"px";t.style.top=ev.clientY+"px"}};e.onmouseleave=()=>t.style.opacity=0}})}}
function legends(){{["legendH1","legendH2","legendH5"].forEach(id=>document.getElementById(id).innerHTML=classes.map(c=>`<span><i class="dot" style="background:${{colors[c]}}"></i>${{c}}</span>`).join(""))}}
function findings(){{let m=DATA.metrics;findingH1.innerHTML=`<h2>Lectura</h2><p>R² colores→Z: <b>${{fmt(m.photoz_r2_colors)}}</b>. R² magnitudes+colores→Z: <b>${{fmt(m.photoz_r2_mags_colors)}}</b>.</p><p>Sirve como baseline lineal; si es moderado, justifica modelos no lineales como XGBoost o redes.</p>`;findingH2.innerHTML=`<h2>Lectura</h2><p>Accuracy exploratoria por centroides: luz <b>${{fmt(m.class_acc_light)}}</b>, forma <b>${{fmt(m.class_acc_morph)}}</b>, colores <b>${{fmt(m.class_acc_colors)}}</b>, combinado/PCA base <b>${{fmt(m.class_acc_multi)}}</b>.</p>`;findingH3.innerHTML=`<h2>Lectura</h2><p>La correlación global PETRO vs DEV r es <b>${{fmt(m.corr_global.PETRO_vs_DEV_r)}}</b>; condicionada por objetos extendidos baja/sube a <b>${{fmt(m.corr_conditioned.extended_PETRO_vs_DEV_r)}}</b>.</p><p>Si cambia bastante, la redundancia global oculta morfologías complejas.</p>`;findingH4.innerHTML=`<h2>Lectura</h2><p>Correlación error físico con AIRMASS: <b>${{fmt(m.quality_corr_airmass_error)}}</b>; EXTINCTION: <b>${{fmt(m.quality_corr_extinction_error)}}</b>; Z: <b>${{fmt(m.quality_corr_z_error)}}</b>; SN: <b>${{fmt(m.quality_corr_sn_error)}}</b>.</p>`;findingH5.innerHTML=`<h2>Lectura</h2><p>Comparar histogramas crudos con scatter de colores muestra si la ingeniería de características desenreda clases que se solapan en una sola variable.</p>`}}
function render(){{renderMetrics();legends();findings();scatter("plotH1","PSFMAG_ug","Z");hist("histZ","Z");if(state.h2Mode=="pca"){{scatter("plotH2","PC1","PC2",DATA.pca)}}else{{scatter("plotH2",state.x,state.y)}}scatter("plotH3","PETROFLUX_r",state.h3Flux);lineQuality();scatter("plotH5","PSFMAG_ug","PSFMAG_gr");hist("histGeneric",state.hist);table("rankTable",DATA.ranking,[["feature","variable",(v)=>labels[v]||v],["eta2_class","eta² clase"],["corr_z","corr Z"]]);let corrRows=Object.entries(DATA.metrics.corr_global).map(([k,v])=>({{grupo:"global",par:k,corr:v}})).concat(Object.entries(DATA.metrics.corr_conditioned).map(([k,v])=>({{grupo:"condicionado",par:k,corr:v}})));table("corrCondTable",corrRows,[["grupo","grupo"],["par","par"],["corr","correlación"]]);renderSteps()}}
init();render();window.onresize=render;
</script>
</body></html>"""


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    payload = build_payload()
    DASHBOARD_PATH.write_text(build_html(payload), encoding="utf-8")
    print(f"Dashboard generado: {DASHBOARD_PATH}")
    print(f"Fuente: {payload['source']['photo']} + {payload['source']['spec']}")
    print(f"Muestra: {payload['sampleRows']:,} filas")


if __name__ == "__main__":
    main()
