# =========================================================
# DATA WRANGLING COMPLETO - SDSS FITS
# =========================================================
# Autor: Michael
# Dataset: SDSS DR17/DR18
#
# Este script procesa varios FITS sin mezclar resultados.
# Para cada dataset genera:
#   output/analisis/<dataset>/tablas/
#   output/analisis/<dataset>/graficas/
#
# Importante:
#   La carpeta "graficas" ya no guarda PNG. Guarda CSV con los datos
#   necesarios para que el dashboard dibuje histogramas, boxplots,
#   barras, heatmaps y scatterplots.
# =========================================================

from astropy.io import fits

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =========================================================
# CONFIGURACION
# =========================================================

DATASETS = [
    {
        "nombre": "photoPosPlate-dr17",
        "ruta": r"data\photoPosPlate-dr17.fits",
    },
    {
        "nombre": "specObj-dr17",
        "ruta": r"data\specObj-dr17.fits",
    },
]

BASE_OUTPUT_DIR = "output"
ANALYSIS_DIR = os.path.join(BASE_OUTPUT_DIR, "analisis")

# Si se activa, genera un CSV expandido por dataset. Puede ocupar muchos GB.
EXPORT_EXPANDED_CSV = False
CHUNK_SIZE = 50000

# Limite de columnas para correlacion. Usar None puede ser muy costoso.
CORRELATION_MAX_COLUMNS = None

# Datos para el dashboard.
HISTOGRAM_BINS = 50
SCATTER_MAX_ROWS = 50000
SCATTER_MAX_PAIRS = 30
CATEGORICAL_TOP_N = 30

BANDS_5 = ["u", "g", "r", "i", "z"]

os.makedirs(ANALYSIS_DIR, exist_ok=True)


# =========================================================
# UTILIDADES
# =========================================================

def clean_name(value):
    return (
        str(value)
        .replace(os.sep, "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def make_dirs(dataset_name):
    dataset_dir = os.path.join(ANALYSIS_DIR, dataset_name)
    table_dir = os.path.join(dataset_dir, "tablas")
    graph_dir = os.path.join(dataset_dir, "graficas")

    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    return dataset_dir, table_dir, graph_dir


def component_suffix(indices, dims):
    parts = []

    for axis, idx in enumerate(indices):
        if axis == 0 and dims[axis] == 5:
            parts.append(BANDS_5[idx])
        else:
            parts.append(f"{idx:02d}")

    return "_".join(parts)


def build_expanded_columns(data):
    expanded = []

    for original in data.names:
        arr = data[original]
        shape = arr.shape
        dtype = arr.dtype

        if len(shape) == 1:
            expanded.append({
                "columna": original,
                "columna_original": original,
                "indices": (),
                "shape_original": str(shape),
                "tipo_dato": str(dtype),
                "dimensiones_internas": "",
            })
            continue

        dims = shape[1:]

        for indices in np.ndindex(*dims):
            suffix = component_suffix(indices, dims)
            expanded.append({
                "columna": f"{original}_{suffix}",
                "columna_original": original,
                "indices": indices,
                "shape_original": str(shape),
                "tipo_dato": str(dtype),
                "dimensiones_internas": str(dims),
            })

    return expanded


def get_column_values(data, info, start=None, end=None):
    arr = data[info["columna_original"]]
    row_slice = slice(start, end) if start is not None or end is not None else slice(None)

    if not info["indices"]:
        values = arr[row_slice]
    else:
        slicer = (row_slice,) + tuple(info["indices"])
        values = arr[slicer]

    return np.asarray(values)


def normalize_strings(values):
    series = pd.Series(values)

    if series.dtype == object:
        return series.map(
            lambda x: x.decode("utf-8", errors="ignore").strip()
            if isinstance(x, (bytes, bytearray))
            else x
        )

    return series.astype(str).str.strip()


def finite_numeric_values(values):
    values = np.asarray(values)

    if np.issubdtype(values.dtype, np.number):
        values = values.astype(np.float64, copy=False)
        return values[np.isfinite(values)]

    return np.array([], dtype=float)


def is_numeric_info(data, info):
    return np.issubdtype(data[info["columna_original"]].dtype, np.number)


def safe_mode(values):
    if len(values) == 0:
        return np.nan

    mode_values = pd.Series(values).mode(dropna=True)

    if mode_values.empty:
        return np.nan

    return mode_values.iloc[0]


def safe_geometric_mean(values):
    positive = values[values > 0]

    if len(positive) == 0:
        return np.nan

    return float(np.exp(np.mean(np.log(positive))))


def safe_harmonic_mean(values):
    positive = values[values > 0]

    if len(positive) == 0:
        return np.nan

    return float(len(positive) / np.sum(1.0 / positive))


def classify_numeric_variable(values, unique_count):
    values = np.asarray(values)

    if unique_count <= 30:
        return "discreta"

    if np.issubdtype(values.dtype, np.integer):
        return "discreta"

    return "continua"


def get_grouped_numeric_specs(numeric_infos):
    """Agrupa atributos cuyo primer eje interno son canales u,g,r,i,z."""
    specs = []
    grouped_cols = set()
    by_original = {}

    for info in numeric_infos:
        by_original.setdefault(info["columna_original"], []).append(info)

    for original, infos in by_original.items():
        indexed = [info for info in infos if info["indices"]]

        if not indexed:
            for info in infos:
                specs.append({
                    "grafica": info["columna"],
                    "titulo": info["columna"],
                    "columnas": [info],
                    "tipo": "escalar",
                })
            continue

        first_axis = sorted({info["indices"][0] for info in indexed})

        if first_axis != list(range(5)):
            for info in infos:
                specs.append({
                    "grafica": info["columna"],
                    "titulo": info["columna"],
                    "columnas": [info],
                    "tipo": "expandida",
                })
            continue

        tail_keys = sorted({tuple(info["indices"][1:]) for info in indexed})

        for tail in tail_keys:
            group = []

            for band_idx in range(5):
                match = next(
                    (
                        info for info in indexed
                        if info["indices"][0] == band_idx
                        and tuple(info["indices"][1:]) == tail
                    ),
                    None,
                )

                if match is not None:
                    group.append(match)

            if len(group) < 2:
                continue

            tail_suffix = "_".join(f"{idx:02d}" for idx in tail)
            graph_name = original if not tail_suffix else f"{original}_{tail_suffix}"
            specs.append({
                "grafica": graph_name,
                "titulo": (
                    f"{original} - canales u,g,r,i,z"
                    if not tail_suffix
                    else f"{original} {tail_suffix} - canales u,g,r,i,z"
                ),
                "columnas": group,
                "tipo": "canales_ugriz",
            })
            grouped_cols.update(info["columna"] for info in group)

        for info in infos:
            if info["columna"] not in grouped_cols:
                specs.append({
                    "grafica": info["columna"],
                    "titulo": info["columna"],
                    "columnas": [info],
                    "tipo": "expandida",
                })

    return specs


def export_expanded_csv(data, expanded_columns, dataset_dir, dataset_name):
    print("\nExportando CSV expandido por chunks...")

    path = os.path.join(dataset_dir, f"{dataset_name}_expandido.csv")
    first_chunk = True
    total_rows = len(data)

    for start in range(0, total_rows, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_rows)
        chunk_dict = {}

        for info in expanded_columns:
            chunk_dict[info["columna"]] = get_column_values(data, info, start, end)

        chunk_df = pd.DataFrame(chunk_dict)

        for col in chunk_df.select_dtypes(include=["object"]).columns:
            chunk_df[col] = chunk_df[col].map(
                lambda x: x.decode("utf-8", errors="ignore").strip()
                if isinstance(x, (bytes, bytearray))
                else x
            )

        chunk_df.to_csv(
            path,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk,
        )

        first_chunk = False
        print(f"  Filas exportadas: {end:,}/{total_rows:,}")

    print("CSV expandido guardado:", path)


# =========================================================
# CSV PARA GRAFICAS
# =========================================================

def histogram_rows(data, plot_specs):
    rows = []

    for pos, spec in enumerate(plot_specs, start=1):
        for info in spec["columnas"]:
            values = finite_numeric_values(get_column_values(data, info))

            if len(values) == 0:
                continue

            q_low = np.percentile(values, 1)
            q_high = np.percentile(values, 99)

            if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low == q_high:
                q_low = np.min(values)
                q_high = np.max(values)

            if q_low == q_high:
                q_low -= 0.5
                q_high += 0.5

            counts, edges = np.histogram(
                np.clip(values, q_low, q_high),
                bins=HISTOGRAM_BINS,
                range=(q_low, q_high),
            )

            serie = info["columna"].replace(f"{info['columna_original']}_", "")

            for idx, count in enumerate(counts):
                rows.append({
                    "grafica": spec["grafica"],
                    "titulo": spec["titulo"],
                    "variable": info["columna"],
                    "serie": serie,
                    "bin_left": edges[idx],
                    "bin_right": edges[idx + 1],
                    "bin_center": (edges[idx] + edges[idx + 1]) / 2,
                    "frecuencia": int(count),
                })

        if pos % 25 == 0 or pos == len(plot_specs):
            print(f"  Histogramas CSV: {pos}/{len(plot_specs)}")

    return pd.DataFrame(rows)


def boxplot_rows(stats_df, plot_specs):
    if stats_df.empty:
        return pd.DataFrame()

    stats_by_col = stats_df.drop_duplicates("columna", keep="last").set_index("columna")
    rows = []

    for spec in plot_specs:
        for info in spec["columnas"]:
            col = info["columna"]

            if col not in stats_by_col.index:
                continue

            row = stats_by_col.loc[col]
            serie = col.replace(f"{info['columna_original']}_", "")
            whislo = max(row["min"], row["limite_inferior_iqr"])
            whishi = min(row["max"], row["limite_superior_iqr"])

            rows.append({
                "grafica": spec["grafica"],
                "titulo": spec["titulo"],
                "variable": col,
                "serie": serie,
                "min": row["min"],
                "q1": row["q1"],
                "mediana": row["mediana"],
                "q3": row["q3"],
                "max": row["max"],
                "whislo": whislo,
                "whishi": whishi,
                "outlier_lower": row["limite_inferior_iqr"],
                "outlier_upper": row["limite_superior_iqr"],
            })

    return pd.DataFrame(rows)


def categorical_graph_rows(data, categorical_infos):
    rows = []

    for info in categorical_infos:
        try:
            col = info["columna"]
            values = normalize_strings(get_column_values(data, info))
            counts = values.value_counts(dropna=False).head(CATEGORICAL_TOP_N)

            for category, count in counts.items():
                rows.append({
                    "grafica": col,
                    "variable": col,
                    "categoria": category,
                    "frecuencia": int(count),
                    "porcentaje": (count / len(values)) * 100,
                })

        except Exception:
            pass

    return pd.DataFrame(rows)


def scatter_rows(data, numeric_infos, corr_matrix):
    info_by_name = {info["columna"]: info for info in numeric_infos}
    pairs = []

    if not corr_matrix.empty:
        candidate_targets = ["Z", "redshift", "REDSHIFT"]
        target = next((col for col in candidate_targets if col in corr_matrix.columns), None)

        if target is not None:
            top_corr = corr_matrix[target].drop(target, errors="ignore").abs().sort_values(ascending=False)

            for col in top_corr.head(SCATTER_MAX_PAIRS).index:
                if col in info_by_name:
                    pairs.append((col, target))

    if not pairs:
        for i in range(min(SCATTER_MAX_PAIRS, len(numeric_infos) - 1)):
            pairs.append((numeric_infos[i]["columna"], numeric_infos[i + 1]["columna"]))

    rows = []

    for x_col, y_col in pairs:
        try:
            x = np.asarray(get_column_values(data, info_by_name[x_col]), dtype=np.float64)
            y = np.asarray(get_column_values(data, info_by_name[y_col]), dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            if len(x) == 0:
                continue

            if len(x) > SCATTER_MAX_ROWS:
                step = max(1, len(x) // SCATTER_MAX_ROWS)
                x = x[::step]
                y = y[::step]

            graph_name = f"{x_col}_vs_{y_col}"

            for idx in range(len(x)):
                rows.append({
                    "grafica": graph_name,
                    "x_col": x_col,
                    "y_col": y_col,
                    "x": x[idx],
                    "y": y[idx],
                })

        except Exception:
            pass

    return pd.DataFrame(rows)


def heatmap_long_rows(corr_matrix):
    rows = []

    if corr_matrix.empty:
        return pd.DataFrame()

    for row_name in corr_matrix.index:
        for col_name in corr_matrix.columns:
            rows.append({
                "feature_x": col_name,
                "feature_y": row_name,
                "correlacion": corr_matrix.loc[row_name, col_name],
            })

    return pd.DataFrame(rows)


# =========================================================
# CORRELACION
# =========================================================

def compute_full_correlation(data, numeric_infos):
    selected = numeric_infos

    if CORRELATION_MAX_COLUMNS is not None and len(selected) > CORRELATION_MAX_COLUMNS:
        selected = selected[:CORRELATION_MAX_COLUMNS]
        print(
            f"Correlacion limitada a {CORRELATION_MAX_COLUMNS} columnas numericas "
            "para evitar una matriz demasiado costosa."
        )

    names = [info["columna"] for info in selected]
    n_cols = len(selected)

    if n_cols == 0:
        return pd.DataFrame()

    sums = np.zeros(n_cols, dtype=np.float64)
    sums_sq = np.zeros(n_cols, dtype=np.float64)
    counts = np.zeros(n_cols, dtype=np.float64)
    cross = np.zeros((n_cols, n_cols), dtype=np.float64)
    pair_counts = np.zeros((n_cols, n_cols), dtype=np.float64)

    total_rows = len(data)

    for start in range(0, total_rows, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_rows)
        matrix = np.empty((end - start, n_cols), dtype=np.float64)

        for idx, info in enumerate(selected):
            matrix[:, idx] = np.asarray(
                get_column_values(data, info, start, end),
                dtype=np.float64,
            )

        mask = np.isfinite(matrix)
        clean = np.where(mask, matrix, 0.0)

        sums += clean.sum(axis=0)
        sums_sq += (clean * clean).sum(axis=0)
        counts += mask.sum(axis=0)
        cross += clean.T @ clean
        pair_counts += mask.astype(np.float64).T @ mask.astype(np.float64)

        print(f"  Correlacion: {end:,}/{total_rows:,} filas")

    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    variances = np.divide(sums_sq, counts, out=np.full_like(sums_sq, np.nan), where=counts > 0) - means**2
    stds = np.sqrt(np.maximum(variances, 0))

    corr = np.full((n_cols, n_cols), np.nan, dtype=np.float64)

    for i in range(n_cols):
        for j in range(n_cols):
            if pair_counts[i, j] == 0 or stds[i] == 0 or stds[j] == 0:
                continue

            cov = (cross[i, j] / pair_counts[i, j]) - (means[i] * means[j])
            corr[i, j] = cov / (stds[i] * stds[j])

    return pd.DataFrame(corr, index=names, columns=names)


# =========================================================
# ANALISIS DE UN DATASET
# =========================================================

def process_dataset(dataset):
    dataset_name = dataset["nombre"]
    fits_path = dataset["ruta"]
    dataset_dir, table_dir, graph_dir = make_dirs(dataset_name)

    print("\n" + "=" * 70)
    print(f"PROCESANDO DATASET: {dataset_name}")
    print("=" * 70)

    if not os.path.exists(fits_path):
        print(f"No existe el archivo: {fits_path}")
        return None

    hdul = fits.open(fits_path, memmap=True)
    data = hdul[1].data

    total_rows = len(data)
    original_columns = len(data.names)

    print("Archivo:", fits_path)
    print("Registros:", f"{total_rows:,}")
    print("Columnas originales:", original_columns)

    print("\nDetectando columnas expandidas...")
    expanded_columns = build_expanded_columns(data)
    numeric_infos = [info for info in expanded_columns if is_numeric_info(data, info)]
    categorical_infos = [info for info in expanded_columns if not is_numeric_info(data, info)]

    metadata_df = pd.DataFrame(expanded_columns)
    metadata_df.to_csv(os.path.join(table_dir, "metadata_expandida.csv"), index=False)

    metadata_original = []

    for name in data.names:
        arr = data[name]
        metadata_original.append({
            "atributo": name,
            "tipo_dato": str(arr.dtype),
            "shape": str(arr.shape),
            "es_array": len(arr.shape) > 1,
            "dimensiones_internas": str(arr.shape[1:]) if len(arr.shape) > 1 else "",
        })

    pd.DataFrame(metadata_original).to_csv(os.path.join(table_dir, "metadata_original.csv"), index=False)

    print("Columnas expandidas:", len(expanded_columns))
    print("Numericas:", len(numeric_infos))
    print("Categoricas/texto:", len(categorical_infos))

    if EXPORT_EXPANDED_CSV:
        export_expanded_csv(data, expanded_columns, dataset_dir, dataset_name)

    print("\nCalculando nulos, estadisticas y outliers...")

    null_info = []
    stats_info = []
    outlier_info = []
    stats_error_info = []

    for pos, info in enumerate(expanded_columns, start=1):
        col = info["columna"]
        values = None

        try:
            values = get_column_values(data, info)
            nulls = int(pd.isnull(values).sum())
            null_info.append({
                "columna": col,
                "nulos": nulls,
                "porcentaje_nulos": (nulls / total_rows) * 100,
            })

        except Exception as exc:
            null_info.append({
                "columna": col,
                "nulos": "ERROR",
                "porcentaje_nulos": "ERROR",
                "error": str(exc),
            })

        if values is not None and is_numeric_info(data, info):
            try:
                finite = finite_numeric_values(values)

                if len(finite) > 0:
                    q1 = np.percentile(finite, 25)
                    q3 = np.percentile(finite, 75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = int(np.sum((finite < lower) | (finite > upper)))
                    unique_count = int(len(np.unique(finite)))
                    mode_value = safe_mode(finite) if unique_count <= 10000 else np.nan

                    stats_info.append({
                        "columna": col,
                        "conteo_validos": int(len(finite)),
                        "tipo_variable": classify_numeric_variable(values, unique_count),
                        "media": float(np.mean(finite)),
                        "media_geometrica": safe_geometric_mean(finite),
                        "media_armonica": safe_harmonic_mean(finite),
                        "mediana": float(np.median(finite)),
                        "moda": mode_value,
                        "std": float(np.std(finite)),
                        "min": float(np.min(finite)),
                        "max": float(np.max(finite)),
                        "q1": float(q1),
                        "q3": float(q3),
                        "iqr": float(iqr),
                        "limite_inferior_iqr": float(lower),
                        "limite_superior_iqr": float(upper),
                        "valores_unicos": unique_count,
                    })

                    outlier_info.append({
                        "columna": col,
                        "limite_inferior": float(lower),
                        "limite_superior": float(upper),
                        "cantidad_outliers": outliers,
                        "porcentaje_outliers": (outliers / len(finite)) * 100,
                    })

            except Exception as exc:
                stats_error_info.append({
                    "columna": col,
                    "error": str(exc),
                })

        if pos % 50 == 0 or pos == len(expanded_columns):
            print(f"  Columnas procesadas: {pos}/{len(expanded_columns)}")

    null_df = pd.DataFrame(null_info)
    stats_df = pd.DataFrame(stats_info)
    outlier_df = pd.DataFrame(outlier_info)

    null_df.to_csv(os.path.join(table_dir, "nulos.csv"), index=False)
    stats_df.to_csv(os.path.join(table_dir, "estadisticas.csv"), index=False)
    outlier_df.to_csv(os.path.join(table_dir, "outliers.csv"), index=False)
    pd.DataFrame(stats_error_info).to_csv(os.path.join(table_dir, "errores_estadisticas.csv"), index=False)

    print("\nAnalizando duplicados...")
    duplicate_rows = []

    for candidate in ["SPECOBJID", "specObjID", "OBJID", "objID", "BESTOBJID", "TARGETOBJID"]:
        match = next((info for info in expanded_columns if info["columna"] == candidate), None)

        if match is None:
            continue

        try:
            values = get_column_values(data, match)
            duplicated = len(values) - len(np.unique(values))
            duplicate_rows.append({
                "columna": candidate,
                "duplicados": int(duplicated),
            })
        except Exception as exc:
            duplicate_rows.append({
                "columna": candidate,
                "duplicados": "ERROR",
                "error": str(exc),
            })

    duplicate_df = pd.DataFrame(duplicate_rows)
    duplicate_df.to_csv(os.path.join(table_dir, "duplicados.csv"), index=False)

    print("\nAnalizando categoricas...")
    categorical_graph_df = categorical_graph_rows(data, categorical_infos)
    categorical_graph_df.to_csv(os.path.join(graph_dir, "barras_categoricas.csv"), index=False)

    for info in categorical_infos:
        try:
            col = info["columna"]
            values = normalize_strings(get_column_values(data, info))
            values.value_counts(dropna=False).head(CATEGORICAL_TOP_N).to_csv(
                os.path.join(table_dir, f"value_counts_top_{clean_name(col)}.csv")
            )
        except Exception:
            pass

    class_info = next((info for info in expanded_columns if info["columna"].upper() == "CLASS"), None)

    if class_info is not None:
        classes = normalize_strings(get_column_values(data, class_info))
        class_counts = classes.value_counts(dropna=False)
        class_counts.to_csv(os.path.join(table_dir, "distribucion_clases.csv"))

        class_rows = []
        for cls, count in class_counts.items():
            class_rows.append({
                "grafica": "distribucion_clases",
                "variable": "CLASS",
                "categoria": cls,
                "frecuencia": int(count),
                "porcentaje": (count / total_rows) * 100,
            })
        pd.DataFrame(class_rows).to_csv(os.path.join(graph_dir, "distribucion_clases.csv"), index=False)

    print("\nGenerando CSV para histogramas y boxplots...")
    plot_specs = get_grouped_numeric_specs(numeric_infos)
    histogram_df = histogram_rows(data, plot_specs)
    histogram_df.to_csv(os.path.join(graph_dir, "histogramas.csv"), index=False)

    boxplot_df = boxplot_rows(stats_df, plot_specs)
    boxplot_df.to_csv(os.path.join(graph_dir, "boxplots.csv"), index=False)

    print("\nCalculando correlacion...")
    corr_matrix = compute_full_correlation(data, numeric_infos)
    corr_matrix.to_csv(os.path.join(table_dir, "correlacion.csv"))
    heatmap_long_rows(corr_matrix).to_csv(os.path.join(graph_dir, "heatmap_correlacion.csv"), index=False)

    cov_matrix = pd.DataFrame()
    if not corr_matrix.empty and not stats_df.empty:
        std_by_col = stats_df.drop_duplicates("columna", keep="last").set_index("columna")["std"].reindex(corr_matrix.columns)
        cov_values = corr_matrix.values * np.outer(std_by_col.values, std_by_col.values)
        cov_matrix = pd.DataFrame(cov_values, index=corr_matrix.index, columns=corr_matrix.columns)
        cov_matrix.to_csv(os.path.join(table_dir, "covarianza.csv"))

    print("\nGenerando CSV para scatterplots...")
    scatter_df = scatter_rows(data, numeric_infos, corr_matrix)
    scatter_df.to_csv(os.path.join(graph_dir, "scatterplots.csv"), index=False)

    graph_manifest = pd.DataFrame([
        {
            "archivo": "histogramas.csv",
            "tipo_grafica": "histograma",
            "descripcion": "Bins y frecuencias por variable/serie. Usa todo el dataset.",
        },
        {
            "archivo": "boxplots.csv",
            "tipo_grafica": "boxplot",
            "descripcion": "Cuartiles, mediana, bigotes y limites IQR por variable/serie.",
        },
        {
            "archivo": "barras_categoricas.csv",
            "tipo_grafica": "barras/pie",
            "descripcion": f"Top {CATEGORICAL_TOP_N} categorias por variable categorica.",
        },
        {
            "archivo": "heatmap_correlacion.csv",
            "tipo_grafica": "heatmap",
            "descripcion": "Matriz de correlacion en formato largo para dibujar en dashboard.",
        },
        {
            "archivo": "scatterplots.csv",
            "tipo_grafica": "scatterplot",
            "descripcion": f"Muestra visual de hasta {SCATTER_MAX_ROWS:,} puntos por par de variables.",
        },
        {
            "archivo": "distribucion_clases.csv",
            "tipo_grafica": "barras/pie",
            "descripcion": "Distribucion de CLASS cuando el dataset contiene esa columna.",
        },
    ])
    graph_manifest.to_csv(os.path.join(graph_dir, "manifest_graficas.csv"), index=False)

    print("\nGenerando tablas resumen...")
    stats_by_name = (
        stats_df.drop_duplicates("columna", keep="last").set_index("columna")
        if not stats_df.empty
        else pd.DataFrame()
    )
    null_by_name = (
        null_df.drop_duplicates("columna", keep="last").set_index("columna")
        if not null_df.empty
        else pd.DataFrame()
    )

    type_rows = []

    for info in expanded_columns:
        col = info["columna"]
        row = {
            "columna": col,
            "columna_original": info["columna_original"],
            "tipo_dato": info["tipo_dato"],
            "tipo_general": "numerica" if is_numeric_info(data, info) else "categorica/texto",
            "tipo_variable": "",
            "valores_unicos": "",
            "min": "",
            "max": "",
            "porcentaje_nulos": "",
        }

        if col in stats_by_name.index:
            row["tipo_variable"] = stats_by_name.loc[col, "tipo_variable"]
            row["valores_unicos"] = stats_by_name.loc[col, "valores_unicos"]
            row["min"] = stats_by_name.loc[col, "min"]
            row["max"] = stats_by_name.loc[col, "max"]

        if col in null_by_name.index:
            row["porcentaje_nulos"] = null_by_name.loc[col, "porcentaje_nulos"]

        type_rows.append(row)

    types_df = pd.DataFrame(type_rows)
    types_df.to_csv(os.path.join(table_dir, "tipos_y_rangos.csv"), index=False)

    high_nulls = pd.DataFrame()
    if not null_df.empty and "porcentaje_nulos" in null_df.columns:
        high_nulls = null_df[pd.to_numeric(null_df["porcentaje_nulos"], errors="coerce") > 30].copy()
    high_nulls.to_csv(os.path.join(table_dir, "columnas_muchos_nulos.csv"), index=False)

    if not outlier_df.empty:
        outlier_df.sort_values("porcentaje_outliers", ascending=False).head(30).to_csv(
            os.path.join(table_dir, "top_outliers.csv"),
            index=False,
        )
    else:
        pd.DataFrame().to_csv(os.path.join(table_dir, "top_outliers.csv"), index=False)

    top_corr_pairs = []

    if not corr_matrix.empty:
        corr_abs = corr_matrix.abs()

        for i, row_name in enumerate(corr_abs.index):
            for j, col_name in enumerate(corr_abs.columns):
                if j <= i:
                    continue

                value = corr_matrix.loc[row_name, col_name]

                if pd.notnull(value):
                    top_corr_pairs.append({
                        "feature_1": row_name,
                        "feature_2": col_name,
                        "correlacion": value,
                        "correlacion_abs": abs(value),
                    })

    top_corr_df = pd.DataFrame(top_corr_pairs)

    if not top_corr_df.empty:
        top_corr_df = top_corr_df.sort_values("correlacion_abs", ascending=False).head(50)

    top_corr_df.to_csv(os.path.join(table_dir, "top_correlaciones.csv"), index=False)

    summary = {
        "Dataset": dataset_name,
        "FITS_Path": fits_path,
        "Directorio_Analisis": dataset_dir,
        "Total_Registros": total_rows,
        "Columnas_Originales": original_columns,
        "Columnas_Expandidas": len(expanded_columns),
        "Columnas_Numericas_Expandidas": len(numeric_infos),
        "Columnas_Categoricas_Expandidas": len(categorical_infos),
        "CSV_Expandido_Generado": EXPORT_EXPANDED_CSV,
        "Graficas_Formato": "CSV",
    }

    pd.DataFrame([summary]).to_csv(os.path.join(table_dir, "resumen_general.csv"), index=False)

    hdul.close()

    print("\nDataset completado:", dataset_name)
    print("Tablas:", table_dir)
    print("Datos para graficas:", graph_dir)

    return summary


# =========================================================
# EJECUCION
# =========================================================

def main():
    summaries = []

    for dataset in DATASETS:
        summary = process_dataset(dataset)

        if summary is not None:
            summaries.append(summary)

    pd.DataFrame(summaries).to_csv(
        os.path.join(ANALYSIS_DIR, "indice_analisis.csv"),
        index=False,
    )

    print("\n" + "=" * 70)
    print("DATA WRANGLING COMPLETADO PARA TODOS LOS DATASETS")
    print("=" * 70)
    print("Indice:", os.path.join(ANALYSIS_DIR, "indice_analisis.csv"))


if __name__ == "__main__":
    main()
