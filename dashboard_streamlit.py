from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "output" / "hipotesis_csv"
ANALYSIS_DIR = BASE_DIR / "output" / "analisis"
IMG_DIR = BASE_DIR / "output" / "hipotesis_img"

CLASS_COLORS = {"STAR": "#2878b5", "GALAXY": "#2f9e44", "QSO": "#d64545"}


st.set_page_config(
    page_title="Dashboard SDSS interactivo",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv(name):
    path = CSV_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_analysis_csv(relative_path, nrows=2000):
    path = ANALYSIS_DIR / relative_path
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, nrows=nrows)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def show_table(df, title, height=330):
    st.subheader(title)
    if df.empty:
        st.info("No hay datos disponibles para esta tabla.")
        return
    search = st.text_input(f"Filtrar {title}", key=f"filter_{title}")
    view = df
    if search:
        mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        view = df[mask]
    st.dataframe(view, use_container_width=True, height=height)


def filter_classes(df):
    if "CLASS" not in df.columns or df.empty:
        return df
    classes = sorted(df["CLASS"].dropna().unique().tolist())
    selected = st.multiselect("Clases visibles", classes, default=classes)
    return df[df["CLASS"].isin(selected)]


def metric_card(label, value):
    st.metric(label, "n/d" if pd.isna(value) else f"{value:.4f}" if isinstance(value, float) else value)


def scatter_controls(df, default_x, default_y, title):
    if df.empty:
        st.info("No hay datos para graficar.")
        return
    numeric = df.select_dtypes(include="number").columns.tolist()
    left, right = st.columns(2)
    with left:
        x = st.selectbox("Eje X", numeric, index=numeric.index(default_x) if default_x in numeric else 0, key=f"{title}_x")
    with right:
        y = st.selectbox("Eje Y", numeric, index=numeric.index(default_y) if default_y in numeric else min(1, len(numeric) - 1), key=f"{title}_y")
    color = "CLASS" if "CLASS" in df.columns else None
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        color_discrete_map=CLASS_COLORS,
        opacity=0.65,
        render_mode="webgl",
        title=title,
    )
    fig.update_layout(height=620)
    st.plotly_chart(fig, use_container_width=True)


def image_gallery(prefix):
    images = sorted(IMG_DIR.glob(f"{prefix}*.png"))
    if not images:
        return
    with st.expander("Imagenes PNG guardadas", expanded=False):
        cols = st.columns(2)
        for i, image in enumerate(images):
            with cols[i % 2]:
                st.image(str(image), caption=image.name, use_container_width=True)


def h1():
    models = load_csv("h1_redshift_modelos.csv")
    scatter = load_csv("h1_scatter_color_z.csv")
    st.header("H1. Estimacion de Redshift Fotometrico")
    st.write("El CSV se usa como fuente rapida y la grafica se recalcula con los filtros seleccionados.")

    cols = st.columns(2)
    if not models.empty:
        for i, row in models.iterrows():
            with cols[i % 2]:
                metric_card(row["modelo"], row["r2"])
        fig = px.bar(models, x="modelo", y="r2", text="r2", title="R2 por conjunto de variables")
        st.plotly_chart(fig, use_container_width=True)

    filtered = filter_classes(scatter)
    z_range = st.slider(
        "Rango de redshift Z",
        float(filtered["Z"].min()) if not filtered.empty else 0.0,
        float(filtered["Z"].max()) if not filtered.empty else 1.0,
        (float(filtered["Z"].min()) if not filtered.empty else 0.0, float(filtered["Z"].max()) if not filtered.empty else 1.0),
    )
    filtered = filtered[(filtered["Z"] >= z_range[0]) & (filtered["Z"] <= z_range[1])]
    scatter_controls(filtered, "PSFMAG_ug", "Z", "Color fotometrico vs redshift")
    show_table(models, "Resultados H1")
    show_table(filtered.head(500), "Datos filtrados H1", height=260)
    image_gallery("h1")


def h2():
    acc = load_csv("h2_accuracy_grupos.csv")
    pca = load_csv("h2_pca_projection.csv")
    ranking = load_csv("h2_ranking_variables.csv")
    st.header("H2. Clasificacion Morfologica vs Fotometrica")

    if not acc.empty:
        fig = px.bar(acc, x="grupo", y="accuracy_centroides", text="accuracy_centroides", title="Accuracy exploratoria por familia de variables")
        st.plotly_chart(fig, use_container_width=True)
        show_table(acc, "Comparacion de familias")

    mode = st.radio("Vista principal", ["PCA", "Ranking"], horizontal=True)
    if mode == "PCA":
        filtered = filter_classes(pca)
        scatter_controls(filtered, "PC1", "PC2", "PCA multivariado por clase")
    else:
        metric = st.selectbox("Metrica del ranking", ["eta2_class", "corr_z"])
        top_n = st.slider("Top variables", 5, 35, 20)
        top = ranking.sort_values(metric, ascending=False).head(top_n)
        fig = px.bar(top, x=metric, y="feature", orientation="h", title=f"Top {top_n} por {metric}")
        fig.update_layout(height=650, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    show_table(ranking, "Ranking completo")
    image_gallery("h2")


def h3():
    corr = load_csv("h3_correlaciones_condicionadas.csv")
    scatter = load_csv("h3_scatter_flujos.csv")
    st.header("H3. Divergencia de Algoritmos en Morfologias Complejas")

    if not corr.empty:
        fig = px.bar(corr, x="par", y="correlacion", color="grupo", title="Correlacion global vs condicionada")
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)

    filtered = filter_classes(scatter)
    theta_min = st.slider(
        "PETROTHETA_r minimo",
        float(filtered["PETROTHETA_r"].min()) if not filtered.empty else 0.0,
        float(filtered["PETROTHETA_r"].max()) if not filtered.empty else 1.0,
        float(filtered["PETROTHETA_r"].quantile(0.0)) if not filtered.empty else 0.0,
    )
    filtered = filtered[filtered["PETROTHETA_r"] >= theta_min]
    y = st.selectbox("Flujo comparado contra PETROFLUX_r", ["DEVFLUX_r", "EXPFLUX_r", "CMODELFLUX_r"])
    fig = px.scatter(
        filtered,
        x="PETROFLUX_r",
        y=y,
        color="CLASS",
        color_discrete_map=CLASS_COLORS,
        opacity=0.55,
        render_mode="webgl",
        title=f"PETROFLUX_r vs {y}",
    )
    fig.update_layout(height=620)
    st.plotly_chart(fig, use_container_width=True)
    show_table(corr, "Correlaciones")
    show_table(filtered.head(500), "Datos filtrados H3", height=260)
    image_gallery("h3")


def h4():
    bins = load_csv("h4_calidad_bins.csv")
    corr = load_csv("h4_correlaciones_sesgo.csv")
    st.header("H4. Sesgo de Seleccion y Degradacion de Calidad")

    variable = st.multiselect("Variables de calidad", sorted(bins["variable"].unique()) if not bins.empty else [], default=sorted(bins["variable"].unique()) if not bins.empty else [])
    view = bins[bins["variable"].isin(variable)] if variable else bins
    if not view.empty:
        y = st.selectbox("Metrica", ["error_rate", "sn_median", "z_median", "count"])
        fig = px.line(view, x="x", y=y, color="variable", markers=True, title=f"{y} por bins")
        fig.update_layout(height=580)
        st.plotly_chart(fig, use_container_width=True)

    if not corr.empty:
        fig = px.bar(corr, x="variable", y="correlacion_error_fisico", title="Correlacion con error fisico")
        st.plotly_chart(fig, use_container_width=True)
    show_table(view, "Bins de calidad")
    show_table(corr, "Correlaciones de sesgo")
    image_gallery("h4")


def h5():
    scatter = load_csv("h5_color_color_scatter.csv")
    hist = load_csv("h5_histogramas_colores.csv")
    st.header("H5. Separabilidad Estadistica con Ingenieria de Caracteristicas")

    filtered = filter_classes(scatter)
    scatter_controls(filtered, "PSFMAG_ug", "PSFMAG_gr", "Espacio color-color")

    features = sorted(hist["feature"].unique()) if not hist.empty else []
    selected = st.multiselect("Histogramas a comparar", features, default=[f for f in ["PSFMAG_r", "PSFMAG_ug", "PSFMAG_gr"] if f in features])
    hist_view = hist[hist["feature"].isin(selected)] if selected else hist
    if not hist_view.empty:
        fig = px.line(hist_view, x="bin_center", y="count", color="class", line_dash="feature", color_discrete_map=CLASS_COLORS, title="Distribuciones por clase")
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)
    show_table(filtered.head(500), "Datos color-color filtrados", height=260)
    show_table(hist_view, "Bins de histogramas")
    image_gallery("h5")


def pasos():
    st.header("Pasos EDA 0-11")
    st.write("Aqui no mostramos solo links: cada paso carga una vista rapida de sus CSV generados por `programa.py`.")
    mapping = {
        "Paso 0 - Contexto": ["indice_analisis.csv"],
        "Paso 1 - Estructura": ["photoPosPlate-dr17/tablas/resumen_general.csv", "specObj-dr17/tablas/resumen_general.csv", "photoPosPlate-dr17/tablas/metadata_expandida.csv", "specObj-dr17/tablas/metadata_expandida.csv"],
        "Paso 2 - Calidad": ["photoPosPlate-dr17/tablas/tipos_y_rangos.csv", "specObj-dr17/tablas/tipos_y_rangos.csv", "photoPosPlate-dr17/tablas/nulos.csv", "specObj-dr17/tablas/nulos.csv"],
        "Paso 3 - Distribuciones": ["photoPosPlate-dr17/tablas/estadisticas.csv", "specObj-dr17/tablas/estadisticas.csv", "photoPosPlate-dr17/graficas/histogramas.csv", "specObj-dr17/graficas/histogramas.csv"],
        "Paso 4 - Outliers": ["photoPosPlate-dr17/tablas/outliers.csv", "specObj-dr17/tablas/outliers.csv", "photoPosPlate-dr17/tablas/top_outliers.csv", "specObj-dr17/tablas/top_outliers.csv"],
        "Paso 5 - Bivariadas": ["photoPosPlate-dr17/tablas/top_correlaciones.csv", "specObj-dr17/tablas/top_correlaciones.csv", "photoPosPlate-dr17/graficas/scatterplots.csv", "specObj-dr17/graficas/scatterplots.csv"],
        "Paso 6 - Multivariadas": ["photoPosPlate-dr17/tablas/covarianza.csv", "specObj-dr17/tablas/covarianza.csv", "photoPosPlate-dr17/graficas/heatmap_correlacion.csv", "specObj-dr17/graficas/heatmap_correlacion.csv"],
        "Paso 8 - Balance": ["specObj-dr17/tablas/distribucion_clases.csv", "specObj-dr17/tablas/value_counts_top_CLASS.csv"],
        "Paso 10 - Conclusiones": ["../hipotesis_csv/indice_resultados_hipotesis.csv"],
    }
    for title, files in mapping.items():
        with st.expander(title, expanded=title.startswith("Paso 0")):
            for file in files:
                path = (ANALYSIS_DIR / file).resolve() if not file.startswith("../") else (ANALYSIS_DIR / file).resolve()
                if "../hipotesis_csv" in file:
                    path = CSV_DIR / "indice_resultados_hipotesis.csv"
                    df = load_csv("indice_resultados_hipotesis.csv")
                else:
                    df = load_analysis_csv(file)
                show_table(df, file, height=260)


def main():
    st.title("Dashboard SDSS interactivo desde CSV")
    st.caption("Los CSV se conservan como resultados reproducibles; esta app los lee y genera graficas en tiempo real.")

    with st.sidebar:
        st.header("Fuente")
        st.write(CSV_DIR)
        if st.button("Recargar CSV"):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        page = st.radio(
            "Seccion",
            [
                "H1 Redshift",
                "H2 Clasificacion",
                "H3 Correlacion",
                "H4 Calidad",
                "H5 Colores",
                "Pasos EDA",
            ],
        )

    if page == "H1 Redshift":
        h1()
    elif page == "H2 Clasificacion":
        h2()
    elif page == "H3 Correlacion":
        h3()
    elif page == "H4 Calidad":
        h4()
    elif page == "H5 Colores":
        h5()
    else:
        pasos()


if __name__ == "__main__":
    main()
