from pathlib import Path
import html

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from dashboard_hipotesis import STEP_TABLES, OUTPUT_DIR, HYPOTHESIS_DIR


BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = OUTPUT_DIR / "dashboard.html"
IMAGE_DIR = OUTPUT_DIR / "hipotesis_img"

COLORS = {"STAR": "#2878b5", "GALAXY": "#2f9e44", "QSO": "#d64545"}


def read_csv(path, **kwargs):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def rel_output(path):
    try:
        return str(path.relative_to(OUTPUT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def fig_html(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=45, r=20, t=55, b=45),
        legend_title_text="Clase",
        hovermode="closest",
        font=dict(family="Segoe UI, Arial, sans-serif"),
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})


def save_bar_image(df, x, y, title, filename, color="#3156d4"):
    if df.empty or x not in df or y not in df:
        return ""
    path = IMAGE_DIR / filename
    plt.figure(figsize=(9, 5))
    plt.bar(df[x].astype(str), df[y], color=color)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return rel_output(path)


def save_scatter_image(df, x, y, title, filename):
    if df.empty or x not in df or y not in df:
        return ""
    path = IMAGE_DIR / filename
    plt.figure(figsize=(8, 6))
    for cls, group in df.groupby("CLASS"):
        plt.scatter(group[x], group[y], s=8, alpha=0.45, label=cls, color=COLORS.get(cls))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return rel_output(path)


def table_html(df, title, source="", max_rows=12):
    if df.empty:
        return f"<section class='data-card'><h4>{html.escape(title)}</h4><p class='muted'>Tabla no disponible.</p></section>"
    preview = df.head(max_rows).copy()
    for col in preview.columns:
        if pd.api.types.is_numeric_dtype(preview[col]):
            preview[col] = preview[col].map(lambda x: "" if pd.isna(x) else f"{x:.5g}")
    source_html = f"<a href='{html.escape(source)}'>CSV</a>" if source else ""
    return (
        "<section class='data-card'>"
        f"<div class='table-head'><h4>{html.escape(title)}</h4>{source_html}</div>"
        "<input class='table-filter' placeholder='Filtrar esta tabla...' oninput='filterTable(this)'>"
        f"<div class='table-wrap'>{preview.to_html(index=False, escape=True)}</div>"
        f"<p class='muted'>{len(df):,} filas totales. Se muestran {min(len(df), max_rows)}.</p>"
        "</section>"
    )


def csv_card(row):
    return (
        "<article class='csv-card'>"
        f"<strong>{html.escape(str(row.get('titulo', 'CSV')))}</strong>"
        f"<span>{html.escape(str(row.get('hipotesis', '')))} - {int(row.get('filas', 0)):,} filas</span>"
        f"<p>{html.escape(str(row.get('descripcion', '')))}</p>"
        f"<a href='{html.escape(str(row.get('archivo', '')))}'>Abrir CSV</a>"
        "</article>"
    )


def step_tables_html():
    chunks = []
    for step, tables in STEP_TABLES.items():
        cards = []
        for title, path in tables:
            df = read_csv(path, nrows=12)
            cards.append(table_html(df, title, rel_output(path), max_rows=12))
        chunks.append(
            "<article class='step-block'>"
            f"<h3>{html.escape(step)}</h3>"
            "<div class='table-grid'>"
            + "".join(cards)
            + "</div></article>"
        )
    return "".join(chunks)


def make_dashboard():
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    h1_models = read_csv(HYPOTHESIS_DIR / "h1_redshift_modelos.csv")
    h1_scatter = read_csv(HYPOTHESIS_DIR / "h1_scatter_color_z.csv")
    h2_acc = read_csv(HYPOTHESIS_DIR / "h2_accuracy_grupos.csv")
    h2_pca = read_csv(HYPOTHESIS_DIR / "h2_pca_projection.csv")
    h2_rank = read_csv(HYPOTHESIS_DIR / "h2_ranking_variables.csv")
    h3_corr = read_csv(HYPOTHESIS_DIR / "h3_correlaciones_condicionadas.csv")
    h3_scatter = read_csv(HYPOTHESIS_DIR / "h3_scatter_flujos.csv")
    h4_bins = read_csv(HYPOTHESIS_DIR / "h4_calidad_bins.csv")
    h4_corr = read_csv(HYPOTHESIS_DIR / "h4_correlaciones_sesgo.csv")
    h5_scatter = read_csv(HYPOTHESIS_DIR / "h5_color_color_scatter.csv")
    h5_hist = read_csv(HYPOTHESIS_DIR / "h5_histogramas_colores.csv")
    csv_index = read_csv(HYPOTHESIS_DIR / "indice_resultados_hipotesis.csv")

    images = {
        "h1_scatter": save_scatter_image(h1_scatter, "PSFMAG_ug", "Z", "H1 color u-g vs redshift", "h1_color_vs_z.png"),
        "h1_bar": save_bar_image(h1_models, "modelo", "r2", "H1 R2 modelos exploratorios", "h1_r2_modelos.png"),
        "h2_bar": save_bar_image(h2_acc, "grupo", "accuracy_centroides", "H2 accuracy por familia", "h2_accuracy.png"),
        "h2_pca": save_scatter_image(h2_pca, "PC1", "PC2", "H2 PCA multivariado", "h2_pca.png"),
        "h3_scatter": save_scatter_image(h3_scatter, "PETROFLUX_r", "DEVFLUX_r", "H3 PETROFLUX vs DEVFLUX", "h3_flujos.png"),
        "h3_bar": save_bar_image(h3_corr, "par", "correlacion", "H3 correlaciones condicionadas", "h3_correlaciones.png"),
        "h4_bar": save_bar_image(h4_corr, "variable", "correlacion_error_fisico", "H4 correlacion con error fisico", "h4_sesgo.png"),
        "h5_scatter": save_scatter_image(h5_scatter, "PSFMAG_ug", "PSFMAG_gr", "H5 color-color", "h5_color_color.png"),
    }

    h1_fig1 = px.scatter(
        h1_scatter,
        x="PSFMAG_ug",
        y="Z",
        color="CLASS",
        color_discrete_map=COLORS,
        title="H1: Color fotometrico u-g vs redshift espectroscopico",
        opacity=0.65,
        render_mode="webgl",
    )
    h1_fig2 = px.bar(h1_models, x="modelo", y="r2", text="r2", title="H1: baseline de regresion lineal para Z")

    h2_fig1 = px.bar(h2_acc, x="grupo", y="accuracy_centroides", text="accuracy_centroides", title="H2: comparacion de familias de features")
    h2_fig2 = px.scatter(
        h2_pca,
        x="PC1",
        y="PC2",
        color="CLASS",
        color_discrete_map=COLORS,
        title="H2: proyeccion PCA de variables combinadas",
        opacity=0.7,
        render_mode="webgl",
    )

    h3_fig1 = px.bar(h3_corr, x="par", y="correlacion", color="grupo", title="H3: correlacion global vs condicionada")
    h3_fig2 = px.scatter(
        h3_scatter,
        x="PETROFLUX_r",
        y="DEVFLUX_r",
        color="CLASS",
        color_discrete_map=COLORS,
        title="H3: metodos de flujo en banda r",
        opacity=0.55,
        render_mode="webgl",
    )

    h4_fig1 = px.line(
        h4_bins,
        x="x",
        y="error_rate",
        color="variable",
        markers=True,
        title="H4: tasa de error fisico por bins de calidad/redshift",
    )
    h4_fig2 = px.bar(h4_corr, x="variable", y="correlacion_error_fisico", title="H4: correlacion con errores fisicos")

    h5_fig1 = px.scatter(
        h5_scatter,
        x="PSFMAG_ug",
        y="PSFMAG_gr",
        color="CLASS",
        color_discrete_map=COLORS,
        title="H5: espacio color-color",
        opacity=0.65,
        render_mode="webgl",
    )
    h5_filter = h5_hist[h5_hist["feature"].isin(["PSFMAG_r", "PSFMAG_ug", "PSFMAG_gr"])]
    h5_fig2 = px.line(
        h5_filter,
        x="bin_center",
        y="count",
        color="class",
        line_dash="feature",
        color_discrete_map=COLORS,
        title="H5: histogramas interactivos de magnitud cruda y colores",
    )

    sections = {
        "h1": [
            fig_html(h1_fig1),
            fig_html(h1_fig2),
            table_html(h1_models, "Resultados numericos H1", "hipotesis_csv/h1_redshift_modelos.csv"),
            table_html(h1_scatter, "Muestra usada para color vs Z", "hipotesis_csv/h1_scatter_color_z.csv"),
        ],
        "h2": [
            fig_html(h2_fig1),
            fig_html(h2_fig2),
            table_html(h2_acc, "Accuracy por grupo de variables", "hipotesis_csv/h2_accuracy_grupos.csv"),
            table_html(h2_rank, "Ranking de variables", "hipotesis_csv/h2_ranking_variables.csv"),
        ],
        "h3": [
            fig_html(h3_fig1),
            fig_html(h3_fig2),
            table_html(h3_corr, "Correlaciones condicionadas", "hipotesis_csv/h3_correlaciones_condicionadas.csv"),
            table_html(h3_scatter, "Muestra de flujos", "hipotesis_csv/h3_scatter_flujos.csv"),
        ],
        "h4": [
            fig_html(h4_fig1),
            fig_html(h4_fig2),
            table_html(h4_bins, "Bins de calidad", "hipotesis_csv/h4_calidad_bins.csv"),
            table_html(h4_corr, "Correlaciones de sesgo", "hipotesis_csv/h4_correlaciones_sesgo.csv"),
        ],
        "h5": [
            fig_html(h5_fig1),
            fig_html(h5_fig2),
            table_html(h5_scatter, "Muestra color-color", "hipotesis_csv/h5_color_color_scatter.csv"),
            table_html(h5_hist, "Histogramas por clase", "hipotesis_csv/h5_histogramas_colores.csv"),
        ],
    }

    image_cards = "".join(
        f"<figure><img src='{src}' alt='{name}'><figcaption>{name}</figcaption></figure>"
        for name, src in images.items()
        if src
    )

    html_doc = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard interactivo SDSS</title>
<script>{get_plotlyjs()}</script>
<style>
body{{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#eef2f8;color:#172033;line-height:1.5}}
header{{background:#101828;color:#fff;padding:24px clamp(16px,4vw,56px);position:sticky;top:0;z-index:5;box-shadow:0 8px 24px #0002}}
h1{{margin:0;font-size:clamp(26px,4vw,44px)}} header p{{max-width:1100px;color:#d0d5dd}}
main{{max-width:1560px;margin:auto;padding:18px clamp(10px,3vw,34px) 70px}}
.tabs{{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}}button{{border:1px solid #475467;background:#fff;color:#172033;border-radius:8px;padding:10px 12px;cursor:pointer;font-weight:700}}button.active{{background:#5b7cfa;color:#fff;border-color:#5b7cfa}}
.panel{{background:#fff;border:1px solid #d8dee9;border-radius:12px;padding:16px;margin:14px 0;box-shadow:0 8px 18px #1018280c}}
.view{{display:none}}.view.active{{display:block}}.graph-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.full{{grid-column:1/-1}}
.data-grid,.table-grid,.csv-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}
.data-card,.csv-card,.step-block{{background:#fbfcff;border:1px solid #d8dee9;border-radius:10px;padding:12px;overflow:hidden}}
.table-head{{display:flex;justify-content:space-between;gap:10px;align-items:center}}.table-head h4{{margin:0 0 8px}}a{{color:#3156d4;text-decoration:none;font-weight:700}}
.table-filter{{width:100%;box-sizing:border-box;padding:9px;border:1px solid #cfd8ea;border-radius:8px;margin:8px 0}}.table-wrap{{max-height:360px;overflow:auto;border:1px solid #e5eaf3;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #edf0f5;text-align:left;white-space:nowrap}}th{{background:#f2f5fb;position:sticky;top:0}}
.muted{{color:#667085;font-size:13px}}.csv-card span{{display:block;color:#667085;font-size:13px;margin-top:3px}}
.img-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}}figure{{margin:0;background:#fff;border:1px solid #d8dee9;border-radius:10px;padding:10px}}img{{max-width:100%;height:auto;display:block}}figcaption{{font-size:13px;color:#667085;margin-top:6px}}
@media(max-width:980px){{.graph-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
<h1>Dashboard interactivo SDSS: hipotesis, tablas y graficos</h1>
<p>Este tablero usa los CSV de resultados generados en el proyecto, conserva los archivos exportados y muestra tablas y graficos interactivos directamente en la pagina. Las imagenes PNG quedan guardadas en <code>output/hipotesis_img</code>.</p>
<nav class="tabs">
<button class="active" data-view="h1">H1 Redshift</button>
<button data-view="h2">H2 Clasificacion</button>
<button data-view="h3">H3 Correlacion</button>
<button data-view="h4">H4 Calidad</button>
<button data-view="h5">H5 Colores</button>
<button data-view="steps">Pasos 0-11</button>
<button data-view="exports">CSV e imagenes</button>
</nav>
</header>
<main>
<section id="h1" class="view active"><div class="panel"><h2>H1. Estimacion de redshift fotometrico</h2><p>La evidencia se muestra con scatter interactivo color-redshift y una tabla de R2.</p><div class="graph-grid">{''.join(sections['h1'][:2])}</div><div class="data-grid">{''.join(sections['h1'][2:])}</div></div></section>
<section id="h2" class="view"><div class="panel"><h2>H2. Clasificacion morfologica vs fotometrica</h2><p>Se comparan familias de variables y se visualiza PCA para separabilidad multivariada.</p><div class="graph-grid">{''.join(sections['h2'][:2])}</div><div class="data-grid">{''.join(sections['h2'][2:])}</div></div></section>
<section id="h3" class="view"><div class="panel"><h2>H3. Correlacion condicionada</h2><p>La correlacion global se contrasta contra subpoblaciones extendidas o elipticas.</p><div class="graph-grid">{''.join(sections['h3'][:2])}</div><div class="data-grid">{''.join(sections['h3'][2:])}</div></div></section>
<section id="h4" class="view"><div class="panel"><h2>H4. Sesgo instrumental y calidad</h2><p>Los errores fisicos se analizan como senal de calidad y sesgo, no solo como basura.</p><div class="graph-grid">{''.join(sections['h4'][:2])}</div><div class="data-grid">{''.join(sections['h4'][2:])}</div></div></section>
<section id="h5" class="view"><div class="panel"><h2>H5. Feature engineering con colores</h2><p>Los indices de color transforman magnitudes en un espacio mas separable.</p><div class="graph-grid">{''.join(sections['h5'][:2])}</div><div class="data-grid">{''.join(sections['h5'][2:])}</div></div></section>
<section id="steps" class="view"><div class="panel"><h2>Pasos EDA 0-11 con tablas del analisis</h2>{step_tables_html()}</div></section>
<section id="exports" class="view"><div class="panel"><h2>CSV generados</h2><div class="csv-grid">{''.join(csv_card(row) for _, row in csv_index.iterrows())}</div></div><div class="panel"><h2>Imagenes PNG guardadas</h2><div class="img-grid">{image_cards}</div></div></section>
</main>
<script>
document.querySelectorAll("button[data-view]").forEach(btn=>btn.onclick=()=>{{
  document.querySelectorAll("button[data-view]").forEach(b=>b.classList.toggle("active", b===btn));
  document.querySelectorAll(".view").forEach(v=>v.classList.toggle("active", v.id===btn.dataset.view));
  setTimeout(()=>window.dispatchEvent(new Event("resize")), 60);
}});
function filterTable(input){{
  const term=input.value.toLowerCase();
  const rows=input.closest(".data-card").querySelectorAll("tbody tr");
  rows.forEach(row=>row.style.display=row.innerText.toLowerCase().includes(term)?"":"none");
}}
</script>
</body>
</html>"""

    DASHBOARD_PATH.write_text(html_doc, encoding="utf-8")
    return DASHBOARD_PATH


def main():
    path = make_dashboard()
    print(f"Dashboard avanzado generado: {path}")
    print(f"Imagenes guardadas en: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
