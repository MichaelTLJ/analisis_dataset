from pathlib import Path
import html

from dashboard_hipotesis import build_payload


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
REPORT_MD = OUTPUT_DIR / "informe_final_eda.md"
REPORT_HTML = OUTPUT_DIR / "informe_final_eda.html"


def fmt(value):
    if value is None:
        return "n/d"
    try:
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def table(rows, headers):
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def build_markdown(payload):
    m = payload["metrics"]
    class_counts = payload["classCounts"]
    total = sum(class_counts.values())
    balance_rows = [
        {"Clase": cls, "Cantidad": f"{count:,}", "Porcentaje": f"{count / total * 100:.2f}%"}
        for cls, count in class_counts.items()
    ]
    corr_rows = []
    for name, value in m["corr_global"].items():
        corr_rows.append({"Grupo": "global", "Par": name, "Correlacion": fmt(value)})
    for name, value in m["corr_conditioned"].items():
        corr_rows.append({"Grupo": "condicionado", "Par": name, "Correlacion": fmt(value)})

    ranking_rows = [
        {"Variable": row["feature"], "eta2_clase": fmt(row["eta2_class"]), "corr_Z": fmt(row["corr_z"])}
        for row in payload["ranking"][:12]
    ]

    return f"""# INFORME FINAL DE ANALISIS EXPLORATORIO DE DATOS DEL CONJUNTO SDSS DR17

## Hipotesis iniciales

### Motivacion
Las hipotesis se originan en un problema central de astrofisica computacional: aprovechar mediciones fotometricas mas baratas para inferir propiedades espectroscopicas, clasificar objetos astronomicos y entender sesgos de medicion. El proyecto usa solo archivos locales del proyecto: `data/photoPosPlate-dr17.fits`, `data/specObj-dr17.fits` y los CSV procesados en `output/analisis`.

### Hipotesis en forma de pregunta
- **Hipotesis 1:** ¿Es posible estimar el redshift espectroscopico `Z` usando magnitudes e indices de color fotometricos?
- **Hipotesis 2:** ¿Clasifican mejor las variables de luz (`*FLUX`, `*MAG`) o las variables morfologicas (`PETROTHETA`, `M_E1`, `M_E2`, `FRACDEV`)?
- **Hipotesis 3:** ¿La correlacion global entre metodos de flujo se rompe al condicionar por morfologias complejas?
- **Hipotesis 4:** ¿Los errores fisicos y la degradacion de calidad dependen de `AIRMASS`, `EXTINCTION`, `SN_MEDIAN_ALL` y redshift?
- **Hipotesis 5:** ¿Los indices de color separan clases mejor que las magnitudes o flujos univariados?

## Plan de analisis
1. Combinar por posicion los FITS locales `photoPosPlate-dr17` y `specObj-dr17`, verificando alineacion `OBJID` vs `BESTOBJID`.
2. Usar `CLASS` y `Z` de `specObj` como variables objetivo.
3. Usar magnitudes, flujos, morfologia, condiciones instrumentales y calidad desde `photoPosPlate`.
4. Aplicar limpieza pendiente: tratar `-9999` y ceros fisicamente invalidos en magnitudes/AIRMASS/PETROTHETA como faltantes.
5. Generar dashboard interactivo con scatterplots, histogramas, ranking de variables, correlaciones condicionadas y PCA.

## Fuente de datos
**Fuente:** Sloan Digital Sky Survey DR17, archivos FITS locales del proyecto.

Los datos provienen de observaciones astronomicas realizadas por la colaboracion SDSS mediante imagenes fotometricas multibanda y espectroscopia. El dominio es astronomia observacional, astrofisica y ciencia de datos. El problema computacional es explorar si variables fotometricas y morfologicas permiten estimar redshift, clasificar objetos y detectar sesgos de calidad.

Referencias base: York et al. (2000), Blanton et al. (2017), Abdurro'uf et al. (2022), documentacion oficial SDSS DR17 Data Model.

## Descripcion

### A nivel de atributos
Los atributos principales son:

{table([
{"Atributo": "Z", "Significado": "redshift espectroscopico", "Uso": "target de regresion H1", "Tipo": "continuo", "Unidad": "adimensional"},
{"Atributo": "CLASS", "Significado": "STAR/GALAXY/QSO", "Uso": "target de clasificacion H2/H5", "Tipo": "categorico nominal", "Unidad": "sin unidad"},
{"Atributo": "PSFMAG, CMODELMAG, MODELMAG, PETROMAG", "Significado": "magnitudes por filtros u,g,r,i,z", "Uso": "fotometria y colores", "Tipo": "continuo", "Unidad": "magnitud"},
{"Atributo": "*FLUX", "Significado": "flujos por metodo de medicion", "Uso": "comparacion de algoritmos H3", "Tipo": "continuo", "Unidad": "flujo/nanomaggies"},
{"Atributo": "PETROTHETA, M_E1, M_E2, FRACDEV", "Significado": "forma y morfologia", "Uso": "clasificacion morfologica H2/H3", "Tipo": "continuo", "Unidad": "varias"},
{"Atributo": "AIRMASS, EXTINCTION, SN_MEDIAN_ALL", "Significado": "condicion observacional y calidad", "Uso": "sesgo/calidad H4", "Tipo": "continuo", "Unidad": "varias"},
], ["Atributo", "Significado", "Uso", "Tipo", "Unidad"])}

### A nivel de registros
Cada registro representa un objeto astronomico observado por SDSS. Al combinar ambos FITS por posicion, cada fila contiene fotometria, morfologia, calidad instrumental, clase astronomica y redshift. Las etiquetas `STAR`, `GALAXY` y `QSO` diferencian estrellas locales, galaxias y cuasares.

Balance del dataset completo:

{table(balance_rows, ["Clase", "Cantidad", "Porcentaje"])}

## Relacion entre atributos
El dashboard evalua correlacion global, correlacion condicionada, relacion color-redshift, relaciones luz-forma y estructura PCA. Ranking exploratorio:

{table(ranking_rows, ["Variable", "eta2_clase", "corr_Z"])}

Correlacion global vs condicionada para flujos:

{table(corr_rows, ["Grupo", "Par", "Correlacion"])}

## Formato
El formato original es FITS. Las salidas procesadas se guardan en CSV dentro de `output/analisis/<dataset>/tablas` y `output/analisis/<dataset>/graficas`. Las graficas ya no se almacenan como imagenes, sino como CSV o como visualizaciones SVG interactivas dentro del dashboard.

## Transformaciones
- Expansion previa de columnas vectoriales.
- Union por posicion de `photoPosPlate` y `specObj`.
- Construccion de colores `PSFMAG_ug`, `PSFMAG_gr`, `PSFMAG_ri`, etc.
- Construccion de elipticidad `sqrt(M_E1_r^2 + M_E2_r^2)`.
- Construccion de bandera `PHOTO_ERROR_FLAG`.
- PCA exploratorio para estructura multidimensional.

## Limpieza de datos
El data wrangling general ya fue realizado. En esta etapa se aplico limpieza focalizada:
- `-9999` se trata como faltante fisico.
- Magnitudes `<= 0` se tratan como mediciones invalidas para el analisis fotometrico.
- `AIRMASS <= 0` y `PETROTHETA <= 0` se tratan como invalidos.
- Outliers se conservan para analisis astronomico salvo justificacion posterior.

## Exploracion
El dashboard `output/dashboard.html` contiene las visualizaciones principales:
1. Redshift vs color fotometrico para H1.
2. Distribucion de redshift por clase.
3. Scatter de variables seleccionables para comparar luz y morfologia.
4. Correlacion global vs condicionada entre metodos de flujo.
5. Tasa de error fisico por AIRMASS/EXTINCTION/Z.
6. Color-color diagram para H5.
7. PCA multidimensional.

## Conclusion
### Hipotesis 1
R2 lineal usando colores para predecir Z: **{fmt(m["photoz_r2_colors"])}**. R2 usando magnitudes + colores: **{fmt(m["photoz_r2_mags_colors"])}**. Esto funciona como baseline; si es moderado, justifica modelos no lineales.

### Hipotesis 2
Accuracy exploratoria por centroides: luz **{fmt(m["class_acc_light"])}**, morfologia **{fmt(m["class_acc_morph"])}**, colores **{fmt(m["class_acc_colors"])}**, combinada **{fmt(m["class_acc_multi"])}**. La comparacion orienta que familia de variables aporta mas.

### Hipotesis 3
La correlacion global PETRO vs DEV r es **{fmt(m["corr_global"]["PETRO_vs_DEV_r"])}** y condicionada en objetos extendidos es **{fmt(m["corr_conditioned"]["extended_PETRO_vs_DEV_r"])}**. Si cambia, la correlacion global oculta subpoblaciones morfologicas.

### Hipotesis 4
Correlaciones de error fisico: AIRMASS **{fmt(m["quality_corr_airmass_error"])}**, EXTINCTION **{fmt(m["quality_corr_extinction_error"])}**, Z **{fmt(m["quality_corr_z_error"])}**, SN **{fmt(m["quality_corr_sn_error"])}**. Esto evalua si los errores no son aleatorios.

### Hipotesis 5
La separabilidad mejora al transformar magnitudes en colores porque se reduce varianza compartida de brillo absoluto y se destacan diferencias espectrales. El dashboard permite comparar histogramas crudos contra diagramas color-color.

## Anexos
- Dashboard: `output/dashboard.html`.
- Analisis procesado: `output/analisis`.
- Codigo principal: `programa.py`, `dashboard_hipotesis.py`.

## Referencias
- York et al. (2000). The Sloan Digital Sky Survey: Technical Summary.
- Blanton et al. (2017). Sloan Digital Sky Survey IV.
- Abdurro'uf et al. (2022). The Seventeenth Data Release of the Sloan Digital Sky Surveys.
- SDSS DR17 Data Model.
"""


def markdown_to_html(md):
    lines = []
    for raw in md.splitlines():
        line = raw.strip()
        if line.startswith("# "):
            lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("| "):
            lines.append(f"<pre>{html.escape(line)}</pre>")
        elif line.startswith("- "):
            lines.append(f"<p>{html.escape(line)}</p>")
        elif line:
            lines.append(f"<p>{html.escape(line)}</p>")
    return "<!doctype html><meta charset='utf-8'><style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1050px;margin:auto;padding:28px;line-height:1.55}pre{overflow:auto;background:#f6f7fb;padding:4px}</style>" + "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    payload = build_payload()
    md = build_markdown(payload)
    REPORT_MD.write_text(md, encoding="utf-8")
    REPORT_HTML.write_text(markdown_to_html(md), encoding="utf-8")
    print(f"Informe actualizado: {REPORT_MD}")
    print(f"Informe HTML actualizado: {REPORT_HTML}")


if __name__ == "__main__":
    main()
