# INFORME FINAL DE ANALISIS EXPLORATORIO DE DATOS DEL CONJUNTO SDSS DR17

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

| Atributo | Significado | Uso | Tipo | Unidad |
| --- | --- | --- | --- | --- |
| Z | redshift espectroscopico | target de regresion H1 | continuo | adimensional |
| CLASS | STAR/GALAXY/QSO | target de clasificacion H2/H5 | categorico nominal | sin unidad |
| PSFMAG, CMODELMAG, MODELMAG, PETROMAG | magnitudes por filtros u,g,r,i,z | fotometria y colores | continuo | magnitud |
| *FLUX | flujos por metodo de medicion | comparacion de algoritmos H3 | continuo | flujo/nanomaggies |
| PETROTHETA, M_E1, M_E2, FRACDEV | forma y morfologia | clasificacion morfologica H2/H3 | continuo | varias |
| AIRMASS, EXTINCTION, SN_MEDIAN_ALL | condicion observacional y calidad | sesgo/calidad H4 | continuo | varias |

### A nivel de registros
Cada registro representa un objeto astronomico observado por SDSS. Al combinar ambos FITS por posicion, cada fila contiene fotometria, morfologia, calidad instrumental, clase astronomica y redshift. Las etiquetas `STAR`, `GALAXY` y `QSO` diferencian estrellas locales, galaxias y cuasares.

Balance del dataset completo:

| Clase | Cantidad | Porcentaje |
| --- | --- | --- |
| STAR | 1,192,886 | 20.56% |
| GALAXY | 3,237,535 | 55.81% |
| QSO | 1,370,779 | 23.63% |

## Relacion entre atributos
El dashboard evalua correlacion global, correlacion condicionada, relacion color-redshift, relaciones luz-forma y estructura PCA. Ranking exploratorio:

| Variable | eta2_clase | corr_Z |
| --- | --- | --- |
| Z | 0.575 | 1 |
| PSFMAG_gr | 0.338 | -0.261 |
| CMODELMAG_gr | 0.255 | -0.238 |
| PSFMAG_z | 0.236 | 0.459 |
| PSFMAG_i | 0.233 | 0.435 |
| PSFMAG_g | 0.222 | 0.229 |
| SN_MEDIAN_ALL | 0.222 | -0.319 |
| CMODELMAG_z | 0.216 | 0.484 |
| CMODELMAG_i | 0.204 | 0.487 |
| PSFMAG_u | 0.2 | 0.094 |
| PSFMAG_r | 0.2 | 0.35 |
| PSFMAG_ri | 0.191 | -0.197 |

Correlacion global vs condicionada para flujos:

| Grupo | Par | Correlacion |
| --- | --- | --- |
| global | PETRO_vs_DEV_r | 0.728 |
| global | PETRO_vs_EXP_r | 0.809 |
| global | PETRO_vs_CMODEL_r | 0.787 |
| condicionado | extended_PETRO_vs_DEV_r | 0.741 |
| condicionado | extended_PETRO_vs_EXP_r | 0.812 |
| condicionado | elliptical_PETRO_vs_DEV_r | 0.927 |
| condicionado | elliptical_PETRO_vs_EXP_r | 0.995 |

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
R2 lineal usando colores para predecir Z: **0.136**. R2 usando magnitudes + colores: **0.301**. Esto funciona como baseline; si es moderado, justifica modelos no lineales.

### Hipotesis 2
Accuracy exploratoria por centroides: luz **0.612**, morfologia **0.561**, colores **0.621**, combinada **0.713**. La comparacion orienta que familia de variables aporta mas.

### Hipotesis 3
La correlacion global PETRO vs DEV r es **0.728** y condicionada en objetos extendidos es **0.741**. Si cambia, la correlacion global oculta subpoblaciones morfologicas.

### Hipotesis 4
Correlaciones de error fisico: AIRMASS **n/d**, EXTINCTION **-0.159**, Z **0.181**, SN **-0.215**. Esto evalua si los errores no son aleatorios.

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
