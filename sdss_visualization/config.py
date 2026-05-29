from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output" / "visualizacion_embedding"
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

PHOTO_FITS = DATA_DIR / "photoPosPlate-dr17.fits"
SPEC_FITS = DATA_DIR / "specObj-dr17.fits"

BANDS = ["u", "g", "r", "i", "z"]
CLASS_ORDER = ["STAR", "GALAXY", "QSO"]
CLASS_COLORS = {
    "STAR": "#2563eb",
    "GALAXY": "#16a34a",
    "QSO": "#dc2626",
}

RANDOM_SEED = 42
MAX_PER_CLASS = 1200
MAX_EMBEDDING_VALUES_IN_PANEL = 24

FEATURE_COLUMNS = [
    "Z",
    "SN_MEDIAN_ALL",
    "PSFMAG_u",
    "PSFMAG_g",
    "PSFMAG_r",
    "PSFMAG_i",
    "PSFMAG_z",
    "CMODELMAG_u",
    "CMODELMAG_g",
    "CMODELMAG_r",
    "CMODELMAG_i",
    "CMODELMAG_z",
    "PSFFLUX_u",
    "PSFFLUX_g",
    "PSFFLUX_r",
    "PSFFLUX_i",
    "PSFFLUX_z",
    "CMODELFLUX_u",
    "CMODELFLUX_g",
    "CMODELFLUX_r",
    "CMODELFLUX_i",
    "CMODELFLUX_z",
    "PETROTHETA_r",
    "M_E1_r",
    "M_E2_r",
    "FRACDEV_r",
    "AIRMASS_r",
    "EXTINCTION_r",
    "PSFMAG_ug",
    "PSFMAG_gr",
    "PSFMAG_ri",
    "PSFMAG_iz",
    "ELLIPTICITY_R",
]

FEATURE_LABELS = {
    "Z": "Desplazamiento al rojo",
    "SN_MEDIAN_ALL": "Señal/ruido mediana",
    "PSFMAG_u": "Brillo aparente ultravioleta",
    "PSFMAG_g": "Brillo aparente verde",
    "PSFMAG_r": "Brillo aparente rojo",
    "PSFMAG_i": "Brillo aparente infrarrojo cercano",
    "PSFMAG_z": "Brillo aparente infrarrojo",
    "CMODELMAG_u": "Brillo de modelo ultravioleta",
    "CMODELMAG_g": "Brillo de modelo verde",
    "CMODELMAG_r": "Brillo de modelo rojo",
    "CMODELMAG_i": "Brillo de modelo infrarrojo cercano",
    "CMODELMAG_z": "Brillo de modelo infrarrojo",
    "PSFFLUX_u": "Flujo puntual ultravioleta",
    "PSFFLUX_g": "Flujo puntual verde",
    "PSFFLUX_r": "Flujo puntual rojo",
    "PSFFLUX_i": "Flujo puntual infrarrojo cercano",
    "PSFFLUX_z": "Flujo puntual infrarrojo",
    "CMODELFLUX_u": "Flujo total de modelo ultravioleta",
    "CMODELFLUX_g": "Flujo total de modelo verde",
    "CMODELFLUX_r": "Flujo total de modelo rojo",
    "CMODELFLUX_i": "Flujo total de modelo infrarrojo cercano",
    "CMODELFLUX_z": "Flujo total de modelo infrarrojo",
    "PETROTHETA_r": "Tamaño aparente rojo",
    "M_E1_r": "Forma horizontal/vertical",
    "M_E2_r": "Forma diagonal",
    "FRACDEV_r": "Perfil tipo galaxia elíptica",
    "AIRMASS_r": "Masa de aire observacional",
    "EXTINCTION_r": "Extinción atmosférica",
    "PSFMAG_ug": "Color ultravioleta-verde",
    "PSFMAG_gr": "Color verde-rojo",
    "PSFMAG_ri": "Color rojo-infrarrojo cercano",
    "PSFMAG_iz": "Color infrarrojo cercano-infrarrojo",
    "ELLIPTICITY_R": "Elipticidad aparente",
}
