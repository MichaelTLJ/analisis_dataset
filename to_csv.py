from astropy.io import fits
import pandas as pd
import numpy as np

# =====================================================
# RUTA DEL FITS
# =====================================================

FITS_PATH = r"data\specObj-dr17.fits"

# =====================================================
# ABRIR FITS
# =====================================================

print("Abriendo FITS...")

hdul = fits.open(
    FITS_PATH,
    memmap=True
)

data = hdul[1].data

print("Dataset cargado")

print("Filas:", len(data))
print("Columnas originales:", len(data.names))

# =====================================================
# DATAFRAME FINAL
# =====================================================

df_dict = {}

# =====================================================
# BANDAS
# =====================================================

bands_5 = ["u", "g", "r", "i", "z"]

bands_10 = [str(i) for i in range(10)]

bands_11 = [str(i) for i in range(1,11)]

# =====================================================
# EXPANDIR COLUMNAS
# =====================================================

for col in data.names:

    try:

        arr = data[col]

        # -------------------------------------------------
        # COLUMNAS NORMALES
        # -------------------------------------------------

        if len(arr.shape) == 1:

            df_dict[col] = arr

        # -------------------------------------------------
        # COLUMNAS DE 5 ELEMENTOS
        # -------------------------------------------------

        elif arr.shape[1] == 5:

            for i, band in enumerate(bands_5):

                new_col = f"{col}_{band}"

                df_dict[new_col] = arr[:, i]

        # -------------------------------------------------
        # COLUMNAS DE 10 ELEMENTOS
        # -------------------------------------------------

        elif arr.shape[1] == 10:

            for i, band in enumerate(bands_10):

                new_col = f"{col}_{band}"

                df_dict[new_col] = arr[:, i]

        # -------------------------------------------------
        # COLUMNAS DE 11 ELEMENTOS
        # -------------------------------------------------

        elif arr.shape[1] == 11:

            for i, band in enumerate(bands_11):

                new_col = f"{col}_{band}"

                df_dict[new_col] = arr[:, i]

        else:

            print(f"No expandida: {col} -> {arr.shape}")

    except Exception as e:

        print(f"Error en {col}: {e}")

# =====================================================
# CREAR DATAFRAME
# =====================================================

print("\nCreando DataFrame...")

df = pd.DataFrame(df_dict)

print("\nDataFrame creado")

print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])

# =====================================================
# LIMPIAR BYTES
# =====================================================

print("\nLimpiando strings...")

for col in df.columns:

    if df[col].dtype == object:

        try:

            df[col] = df[col].apply(
                lambda x:
                    x.decode("utf-8").strip()
                    if isinstance(x, bytes)
                    else x
            )

        except:
            pass

# =====================================================
# GUARDAR CSV
# =====================================================

print("\nGuardando CSV...")

df.to_csv(
    "specObj_expandido.csv",
    index=False
)

print("\nCSV generado correctamente")