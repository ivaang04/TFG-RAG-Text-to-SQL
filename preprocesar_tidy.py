# preprocesar_tidy.py
import re
import pandas as pd

ID_VARS = ["Estacion", "Lugar", "Date"]

MAPA_VARIABLES = {
    "PH": "pH",
    "Organic matter_(mg/L total organic C)": "Organic_matter_mgL",
    "Temperature_(°C)": "Temperature_C",
    "Salinity": "Salinity",
    "Conductivity_(S/m)": "Conductivity_Sm",
    "Dissolved oxygen_(mg/L)": "Dissolved_oxygen_mgL",
    "Chlorophyll_(mg/m3)": "Chlorophyll_mgm3",
    "Turbidity_(FTU)": "Turbidity_FTU",
    "PE_(RFU)": "PE_RFU",
}

PATRON = re.compile(r"^Depth_(-?\d+(?:\.\d+)?)m_(.+)$")


def tidy_boyas(ruta_in, ruta_out):
    df = pd.read_csv(ruta_in)

    # 1. Ancho -> largo: una fila por (id, columna_original, valor)
    largo = df.melt(id_vars=ID_VARS, var_name="col", value_name="valor")

    # 2. Separar profundidad y variable del nombre de la columna
    ext = largo["col"].str.extract(PATRON)
    largo["Depth"] = ext[0].astype(float)
    largo["Variable"] = ext[1]

    # Aviso si alguna columna no encaja con el patrón (no se perderá en silencio)
    sin_patron = largo.loc[largo["Variable"].isna(), "col"].unique()
    if len(sin_patron):
        print(f"AVISO: columnas ignoradas (no encajan el patrón): {list(sin_patron)}")
    largo = largo.dropna(subset=["Variable"]).drop(columns="col")

    # 3. Normalizar nombres de variable a algo limpio
    no_mapeadas = set(largo["Variable"]) - set(MAPA_VARIABLES)
    if no_mapeadas:
        print(f"AVISO: variables sin mapeo (se dejan tal cual): {no_mapeadas}")
    largo["Variable"] = largo["Variable"].map(MAPA_VARIABLES).fillna(largo["Variable"])

    # 4. Largo -> tidy: una columna por variable física
    tidy = largo.pivot_table(
        index=ID_VARS + ["Depth"],
        columns="Variable",
        values="valor",
        aggfunc="first",   # ver nota sobre duplicados más abajo
    ).reset_index()
    tidy.columns.name = None

    # 5. Date como fecha real (para que EXTRACT(YEAR ...) funcione en DuckDB)
    tidy["Date"] = pd.to_datetime(tidy["Date"], errors="coerce")

    # 6. Quitar filas-fantasma: (estación, fecha, profundidad) sin ninguna medida
    medidas = [c for c in tidy.columns if c not in ID_VARS + ["Depth"]]
    tidy = tidy.dropna(subset=medidas, how="all")

    # 7. Orden de columnas legible
    tidy = tidy[ID_VARS + ["Depth"] + sorted(medidas)]
    tidy = tidy.sort_values(ID_VARS + ["Depth"]).reset_index(drop=True)

    tidy.to_csv(ruta_out, index=False)
    print(f"Tidy escrito en {ruta_out}: {tidy.shape[0]} filas, "
          f"{tidy.shape[1]} columnas.")
    return tidy


if __name__ == "__main__":
    base = "/home/ivan/Universidad/TFG/datos/datos_procesados/"
    tidy_boyas(
        base + "BoyasProf_IMIDA_UPCT_unificadas_limpias.csv",
        base + "BoyasProf_IMIDA_UPCT_tidy.csv",
    )