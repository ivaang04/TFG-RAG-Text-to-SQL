import pandas as pd
import os
import glob

# Rutas
directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_datos = os.path.join(directorio_actual, 'datos')
ruta_boyas = os.path.join(ruta_datos, 'BoyasNautilus_clean') 
ruta_salida = os.path.join(ruta_datos, 'Boyas_Nautilus_unificadas')



os.makedirs(ruta_salida, exist_ok=True) 

# Búsqueda
patron_archivos = os.path.join(ruta_boyas, '*_buoy.csv')
archivos = glob.glob(patron_archivos)

print(f"Buscando archivos en: {ruta_boyas}")
print(f"Archivos encontrados: {len(archivos)}\n")

dfs = []
for ruta_archivo in archivos:
    df = pd.read_csv(ruta_archivo, sep=';', skiprows=4)
    df.columns = df.columns.str.strip()
    
    # Extraemos el nombre original (ej. "Cubeta Sur_buoy.csv" -> "Cubeta Sur")
    nombre_archivo = os.path.basename(ruta_archivo)
    nombre_boya = nombre_archivo.replace('_buoy.csv', '').strip()
    
    nombre_boya = nombre_boya.replace(' ', '_')
    
    # Insertamos la columna
    df.insert(0, 'Boya', nombre_boya)
    dfs.append(df)

# Concatenamos y guardamos
df_final = pd.concat(dfs, ignore_index=True)
ruta_final = os.path.join(ruta_salida, 'Boyas_Unificadas_Limpias.csv')
df_final.to_csv(ruta_final, index=False)

print(f"Proceso terminado. Se han unido {len(archivos)} boyas.")
print(f"Total de filas: {len(df_final)}")
print(f"Archivo guardado con éxito en:\n{ruta_final}")