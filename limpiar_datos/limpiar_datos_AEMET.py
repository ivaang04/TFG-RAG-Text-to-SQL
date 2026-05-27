import pandas as pd
import os
import glob

directorio_actual = os.path.dirname(os.path.abspath(__file__))

ruta_datos = os.path.join(directorio_actual, 'datos')

ruta_aemet = os.path.join(ruta_datos, 'AEMET_clean')
ruta_salida = os.path.join(ruta_datos, 'AEMET_unificados')

os.makedirs(ruta_salida, exist_ok=True) 

patron_archivos = os.path.join(ruta_aemet, '*1.csv')
archivos = glob.glob(patron_archivos)

print(f"Buscando archivos en: {ruta_aemet}")
print(f"Archivos encontrados: {len(archivos)}\n")

dfs = []
for ruta_archivo in archivos:
    # Leemos saltando las 4 primeras líneas (ignoramos metadatos) y usando ';' como separador
    df = pd.read_csv(ruta_archivo, sep=';', skiprows=4)
    df.columns = df.columns.str.strip()
    
    # Extraemos solo el nombre del archivo
    nombre_archivo = os.path.basename(ruta_archivo)
    
    # Limpiamos para quedarnos con el nombre de la estación
    nombre_estacion = nombre_archivo.replace('1.csv', '').replace('MS_', '').strip()
    nombre_estacion = nombre_estacion.replace(' ', '_')  # Reemplazamos espacios por guiones bajos
    # Insertamos la columna
    df.insert(0, 'Estacion', nombre_estacion)
    dfs.append(df)

# Juntamos todos los dataframes
df_final = pd.concat(dfs, ignore_index=True)

ruta_final = os.path.join(ruta_salida, 'AEMET_unificada_limpia.csv')
df_final.to_csv(ruta_final, index=False)

print(f"Proceso terminado. Se han unido {len(archivos)} estaciones.")
print(f"Total de filas: {len(df_final)}")
print(f" Archivo guardado con éxito en:\n{ruta_final}")