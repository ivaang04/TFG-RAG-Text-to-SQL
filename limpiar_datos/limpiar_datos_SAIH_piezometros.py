import pandas as pd
import os
import glob

directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_datos = os.path.join(directorio_actual, 'datos')

nombre_carpeta_origen = 'SAIH_Piezometros_clean' 
ruta_origen = os.path.join(ruta_datos, nombre_carpeta_origen) 
ruta_salida = os.path.join(ruta_datos, 'SAIH_Piezometros_unificados')
os.makedirs(ruta_salida, exist_ok=True)

carpetas_estaciones = [d for d in os.listdir(ruta_origen) if os.path.isdir(os.path.join(ruta_origen, d))]
print(f"Encontradas {len(carpetas_estaciones)} carpetas de piezómetros.")

dfs_todas_estaciones = []

for estacion in carpetas_estaciones:
    ruta_estacion = os.path.join(ruta_origen, estacion)
    archivos_csv = glob.glob(os.path.join(ruta_estacion, "*.csv"))
    
    archivos_csv = [f for f in archivos_csv if 'metadata' not in f.lower() and 'metadato' not in f.lower()]
    if not archivos_csv:
        continue
        
    print(f"Procesando piezómetro '{estacion}' ({len(archivos_csv)} variables)")
    
    df_estacion = None
    
    for ruta_archivo in archivos_csv:

        linea_cabecera = -1
        separador_detectado = ';' 
        
        with open(ruta_archivo, 'r', encoding='utf-8', errors='ignore') as f:
            for i, linea in enumerate(f):
                if linea.strip().lower().startswith('date'):
                    linea_cabecera = i
                    if '\t' in linea:
                        separador_detectado = '\t'
                    break
        
        if linea_cabecera == -1:
            continue
            
        
        df_var = pd.read_csv(ruta_archivo, sep=separador_detectado, skiprows=linea_cabecera)
        df_var.columns = df_var.columns.str.strip()
        
        # Merge (unimos por 'Date' Y 'Time')
        if df_estacion is None:
            df_estacion = df_var
        else:
            if 'Time' in df_estacion.columns and 'Time' in df_var.columns:
                df_estacion = pd.merge(df_estacion, df_var, on=['Date', 'Time'], how='outer')
            else:
                df_estacion = pd.merge(df_estacion, df_var, on=['Date'], how='outer')
            
    if df_estacion is not None and not df_estacion.empty:
        # Ordenamos cronológicamente usando "Fecha" y "Hora"
        if 'Time' in df_estacion.columns:
            df_estacion = df_estacion.sort_values(by=['Date', 'Time'])
        else:
            df_estacion = df_estacion.sort_values(by=['Date'])
            
        df_estacion.insert(0, 'Sondeo', estacion)
        dfs_todas_estaciones.append(df_estacion)


df_final = pd.concat(dfs_todas_estaciones, ignore_index=True)

ruta_final = os.path.join(ruta_salida, 'SAIH_Piezometros_unificados_limpios.csv')
df_final.to_csv(ruta_final, index=False)

print(f"Total de filas: {len(df_final)}")