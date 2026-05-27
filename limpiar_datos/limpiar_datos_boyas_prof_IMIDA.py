import pandas as pd
import os
import glob

directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_datos = os.path.join(directorio_actual, 'datos')

nombre_carpeta_origen = 'BoyasProf_clean' 

ruta_origen = os.path.join(ruta_datos, nombre_carpeta_origen) 
ruta_salida = os.path.join(ruta_datos, 'BoyasProf_IMIDA_unificadas')
os.makedirs(ruta_salida, exist_ok=True)

carpetas_estaciones = [d for d in os.listdir(ruta_origen) if os.path.isdir(os.path.join(ruta_origen, d))]
print(f" Encontradas {len(carpetas_estaciones)} carpetas de estaciones en '{nombre_carpeta_origen}'.")

# Diccionario de unidades
diccionario_unidades = {
    "Chlorophyll": "(mg/m3)",
    "Conductivity": "(S/m)",
    "Dissolved oxygen": "(mg/L)",
    "Organic matter": "(mg/L total organic C)",
    "PH": "",
    "Salinity": "", 
    "Temperature": "(°C)",
    "Turbidity": "(FTU)"
}

dfs_todas_estaciones = []

for estacion in carpetas_estaciones:
    ruta_estacion = os.path.join(ruta_origen, estacion)
    archivos_csv = glob.glob(os.path.join(ruta_estacion, "*.csv"))
    
    # Ignoramos metadatos
    archivos_csv = [f for f in archivos_csv if 'metadata' not in f.lower() and 'metadatos' not in f.lower()]
    
    if not archivos_csv:
        continue
        
    print(f"Procesando estación '{estacion}' ({len(archivos_csv)} archivos de variables)")
    
    df_estacion = None
    
    for ruta_archivo in archivos_csv:
        nombre_archivo = os.path.basename(ruta_archivo)
        variable_medida = nombre_archivo.split('_')[0].strip()
        unidad_estandar = diccionario_unidades.get(variable_medida, "")
    

        linea_cabecera = -1
        with open(ruta_archivo, 'r', encoding='utf-8', errors='ignore') as f:
            for i, linea in enumerate(f):
                if linea.strip().lower().startswith('date'):
                    linea_cabecera = i
                    break
        
        # Si es un archivo que no tiene la columna Date, lo avisamos y lo saltamos
        if linea_cabecera == -1:
            print(f" Ignorando archivo '{nombre_archivo}', no tiene columna 'Date'.")
            continue
            
        df_var = pd.read_csv(ruta_archivo, sep=';', skiprows=linea_cabecera)
        df_var.columns = df_var.columns.str.strip()
        
        nuevas_columnas = []
        for col in df_var.columns:

            if 'date' in col.lower():
                nuevas_columnas.append('Date')
            else:
                col_limpia = col.split('(')[0].strip()
                if unidad_estandar:
                    nombre_final = f"{col_limpia}_{variable_medida}_{unidad_estandar}"
                else:
                    nombre_final = f"{col_limpia}_{variable_medida}"
                nuevas_columnas.append(nombre_final)
                
        df_var.columns = nuevas_columnas
        
        if df_estacion is None:
            df_estacion = df_var
        else:
            df_estacion = pd.merge(df_estacion, df_var, on=['Date'], how='outer')
            
    # Solo añadimos la estación si realmente ha extraído datos
    if df_estacion is not None and not df_estacion.empty:
        df_estacion = df_estacion.sort_values(by='Date')
        df_estacion.insert(0, 'Estacion', estacion)
        dfs_todas_estaciones.append(df_estacion)

# Juntamos todas las estaciones
df_final = pd.concat(dfs_todas_estaciones, ignore_index=True)

# Guardamos
ruta_final = os.path.join(ruta_salida, 'BoyasProf_IMIDA_unificadas_limpias.csv')
df_final.to_csv(ruta_final, index=False)

print(f"Total de filas: {len(df_final)}")
print(f"Total de columnas: {len(df_final.columns)}")