import pandas as pd
import os

directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_datos = os.path.join(directorio_actual, 'datos', 'BoyasUPCT_Prof')

df_prof = pd.read_csv(os.path.join(ruta_datos, 'BoyasProf_unificadas_limpias.csv'))
df_upct = pd.read_csv(os.path.join(ruta_datos, 'BoyasUPCT_unificadas_limpias.csv'))

# Añadimos una columna 'Lugar' para que el LLM sepa quién midió qué
df_prof.insert(1, 'Lugar', 'PROF')
df_upct.insert(1, 'Lugar', 'UPCT')

# Los unimos (union vertical)
df_total = pd.concat([df_prof, df_upct], ignore_index=True)

ruta_final = os.path.join(ruta_datos, 'BoyasUPCT_Prof_unificadas_limpias.csv')
df_total.to_csv(ruta_final, index=False)

print(f"Estaciones totales: {df_total['Estacion'].nunique()}")
print(f"Columnas totales: {len(df_total.columns)}")