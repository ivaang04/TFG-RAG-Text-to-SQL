from generadorSQL import ConsultaSQL
import os

def main():

    nombre_archivo = "/home/ivan/Universidad/TFG/datos/AEMET_clean/MS_Torrepacheco1.csv" 
    ruta_csv = os.path.join("datos", nombre_archivo)
    
    if not os.path.exists(ruta_csv):
        print(f"Error: No se encuentra el archivo en {ruta_csv}")
        return

    print(f"Cargando archivo: {nombre_archivo}...")
    
    try:
        agente = ConsultaSQL(ruta_csv)
        print("Conexión establecida con éxito.")
        print("\nEsquema de datos detectado:")
        print(agente.obtener_esquema())
    except Exception as e:
        print(f"Error al inicializar el motor: {e}")
        return

    # --- BUCLE DE CONSULTA ---
    while True:
        pregunta = input("\n¿Qué quieres saber de los datos? (o 'salir'): ")
        
        if pregunta.lower() in ['salir', 'exit', 'q']:
            print("Cerrando el sistema...")
            break
        
        if not pregunta.strip(): continue
        try:
            # 1. Generar SQL
            resultado_sql = agente.generar_y_ejecutar(pregunta)
            
            print("\n Resultado:")
            if resultado_sql.empty:
                print("La consulta se pudo ejecutar pero no hay datos que la cumplan.")
            else:
                # Mostramos las primeras 10 filas para no saturar la terminal
                print(resultado_sql.head(10))
                if len(resultado_sql) > 10:
                    print(f"\n... (mostrando 10 de {len(resultado_sql)} filas)")
                explicacion = agente.generar_respuesta_natural(pregunta, resultado_sql)
                print(explicacion)    
                
        # Capturamos errores de validación (Seguridad, SQL inválido)
        except ValueError as e_validacion:
            print(f"\n ACCESO DENEGADO / ERROR DE ENTRADA:")
            print(f"   >> {e_validacion}")
            
        # Capturamos errores de ejecución (DuckDB falló 3 veces)
        except RuntimeError as e_ejecucion:
            print(f"\n ERROR DEL SISTEMA (Tras varios intentos):")
            print(f"   >> {e_ejecucion}")
            
        # Capturamos cualquier otra cosa inesperada
        except Exception as e_generico:
            print(f"\n ERROR INESPERADO: {e_generico}")

if __name__ == "__main__":
    main()