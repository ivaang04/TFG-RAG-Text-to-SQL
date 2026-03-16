from duckdb import df
import pandas as pd
import os
import asyncio
import re
from langchain_ollama import OllamaLLM
# Importamos TUS módulos reales
from generadorSQL import ConsultaSQL 
from RAG import GestorRAG   
from clasificadorIntencion import ClasificadorIntencion

LIMITE_FILAS = 1000
LIMITE_INTERPRETACION = 15
PROMPTS_POR_TIPO = {
    "AGREGACION": (
        "La tabla contiene un único valor agregado (máximo, mínimo, media, conteo...).\n"
        "Responde en una o dos frases directas mencionando el valor y la fecha si está disponible."
    ),
    "SERIE_TEMPORAL": (
        "Muestra la tabla en formato markdown.\n"
        "Describe brevemente la evolución temporal que muestran los datos."
    ),
    "TABLA_PEQUEÑA": (
        "Responde de forma directa, natural y profesional.\n"
        "Integra los valores de la tabla en tu respuesta o muéstralos como una pequeña lista si es más claro."
    )
}
class Orquestador:
    def __init__(self, ruta_csv, ruta_pdf, modelo="llama3"):
        
        #DuckDB 
        if not os.path.exists(ruta_csv):
            raise FileNotFoundError(f"No encuentro el CSV: {ruta_csv}")
        self.sql_engine = ConsultaSQL(ruta_csv)
        print("Motor SQL: Listo (DuckDB conectado)")

        # Clasificador de intención
        self.clasificador_intencion = ClasificadorIntencion()

        # RAG
        self.rag_engine = GestorRAG()
        self.rag_engine.ingestar(ruta_pdf)
        print("Motor RAG: Listo (Base de datos vectorial cargada)")
        
        self.llm_clasificador = OllamaLLM(model=modelo, temperature=0.0)
        self.llm_generador = OllamaLLM(model=modelo, temperature=0.0)

    def clasificar_pregunta(self, pregunta):
        tipo, scores = self.clasificador_intencion.clasificar(pregunta)
        print("Scores clasificación:", scores)
        return tipo

    def ejecutar_sql(self, pregunta_usuario):

        print("Consultando base de datos DuckDB...")
        try:

            df_resultado = self.sql_engine.generar_y_ejecutar(pregunta_usuario)
            
            if not df_resultado.empty:
                print("\nDATOS EXTRAÍDOS DE DUCKDB:")
                print(df_resultado.to_markdown(index=False)) 
                print("-" * 40)
                
            return df_resultado
                
        except ValueError as ve:
            return f"ERROR DE SEGURIDAD/VALIDACIÓN: {ve}"
        except RuntimeError as re:
            return f"ERROR DE EJECUCIÓN SQL (Tras reintentos): {re}"
        except Exception as e:
            return f"ERROR DESCONOCIDO EN SQL: {e}"

    def ejecutar_RAG(self, pregunta_usuario):

        print("\nConsultando PDFs con RAG para conseguir contextos teóricos...")
        resultado = self.rag_engine.consultar(pregunta_usuario)
        print("   -> Literatura recuperada correctamente.")
        return resultado
    
    def ejecutar_pipeline(self, pregunta_usuario):

        print(f"Procesando solicitud: '{pregunta_usuario}'\n")

        tipo = self.clasificar_pregunta(pregunta_usuario)
        print(f"Tipo detectado: {tipo}")

        if tipo == "SQL":
            datos = self.ejecutar_sql(pregunta_usuario)
            return self.generar_sql_only(pregunta_usuario, datos)

        elif tipo == "RAG":
            literatura = self.ejecutar_RAG(pregunta_usuario)
            return self.generar_rag_only(pregunta_usuario, literatura)

        elif tipo == "HIBRIDO":
            datos = self.ejecutar_sql(pregunta_usuario)
            literatura = self.ejecutar_RAG(pregunta_usuario)

            if isinstance(datos, str) and "ERROR" in datos:
                return "No se pudo obtener evidencia empírica válida...\n\n" + self.generar_rag_only(pregunta_usuario, literatura)
            
            return self.generar_sintesis(pregunta_usuario, datos, literatura)

    def detectar_tipo_resultado(self, df):
        total_filas = len(df)

        if total_filas == 1:
            return "AGREGACION"

        tiene_fecha = any(
            pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower() or "time" in col.lower()
            for col in df.columns
        )

        if total_filas <= LIMITE_INTERPRETACION:
            if tiene_fecha:
                return "SERIE_TEMPORAL"
            return "TABLA_PEQUEÑA"

        return "TABLA_GRANDE"

    def generar_sql_only(self, pregunta, df_resultado):

        if isinstance(df_resultado, str): return df_resultado
        if df_resultado.empty:
            return "La consulta no devolvió datos (tabla vacía)."

        total_filas = len(df_resultado)
        tipo = self.detectar_tipo_resultado(df_resultado)

        if tipo == "TABLA_GRANDE":
            print("Filas devueltas por SQL:", total_filas)
            
            if total_filas > LIMITE_FILAS:
                return (
                    f"Se encontraron {total_filas} registros en total.\n\n"
                    f"Se muestran los primeros {LIMITE_FILAS}:\n\n"
                    f"{df_resultado.head(LIMITE_FILAS).to_markdown(index=False)}\n\n"
                    f"*(Tabla truncada. {total_filas - LIMITE_FILAS} registros adicionales no mostrados.)*"
                )
            else:
                return (
                    f"Se encontraron {total_filas} registros en total:\n\n"
                    f"{df_resultado.to_markdown(index=False)}"
                )

        datos = (
            f"Total de registros encontrados: {total_filas}.\n\n"
            f"{df_resultado.to_markdown(index=False)}"
        )

        instruccion_especifica = PROMPTS_POR_TIPO[tipo]

        prompt = (
            "Eres un analista de datos ambiental. Responde a la pregunta del usuario usando exclusivamente los datos proporcionados.\n\n"
            f"<<< PREGUNTA >>>\n{pregunta}\n<<< FIN_PREGUNTA >>>\n\n"
            f"<<< DATOS >>>\n{datos}\n<<< FIN_DATOS >>>\n\n"
            "INSTRUCCIONES:\n"
            f"{instruccion_especifica}\n\n"
            "REGLAS GENERALES:\n"
            "- NO repitas las etiquetas <<< PREGUNTA >>> ni <<< DATOS >>> en tu respuesta.\n"
            "- NO inventes datos que no estén en la tabla.\n"
            "- Responde en el mismo idioma de la pregunta."
        )

        return self.llm_generador.invoke(prompt)


    def generar_rag_only(self, pregunta, resultado_rag):

        if isinstance(resultado_rag, str):
            contexto = resultado_rag  # El mensaje de "No se encontró..."
            fuentes = []              # Lista de fuentes vacía
        else:
            contexto = resultado_rag["contexto"]
            fuentes = resultado_rag["fuentes"]

        prompt = (
            "Eres un investigador científico riguroso. Tu tarea es responder a la pregunta del usuario "
            "basándote ÚNICAMENTE en el contexto de la literatura proporcionada.\n\n"
            f"<<<PREGUNTA>>>\n{pregunta}\n<<<FIN_PREGUNTA>>>\n\n"
            f"<<<CONTEXTO_CIENTIFICO>>>\n{contexto}\n<<<FIN_CONTEXTO>>>\n\n"
            "INSTRUCCIONES CRÍTICAS:\n\n"
            "1. CERO ALUCINACIONES\n"
            "Responde solo con hechos que aparezcan explícitamente en el contexto. No uses tus conocimientos previos.\n\n"
            "2. ANTI-COMPLACENCIA\n"
            "Si la pregunta menciona lugares, herramientas o conceptos que NO aparecen en el contexto, "
            "TIENES PROHIBIDO intentar asimilarlos o forzar una relación.\n\n"
            "3. CONFESIÓN DE IGNORANCIA\n"
            "Si el contexto no responde a la pregunta, di textualmente: "
            "'La literatura científica recuperada no menciona [lo que el usuario ha preguntado]. "
            "El documento trata sobre [de qué trata el contexto].'\n\n"
            "4. Sé directo y académico. Responde en el mismo idioma de la pregunta."
        )

        respuesta = self.llm_generador.invoke(prompt)

        if fuentes:
            respuesta_final = respuesta + "\n\n---\nFuentes consultadas:\n"
            for fuente in fuentes:
                respuesta_final += f"- {fuente}\n"
            return respuesta_final

        return respuesta


    def generar_sintesis(self, pregunta, datos, resultado_rag):

        if isinstance(datos, str):
            datos_str = datos
            total_filas = 0
        else:
            total_filas = len(datos)

            if total_filas > LIMITE_FILAS:
                df_reducido = datos.head(LIMITE_FILAS)
                datos_str = (
                    f"Total de registros devueltos: {total_filas}. "
                    f"Se muestran los primeros {LIMITE_FILAS}:\n\n"
                    f"{df_reducido.to_markdown(index=False)}\n\n"
                    f"[NOTA DEL SISTEMA: Tabla truncada. Solo se muestran {LIMITE_FILAS} de {total_filas} registros.]"
                )
            else:
                datos_str = (
                    f"Total de registros devueltos: {total_filas}.\n\n"
                    f"{datos.to_markdown(index=False)}"
                )

        if isinstance(resultado_rag, str):
            contexto = resultado_rag
            fuentes = []
        else:
            contexto = resultado_rag["contexto"]
            fuentes = resultado_rag["fuentes"]

        prompt_sintesis = (
            "Eres un analista de datos estrictamente objetivo. Usa solo la información de los bloques proporcionados.\n\n"
            f"<<<PREGUNTA>>>\n{pregunta}\n<<<FIN_PREGUNTA>>>\n\n"
            f"<<<DATOS_SQL>>>\n{datos_str}\n<<<FIN_DATOS_SQL>>>\n\n"
            f"<<<CONTEXTO_CIENTIFICO>>>\n{contexto}\n<<<FIN_CONTEXTO_CIENTIFICO>>>\n\n"
            "INSTRUCCIONES:\n\n"
            "1. SOBERANÍA EMPÍRICA\n"
            "Los DATOS_SQL son la fuente principal. No modifiques ni inventes valores.\n\n"
            "2. CERO ALUCINACIONES\n"
            "No inventes datos, fechas ni conclusiones que no estén en la tabla.\n\n"
            "3. CONTEXTO CIENTÍFICO\n"
            "Usa ÚNICAMENTE el texto del bloque CONTEXTO_CIENTIFICO para la parte teórica. "
            "Aunque conozcas información relevante sobre el tema, NO la incluyas si no está en el contexto. \n\n"
            "4. IDIOMA Y FORMATO\n"
            "Responde en el mismo idioma de la pregunta. Sé directo y conciso."
        )

        respuesta_limpia = self.llm_generador.invoke(prompt_sintesis)

        if fuentes:
            respuesta_final = (
                respuesta_limpia + "\n\n---\nFuentes consultadas:\n"
                + "".join(f"- {fuente}\n" for fuente in fuentes)
            )
            return respuesta_final

        return respuesta_limpia

if __name__ == "__main__":

    ruta_csv_real = "/home/ivan/Universidad/TFG/datos/AEMET_clean/MS_Torrepacheco1.csv"
    ruta_pdf_real = "/home/ivan/Universidad/TFG/datos/Docs/Delegido et al_2014_Modelo empírico para la determinación de clorofila-a en aguas continentales a.pdf"
    
    
    try:
        app = Orquestador(ruta_csv_real, ruta_pdf_real)

        while True:
            pregunta = input("\n¿Qué quieres saber de los datos? (o 'salir'): ")
            
            if pregunta.lower() in ["salir", "exit", "q"]:
                break
            if not pregunta.strip(): continue

            respuesta = app.ejecutar_pipeline(pregunta)
            
            print("\n:Respuesta final inferida:")
            print("-" * 40)
            print(respuesta)
            print("-" * 40)

    except Exception as e:
        print(f"\n Error fatal iniciando el orquestador: {e}")