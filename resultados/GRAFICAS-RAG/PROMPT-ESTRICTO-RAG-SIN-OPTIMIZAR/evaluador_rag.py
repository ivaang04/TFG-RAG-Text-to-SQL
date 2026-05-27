from time import time

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import re

from src.orquestador import Orquestador

class JuezRAG:
    def __init__(self):
        self.juez_llm = OllamaLLM(model="llama3", temperature=0.0)

    def _parsear_respuesta(self, texto_respuesta):
        """Extrae la puntuación numérica y el razonamiento de forma robusta."""
        try:
            puntuacion = 0
            puntuacion_match = re.search(r"<puntuaci[oó]n>\s*([1-5])\s*</puntuaci[oó]n>", texto_respuesta, re.IGNORECASE)
            
            if puntuacion_match:
                puntuacion = int(puntuacion_match.group(1))
            else:
                #Si no puso etiqueta, buscamos el último número del 1 al 5
                numeros = re.findall(r"\b[1-5]\b", texto_respuesta)
                if numeros:
                    puntuacion = int(numeros[-1])

            # Extraer el razonamiento 
            razonamiento = "No se pudo extraer el razonamiento."
            
            #Si tiene la etiqueta <razonamiento> explícita
            razonamiento_match = re.search(r"<raz[oó]namiento>(.*?)(?:</raz[oó]namiento>|<puntuaci[oó]n>)", texto_respuesta, re.IGNORECASE | re.DOTALL)
            
            if razonamiento_match:
                razonamiento = razonamiento_match.group(1).strip()
            else:
                #Si no tiene etiqueta, cogemos todo lo que haya antes de <puntuacion>
                texto_antes_puntuacion = re.split(r"<puntuaci[oó]n>", texto_respuesta, flags=re.IGNORECASE)[0]
                if texto_antes_puntuacion.strip():
                    razonamiento = texto_antes_puntuacion.strip()

            return puntuacion, razonamiento
            
        except Exception as e:
            return 0, f"Error al parsear: {str(e)} | Texto original: {texto_respuesta}"

    def evaluar_context_relevance(self, pregunta, contexto):
        """Mide si los chunks de ChromaDB son útiles para responder la pregunta."""
        template = """
        Eres un evaluador experto en recuperación de información científica. Tu tarea es evaluar la RELEVANCIA DEL CONTEXTO.
        Debes evaluar si el CONTEXTO proporcionado por el buscador contiene información útil para responder a la PREGUNTA.
        
        ESCALA DE PUNTUACIÓN (1 a 5):
        1: Completamente irrelevante. No tiene nada que ver con la pregunta.
        2: Mención muy tangencial. Aparece alguna palabra clave pero no aporta valor.
        3: Parcialmente relevante. Menciona el tema, pero falta información para dar una respuesta completa.
        4: Muy relevante. Responde a casi toda la pregunta, pero falta algún detalle menor.
        5: Perfectamente relevante. Contiene toda la información exacta para responder perfectamente.

        PREGUNTA: 
        {pregunta}
        ---
        CONTEXTO OBTENIDO:
        {contexto}

        INSTRUCCIONES DE FORMATO OBLIGATORIAS:
        1. Escribe tu análisis paso a paso dentro de la etiqueta <razonamiento>.
        2. Después del análisis, escribe SOLO el número de tu nota (del 1 al 5) dentro de la etiqueta <puntuacion>.
        
        Ejemplo:
        <razonamiento>El texto habla de X, lo cual responde a la primera parte de la pregunta, pero omite Y.</razonamiento>
        <puntuacion>3</puntuacion>

        <razonamiento>
        """
        prompt = PromptTemplate.from_template(template)
        texto_formateado = prompt.format(pregunta=pregunta, contexto=contexto)
        respuesta = self.juez_llm.invoke(texto_formateado)
        
        respuesta_completa = "<razonamiento>\n" + respuesta
        return self._parsear_respuesta(respuesta_completa)
    def evaluar_groundedness(self, respuesta_generada, contexto):
        """Mide si la respuesta del asistente está 100% basada en el texto recuperado (Cero Alucinaciones)."""
        template = """
        Eres un auditor de datos implacable, literal y extremadamente estricto. Tu tarea es evaluar la FUNDAMENTACIÓN (Groundedness).

        DEFINICIÓN CRÍTICA:
        Una "afirmación" es cualquier dato concreto: cifra, nombre propio, lugar, fecha, método o conclusión específica.

        REGLAS DE EVALUACIÓN:
        1. Debes EXTRAER y evaluar TODAS las afirmaciones de la respuesta (mínimo 1 línea por afirmación).
        2. Cada afirmación debe marcarse obligatoriamente como:
           - PRESENTE (aparece o es paráfrasis clara del contexto)
           - NO PRESENTE (no aparece en el contexto)
        3. NO se permiten inferencias, generalizaciones ni conocimiento externo.
        4. Si tienes dudas → marcar como NO PRESENTE (criterio conservador).

        REGLA DE PENALIZACIÓN (DETERMINISTA):
        - 0 afirmaciones incorrectas (NO PRESENTE) → puntuación = 5
        - 1 afirmación incorrecta (NO PRESENTE) → puntuación = 3
        - 2 o más afirmaciones incorrectas (NO PRESENTE) → puntuación = 1

        CASO ESPECIAL:
        - Si la respuesta dice que no hay información:
            - y es correcto → 5
            - y el contexto SÍ tenía información → 1

        CONTROL DE CALIDAD:
        - Si no analizas afirmaciones o el análisis es incompleto → puntuación = 1

        FORMATO OBLIGATORIO:
        <razonamiento>
        - Afirmación 1: [Texto de la afirmación] → PRESENTE / NO PRESENTE
        - Afirmación 2: [Texto de la afirmación] → PRESENTE / NO PRESENTE
        ...
        Resumen: X correctas, Y incorrectas. Aplicando la regla de penalización, la nota es Z.
        </razonamiento>
        <puntuacion>Z</puntuacion>

        DATOS:
        CONTEXTO:
        {contexto}
        ---
        RESPUESTA:
        {respuesta_generada}

        <razonamiento>
        """
        prompt = PromptTemplate.from_template(template)
        texto_formateado = prompt.format(respuesta_generada=respuesta_generada, contexto=contexto)
        
        respuesta = self.juez_llm.invoke(texto_formateado)
        respuesta_completa = "<razonamiento>\n" + respuesta
        
        return self._parsear_respuesta(respuesta_completa)  


import csv

if __name__ == "__main__":
    
    ruta_csv_real = "/home/ivan/Universidad/TFG/datos/AEMET_clean/MS_Torrepacheco1.csv"
    ruta_directorio_real = "/home/ivan/Universidad/TFG/datos/Docs/" 
    
    app = Orquestador(ruta_csv_real, ruta_directorio_real)
    juez = JuezRAG()

    # 30 preguntas de test
    preguntas_test = [
        "Según el estudio de Caballero et al., ¿qué satélites se proponen utilizar como herramienta de alerta temprana para la monitorización de la calidad del agua en el Mar Menor?",
        "¿Qué cuatro algoritmos de Machine Learning y Deep Learning evaluaron Gómez et al. (2021) para estimar el contenido de clorofila-a en el Mar Menor usando datos de Sentinel-2?",
        "Según el artículo de Erena et al. (2019), ¿cuáles son las principales presiones humanas que sufre el frágil ecosistema del Mar Menor?",
        "¿Con qué sensor específico validaron Delegido et al. (2014) sus modelos empíricos de estimación de clorofila-a diseñados para los satélites Sentinel-2 y Sentinel-3?",
        "¿Qué parámetros biofísicos se relacionaron mediante el uso de datos in situ y técnicas de teledetección en el análisis multisensor de Erena et al. (2017)?",
        "Según el informe del IEO-CSIC sobre la Mancha Blanca, ¿cuáles son las principales causas ambientales que originan este fenómeno en el Mar Menor?",
        "¿A qué proyecto marco del Ministerio para la Transición Ecológica (MITERD) pertenece el estudio Origen, naturaleza, causas y consecuencias de la Mancha Blanca?",
        "Según los perfiles CTD del informe de actualización del IEO-CSIC (octubre 2024), ¿se pueden publicar y difundir libremente los datos contenidos en el informe?",
        "Según Jimeno-Sáez et al. (2020), ¿qué indicador biológico de eutrofización se modeliza en el Mar Menor utilizando algoritmos de Machine Learning?",
        "¿Cuáles fueron los dos años en los que el Mar Menor sufrió crisis de eutrofización sin precedentes, según señalan Jimeno-Sáez et al. (2020)?",
        "En la investigación de Giménez et al. (2024) sobre aguas someras, ¿qué tipo específico de arquitectura de Deep Learning destaca para procesar la información del satélite Sentinel-3?",
        "¿En qué programa espacial o servicio europeo se fundamenta el sistema de monitorización diaria de clorofila-a propuesto por Giménez et al. (2023)?",
        "¿Qué índice proponen específicamente Mishra y Mishra (2012) para estimar la clorofila-a en aguas productivas y turbias?",
        "Según los ensayos de Mishra y Mishra (2012), ¿qué tipo de regresión o función matemática explicaba mejor la varianza del NDCI frente a la clorofila-a simulada?",
        "¿Qué banda espectral y qué umbral empírico proponen Katlane et al. (2024) en Sentinel-2 para enmascarar los píxeles de aguas poco profundas influenciados por el fondo?",
        "Según Katlane et al. (2024), ¿cuál es la mejor banda proxy del satélite Sentinel-2 MSI para medir la turbidez del agua?",
        "¿Para qué sensor satelital concreto desarrollaron Doerffer y Schiller (2007) su algoritmo específico de aguas de Caso 2?",
        "¿Por qué resulta difícil la estimación remota de clorofila-a en estuarios turbios y productivos según el estudio de Lins et al. (2017)?",
        "Según Pahlevan et al. (2020), ¿qué enfoque metodológico permite realizar extracciones continuas de clorofila-a integrando datos de Sentinel-2 y Sentinel-3?",
        "¿Qué tipo de estrategia general promueve el libro de Lillebø et al. (2015) para las lagunas costeras europeas?",
        "¿Cuál es el objetivo principal de integrar datos masivos heterogéneos según la investigación de López-Andreu et al. (2022) en el Mar Menor?",
        "Según Legates y McCabe (1999), ¿por qué las medidas como el coeficiente de correlación (R2) son peligrosas para validar modelos hidrológicos?",
        "¿Qué estadísticos recomiendan Legates y McCabe (1999) utilizar como alternativa al R2?",
        "En el estudio de Fernández-Tejedor et al. (2022) en el Delta del Ebro, ¿para qué sector económico utilizan las estimaciones precisas de clorofila obtenidas con Sentinel-2?",
        "Además de lagos en España, ¿en qué otro país tomaron muestras Delegido et al. (2014) para validar sus modelos empíricos aplicables a Sentinel-2 y 3?",
        "¿A qué instituto público de investigación español pertenecen los autores principales del estudio de alerta temprana con Sentinel-2 de Caballero et al. (2022)?",
        "Según Gómez et al. (2021), ¿por qué es inviable depender únicamente de las campañas oceanográficas in situ tradicionales para monitorizar el Mar Menor?",
        "Según el artículo de Erena et al. (2019), ¿de dónde proceden las principales presiones humanas que afectan a la turbidez y clorofila del Mar Menor?",
        "¿Qué institución lidera o está fuertemente implicada en el análisis multisensor espacial presentado en el congreso por Erena et al. (2017)?",
        "Según el trabajo de Lins et al. (2017) y los estudios relacionados que citan, ¿qué otro pigmento relacionado con las cianobacterias es importante cuantificar mediante teledetección en lagunas eutrofizadas?"
    ]

    nombre_archivo_csv = "resultados_evaluacion_rag.csv"
    campos_csv = [
        "Pregunta", "Context_Relevance_Score", "Groundedness_Score"
    ]
    total_preguntas = len(preguntas_test)

    lista_k = [3,5,7]  
    lista_umbrales = [0.25, 0.35, 0.45, 0.55]

    total_combinaciones = len(lista_k) * len(lista_umbrales)
    combinacion_actual = 1

    print(f"\n Iniciando evaluación de {total_combinaciones} combinaciones posibles.")

    tiempo_inicio = time()

    for k in lista_k:
        for umbral in lista_umbrales:
            
            # Generamos un nombre de archivo único para esta combinación
            nombre_archivo_csv = f"resultados_k{k}_umbral{umbral}.csv"
            nombre_archivo_log = f"log_k{k}_umbral{umbral}.txt"
            
            print("\n" + "="*70)
            print(f"[{combinacion_actual}/{total_combinaciones}] INICIANDO PRUEBA -> K: {k} | Umbral: {umbral}")
            print(f"Guardando en: {nombre_archivo_csv}")
            print("="*70)
            
            # Creamos el archivo y escribimos la cabecera
            with open(nombre_archivo_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=campos_csv)
                writer.writeheader()

            with open(nombre_archivo_log, mode="w", encoding="utf-8") as f_log:
                f_log.write(f"=== DIARIO DEL JUEZ | K={k} | Umbral={umbral} ===\n\n")

            # 3. Recorre las 30 preguntas para la combinación actual

            for i, pregunta in enumerate(preguntas_test, 1):
                print(f"\n  -> Pregunta {i}/{total_preguntas}: '{pregunta}'")
                
                # Pasamos los valores K y umbral
                resultado_busqueda = app.ejecutar_RAG(pregunta, k_relevantes=k, umbral=umbral)
                respuesta_final = app.generar_rag_only(pregunta, resultado_busqueda, incluir_fuentes=False)
                
                # Extraemos el contexto en crudo para pasárselo al juez
                if isinstance(resultado_busqueda, dict):
                    contexto_real = resultado_busqueda["contexto"]
                else:
                    contexto_real = resultado_busqueda


                print("Evaluando métricas...")
                score_cr, razon_cr = juez.evaluar_context_relevance(pregunta, contexto_real)
                score_g, razon_g = juez.evaluar_groundedness(respuesta_final, contexto_real)
                
                print(f"  Resultados: CR={score_cr}/5 | Groundedness={score_g}/5")

                # Guardamos la fila en el CSV de esta combinación
                fila = {
                    "Pregunta": pregunta,
                    "Context_Relevance_Score": score_cr,
                    "Groundedness_Score": score_g
                }
                
                with open(nombre_archivo_csv, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=campos_csv)
                    writer.writerow(fila)

                with open(nombre_archivo_log, mode="a", encoding="utf-8") as f_log:
                    f_log.write(f"--- PREGUNTA {i} ---\n")
                    f_log.write(f"USUARIO: {pregunta}\n\n")

                    f_log.write("=== CONTEXTO RECUPERADO (CHROMA DB) ===\n")
                    f_log.write(f"{contexto_real}\n")
                    f_log.write("=======================================\n\n")
                    
                    f_log.write(f"[NOTA CONTEXT RELEVANCE: {score_cr}/5]\n")
                    f_log.write(f"{razon_cr}\n\n")
                    
                    f_log.write(f"[NOTA GROUNDEDNESS: {score_g}/5]\n")
                    f_log.write(f"{razon_g}\n\n")
                    
                    f_log.write("*"*70 + "\n\n")
            
            print(f"\nCOMBINACIÓN COMPLETADA: {nombre_archivo_csv}")
            combinacion_actual += 1

    tiempo_fin = time()
    minutos = (tiempo_fin - tiempo_inicio) / 60
    print("\n" + "*"*70)
    print(f"Finalizado en {minutos:.1f} minutos")
    print("*"*70)