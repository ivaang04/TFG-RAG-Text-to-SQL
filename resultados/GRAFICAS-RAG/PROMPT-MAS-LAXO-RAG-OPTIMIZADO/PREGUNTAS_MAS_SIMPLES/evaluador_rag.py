from time import time

from langchain_ollama import OllamaLLM
import re

from src.orquestador import Orquestador

class JuezRAG:
    def __init__(self):
        self.juez_llm = OllamaLLM(model="llama3", temperature=0.0)

    def _parsear_respuesta(self, texto_respuesta):
        try:
            puntuacion = 0
            # Intento 1: etiqueta completa con cierre
            puntuacion_match = re.search(
                r"<puntuaci[oó]n>\s*([1-5])\s*</puntuaci[oó]n>",
                texto_respuesta, re.IGNORECASE
            )

            # Intento 2: etiqueta sin cierre
            if not puntuacion_match:
                puntuacion_match = re.search(
                    r"<puntuaci[oó]n>\s*([1-5])",
                    texto_respuesta, re.IGNORECASE
                )

            if puntuacion_match:
                puntuacion = int(puntuacion_match.group(1))
            else:
                # Fallback: último número del texto (menos fiable)
                numeros = re.findall(r"\b[1-5]\b", texto_respuesta)
                if numeros:
                    puntuacion = int(numeros[-1])

            razonamiento = "No se pudo extraer el razonamiento."

            razonamiento_match = re.search(
                r"<raz[oó]namiento>(.*?)(?:</raz[oó]namiento>|<puntuaci[oó]n>)",
                texto_respuesta, re.IGNORECASE | re.DOTALL
            )

            if razonamiento_match:
                razonamiento = razonamiento_match.group(1).strip()
            else:
                texto_antes = re.split(r"<puntuaci[oó]n>", texto_respuesta, flags=re.IGNORECASE)[0]
                if texto_antes.strip():
                    razonamiento = texto_antes.strip()

            return puntuacion, razonamiento

        except Exception as e:
            return 0, f"Error al parsear: {str(e)} | Texto: {texto_respuesta[:200]}"

    def evaluar_context_relevance(self, pregunta, contexto):
        prompt = (
            "Eres un evaluador experto en recuperación de información científica. Tu tarea es evaluar la RELEVANCIA DEL CONTEXTO.\n"
            "Debes evaluar de forma razonada si el CONTEXTO proporcionado por el buscador contiene información útil para responder a la PREGUNTA.\n\n"
            "ESCALA DE PUNTUACIÓN CUALITATIVA (1 a 5):\n"
            "1: Completamente irrelevante. No ayuda en absoluto a responder la pregunta.\n"
            "2: Utilidad baja. Menciona el tema de pasada pero no proporciona datos útiles.\n"
            "3: Utilidad media. Responde parcialmente a la pregunta, pero faltan piezas clave.\n"
            "4: Utilidad alta. Proporciona casi toda la información necesaria.\n"
            "5: Excelente. Información directa, clara y suficiente para responder completamente.\n\n"
            "INSTRUCCIONES DE FORMATO OBLIGATORIAS:\n"
            "1. Escribe tu análisis dentro de <razonamiento>...</razonamiento>.\n"
            "2. Escribe SOLO el número de tu nota dentro de <puntuacion>...</puntuacion>.\n\n"
            "Ejemplo de formato correcto:\n"
            "<razonamiento>El contexto menciona X pero no responde a Y.</razonamiento>\n"
            "<puntuacion>3</puntuacion>\n\n"
            f"<<<PREGUNTA>>>\n{pregunta}\n<<<FIN_PREGUNTA>>>\n\n"
            f"<<<CONTEXTO_OBTENIDO>>>\n{contexto}\n<<<FIN_CONTEXTO>>>\n\n"
            "Recuerda: tu respuesta debe terminar con <puntuacion>N</puntuacion>\n"
            "<razonamiento>"
        )

        respuesta = self.juez_llm.invoke(prompt)
        respuesta_completa = "<razonamiento>\n" + respuesta
        return self._parsear_respuesta(respuesta_completa)


    def evaluar_groundedness(self, respuesta_generada, contexto):
        prompt = (
            "Eres un juez experto encargado de evaluar la FUNDAMENTACIÓN (Groundedness) de la respuesta de un asistente de IA.\n"
            "Tu tarea es determinar en qué medida la RESPUESTA está respaldada por la información del CONTEXTO.\n\n"
            "INSTRUCCIONES CLAVE:\n"
            "- Evalúa el SIGNIFICADO y los HECHOS, no la sintaxis.\n"
            "- Ignora frases introductorias o conectores de formato (ej. 'A continuación:', 'Según el texto:'). No penalices la naturalidad del lenguaje.\n"
            "- Céntrate en buscar alucinaciones: ¿El asistente ha inventado datos, cifras o conclusiones que no estaban en el contexto?\n\n"
            "ESCALA DE PUNTUACIÓN CUALITATIVA (1 a 5):\n"
            "1: Alucinación total. Contradice el contexto o inventa todos los datos.\n\n"
            "2: Poco fundamentado. La mayor parte de los datos aportados no aparecen en el contexto.\n"
            "3: Parcialmente fundamentado. Se basa en el contexto pero añade información externa notable.\n"
            "4: Altamente fundamentado. Correcto casi en su totalidad, con alguna deducción menor que no altera la verdad.\n"
            "5: Totalmente fundamentado. Toda la información sustancial está en el contexto. Sin alucinaciones.\n" 
            "CASO ESPECIAL: Si el contexto no tiene la respuesta y el asistente dice honestamente "
            "'No tengo información suficiente', la nota debe ser 5.\n\n"
            "INSTRUCCIONES DE FORMATO OBLIGATORIAS:\n"
            "1. Escribe tu análisis dentro de <razonamiento>...</razonamiento>.\n"
            "2. Escribe SOLO el número de tu nota dentro de <puntuacion>...</puntuacion>.\n\n"
            "Ejemplo de formato correcto:\n"
            "<razonamiento>La respuesta menciona X que sí aparece en el contexto, pero inventa Y que no está.</razonamiento>\n"
            "<puntuacion>3</puntuacion>\n\n"
            f"<<<CONTEXTO>>>\n{contexto}\n<<<FIN_CONTEXTO>>>\n\n"
            f"<<<RESPUESTA>>>\n{respuesta_generada}\n<<<FIN_RESPUESTA>>>\n\n"
            "Recuerda: tu respuesta debe terminar con <puntuacion>N</puntuacion>\n"
            "<razonamiento>"
        )

        respuesta = self.juez_llm.invoke(prompt)
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

    preguntas_test_semantico = [
        # BLOQUE 1: Teledetección y Satélites
        "¿Qué ventajas ofrecen los satélites de la serie Sentinel frente a las campañas oceanográficas tradicionales para medir la calidad del agua en el Mar Menor?",
        "¿Por qué resulta complicado estimar de forma precisa la concentración de clorofila-a mediante satélite en aguas someras y turbias?",
        "¿Cómo afectan las comunidades de macrófitos del fondo (como Caulerpa prolifera) a la reflectancia captada por los satélites en las zonas poco profundas de la laguna?",
        "A la hora de procesar imágenes satelitales, ¿por qué es necesario aplicar una corrección atmosférica antes de calcular los índices de calidad del agua?",
        "¿Qué parámetros biofísicos del agua se pueden monitorizar habitualmente mediante el uso de índices de reflectividad en imágenes multiespectrales?",

        # BLOQUE 2: Eutrofización y Presiones Humanas
        "¿Cuáles son las principales actividades humanas en la cuenca vertiente que han provocado el aumento de nutrientes y la eutrofización del ecosistema?",
        "¿Qué consecuencias visuales y ecológicas directas tiene la proliferación masiva de fitoplancton en las aguas del Mar Menor?",
        "¿Cómo influyen los episodios de lluvias torrenciales, como las DANAs, en los niveles de turbidez y clorofila-a de la laguna?",
        "¿Qué impacto ha tenido la actividad minera histórica de la Sierra de Cartagena-La Unión en los sedimentos y la calidad del medio natural?",
        "¿Por qué se considera que el Mar Menor ha perdido el carácter oligotrófico que lo caracterizaba en décadas pasadas?",

        # BLOQUE 3: La "Mancha Blanca" y Monitorización in situ
        "¿En qué consiste el fenómeno conocido como 'mancha blanca' en el Mar Menor y cómo difiere ópticamente del resto de las aguas?",
        "¿Qué hipótesis se barajan sobre el origen ambiental y la permanencia temporal de la 'mancha blanca' en la zona occidental de la laguna?",
        "¿Por qué es crítico medir la penetración de la luz en la columna de agua para garantizar la supervivencia de la flora bentónica?",
        "Además de la espectrofotometría en laboratorio, ¿qué otros métodos o instrumentos in situ se utilizan para medir los niveles de clorofila-a en tiempo real?",
        "¿De qué manera interactúa la descarga de aguas subterráneas procedentes del acuífero Cuaternario con la hidrología general del Mar Menor?",

        # BLOQUE 4: Modelización, Machine Learning y Ecología
        "¿Por qué se utiliza frecuentemente la concentración de clorofila-a como el principal indicador biológico para evaluar el estado trófico de un ecosistema costero?",
        "¿Qué ventajas aporta el uso de algoritmos de Machine Learning (inteligencia artificial) en la predicción y modelización de indicadores de calidad del agua?",
        "¿Cómo se relacionan los altos niveles de materia orgánica y nutrientes disueltos con los episodios de anoxia o falta de oxígeno en el fondo marino?",
        "¿Qué papel juegan el drenaje agrícola y las aguas residuales urbanas en la alteración de los mecanismos homeostáticos y de autorregulación de la laguna?",
        "¿Por qué es importante integrar datos masivos heterogéneos (satélites, boyas, estaciones meteorológicas) en un único sistema para gestionar el Mar Menor?",

        # BLOQUE 5: Análisis Estadístico y Aguas Complejas
        "En la validación de modelos hidrológicos y ambientales, ¿por qué se desaconseja el uso exclusivo del coeficiente de correlación (R2) y qué métricas (como el RMSE) se recomiendan como alternativa?",
        "¿Qué características ópticas definen a las llamadas 'aguas de Caso 2' y por qué requieren algoritmos de procesamiento satelital diferentes a los de aguas oceánicas abiertas?",
        "Además de la clorofila-a, ¿qué otros pigmentos fotosintéticos es crucial cuantificar mediante teledetección para detectar floraciones tóxicas de cianobacterias en lagunas eutrofizadas?",
        "¿Qué índices espectrales específicos (como el NDCI) o combinaciones de bandas se proponen en la literatura para estimar la clorofila-a en aguas continentales productivas?",
        "¿Qué ventajas metodológicas ofrece combinar datos satelitales de alta resolución espacial con sensores de alta frecuencia temporal (visita diaria) para monitorizar masas de agua dinámicas?",

        # BLOQUE 6: Gestión, Normativa y Socioeconomía
        "¿Cómo interactúan los ciclos climáticos de sequía y los descensos de los niveles piezométricos del acuífero en la atenuación o desaparición de los episodios de blanqueamiento del agua?",
        "¿Qué tipo de estrategias integradas de gestión de recursos hídricos se promueven a nivel europeo e internacional para la conservación y recuperación de lagunas costeras?",
        "¿Qué restricciones de confidencialidad o normativas de propiedad intelectual suelen aplicarse a la difusión y publicación de los datos brutos obtenidos por las redes de monitorización institucionales?",
        "¿Qué arquitecturas de aprendizaje profundo (Deep Learning) han demostrado ser efectivas para procesar series temporales de imágenes satelitales en la predicción de la calidad del agua?",
        "¿Qué sectores económicos costeros, como la acuicultura o la pesca tradicional, son los principales beneficiarios directos de la implementación de sistemas de alerta temprana basados en teledetección?"
    ]

    nombre_archivo_csv = "resultados_evaluacion_rag.csv"
    campos_csv = [
        "Pregunta", "Context_Relevance_Score", "Groundedness_Score"
    ]
    total_preguntas = len(preguntas_test_semantico)

    lista_k = [3,5,7]
    lista_umbrales = [0.175, 0.25, 0.35, 0.45, 0.55, 0.65]

    total_combinaciones = len(lista_k) * len(lista_umbrales)
    combinacion_actual = 1

    print(f"\n Iniciando evaluación de {total_combinaciones} combinaciones posibles.")

    tiempo_inicio = time()

    for k in lista_k:
        for umbral in lista_umbrales:
            
            # Generamos un nombre de archivo único para esta combinación
            nombre_archivo_csv = f"resultados_k{k}_umbral{umbral}_preguntas_simples.csv"
            nombre_archivo_log = f"log_k{k}_umbral{umbral}_preguntas_simples.txt"
            
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

            # Recorre las preguntas

            for i, pregunta in enumerate(preguntas_test_semantico, 1):
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
                    
                    
                    f_log.write("=== RESPUESTA GENERADA ===\n")
                    f_log.write(f"{respuesta_final}\n")
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
    print(f"Finalizado en {minutos:.1f}")
    print("*"*70)