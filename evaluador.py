import numpy as np
import csv
import random
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings

class EvaluadorEnrutador:
    def __init__(self, ejemplos_base):
        print("Cargando modelo de embeddings...")

        self.embedder = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        print("Calculando vectores base...")
        self.vectores_ejemplos = []

        for categoria, preguntas in ejemplos_base.items():
            for q in preguntas:
                emb = self.embedder.embed_query("query: " + q)
                self.vectores_ejemplos.append({
                    "texto": q,
                    "tipo": categoria,
                    "vector": np.array(emb)
                })

    def similitud_coseno(self, v1, v2):
        return np.dot(v1, v2)

    def evaluar_pregunta(self, pregunta):

        emb_pregunta = np.array(
            self.embedder.embed_query("query: " + pregunta)
        )

        resultados = []

        for item in self.vectores_ejemplos:
            score = self.similitud_coseno(emb_pregunta, item["vector"])
            resultados.append({
                "tipo": item["tipo"],
                "score": float(score)
            })

        resultados = sorted(resultados, key=lambda x: x["score"], reverse=True)


        # Agrupar scores por categoría
        scores_por_categoria = defaultdict(list)

        for r in resultados:
            scores_por_categoria[r["tipo"]].append(r["score"])

        # 1. 1-NN
        pred_1nn = resultados[0]["tipo"]


        # Media total
        pred_media_total = max(
            scores_por_categoria,
            key=lambda cat: np.mean(scores_por_categoria[cat])
        )


        #Media Top-K
        ks = [1, 2, 3, 4, 5, 7, 10]
        predicciones_media_k = {}

        for k in ks:
            pred = max(
                scores_por_categoria,
                key=lambda cat: np.mean(
                    sorted(scores_por_categoria[cat], reverse=True)[:k]
                )
            )
            predicciones_media_k[f"Media_Top_{k}"] = pred

        #Voto Top-K
        predicciones_voto_k = {}

        for k in ks:
            conteo = defaultdict(int)

            for r in resultados[:k]:
                conteo[r["tipo"]] += 1

            ganador = max(
                conteo,
                key=lambda cat: (
                    conteo[cat],
                    max(scores_por_categoria[cat])
                )
            )

            predicciones_voto_k[f"Voto_Top_{k}"] = ganador


        #Random
        pred_random = random.choice(["SQL", "RAG", "HIBRIDO"])

        #SCORES
        scores_medios = {
            cat: float(np.mean(vals))
            for cat, vals in scores_por_categoria.items()
        }

        return (
            pred_1nn,
            pred_media_total,
            predicciones_media_k,
            predicciones_voto_k,
            pred_random,
            scores_medios
        )


# DATASET BASE (63 PREGUNTAS)
dataset_tfg = {
     "SQL": [
        "¿Cuál fue la temperatura media mensual registrada en la boya de Los Urrutias durante el verano?",
        "Filtra los registros de la estación E3 donde la concentración de clorofila-a superó los 5 mg/m3 y devuélveme la fecha exacta.",
        "Dime el valor máximo de precipitación acumulada en la estación meteorológica de San Javier.",
        "¿Cuántos días el nivel de oxígeno disuelto en la Cubeta Sur estuvo por debajo de 4 mg/L?",
        "¿Cuál es el registro de caudal más alto en la rambla del Albujón según los datos del SAIH?",
        "Muestra los valores de conductividad medidos en la boya de Entreislas cuando la temperatura del agua superó los 28 grados.",
        "¿Qué día se registró el nivel freático más bajo en los piezómetros del Campo de Cartagena?",
        "Busca en la base de datos los registros donde el pH de la boya de Los Nietos haya bajado de 7.5.",
        "Dime a qué hora exacta se midió la temperatura del agua más alta en la boya de Los Narejos.",
        "Extrae todos los registros de la estación E3 donde la salinidad sea mayor a 44 PSU.",
        "¿Cuál fue el valor máximo o mínimo registrado?",
        "¿Cuántos registros superan este umbral en la tabla?",
        "Muéstrame los datos de la estación del mes de julio.",
        "Dame el promedio mensual y la desviación estándar.",
        "¿Qué día y hora exactos se registró el valor más extremo?",
        "Filtra los registros por variable y devuelve la tabla resultante.",
        "¿Cuál es el valor mínimo de oxígeno disuelto registrado en la base de datos?",
        "Calcula la media de turbidez agrupada por mes.",
        "¿Cuántos registros superan el umbral de salinidad de 38 PSU?",
        "Dame la desviación típica de la temperatura del agua para cada año disponible.",
        "Muestra los registros donde la concentración de clorofila-a fue mayor a 10 mg/m3."
    ],
    "RAG": [
        "Según los informes del IEO-CSIC, ¿cuáles son las principales causas del fenómeno de la mancha blanca en el Mar Menor?",
        "¿Qué ventajas tiene el algoritmo C2RCC para estimar clorofila-a con Sentinel-2 en aguas someras según la literatura?",
        "¿Qué modelos de Machine Learning y Deep Learning se proponen en la literatura para monitorizar la calidad del agua?",
        "¿Qué metodología se utilizó para aplicar Landsat-8 y Sentinel-2 como sistema de alerta temprana en lagunas costeras?",
        "¿Cómo se realiza la corrección atmosférica sobre aguas continentales según los estudios de teledetección?",
        "¿Qué relación causal existe entre la anoxia y la mortandad masiva de peces según los papers?",
        "¿Qué es el índice NDCI propuesto por Mishra y Mishra y cómo se calcula?",
        "Detalla el uso de redes neuronales artificiales para modelar la eutrofización en ecosistemas acuáticos.",
        "¿Qué metodología, modelo empírico o algoritmo propone el estudio?",
        "Según el artículo, ¿cómo se define este fenómeno?",
        "¿Qué conclusiones o resultados principales extrae el paper?",
        "Según la literatura científica, ¿qué factores influyen en este proceso?",
        "¿Qué instrumentos o herramientas se recomiendan según el documento?",
        "Explícame la teoría descrita en el documento sobre este proceso.",
        "¿Qué factores influyen en la eutrofización de lagunas costeras según la literatura científica?",
        "¿Qué sensores satelitales se recomiendan para monitorizar la calidad del agua marina?",
        "¿Cómo afecta la turbidez a la penetración de luz en ecosistemas acuáticos según los estudios?",
        "¿Qué índices espectrales se usan para detectar floraciones de algas en aguas costeras?",
        "Según la literatura, ¿cómo afecta la escorrentía agrícola a la salinidad y nutrientes de una laguna?",
        "¿Qué métodos analíticos se emplean para medir parámetros biofísicos en agua de mar?",
        "¿Qué variabilidad espacio-temporal presentan los parámetros de calidad del agua en lagunas costeras?"
    ],
    "HIBRIDO": [
        "Cruza los picos de clorofila-a medidos en la estación E3 con los umbrales de eutrofización definidos en los informes del IEO-CSIC.",
        "¿Explica la teoría de Pérez-Ruzafa la bajada repentina de salinidad que muestran los datos de la boya de Los Urrutias?",
        "Verifica si las temperaturas extremas registradas en la boya de Pedruchillo coinciden con los periodos de crisis ecológica mencionados en los artículos.",
        "Basado en la evolución del pH y materia orgánica de la estación E3 en 2023, ¿qué fase de eutrofización estaría atravesando el Mar Menor según los autores?",
        "Relaciona los niveles altos de sólidos disueltos totales en los datos empíricos con las causas de la mancha blanca descritas por el CSIC.",
        "Analiza los datos de precipitación de San Javier e indica cómo este tipo de eventos DANA afectan a la turbidez según la literatura.",
        "¿Coinciden los datos medidos en nuestra tabla con lo que describe la literatura?",
        "Compara los registros reales de la estación con la teoría del artículo.",
        "¿Es normal el valor registrado hoy si nos basamos en los estudios científicos?",
        "¿Hay anomalías en nuestra base de datos respecto a los umbrales del paper?",
        "Valida los picos de nuestra tabla usando el modelo teórico del documento.",
        "¿Nuestra tabla empírica refleja los episodios descritos por los autores?",
        "¿Podemos usar la teoría del artículo para interpretar los datos de nuestra tabla?",
        "¿Los patrones de nuestra base de datos coinciden con las tendencias descritas en el paper?",
        "Cómo se relacionan los registros de nuestra tabla con los conceptos teóricos del documento?",
        "Extrae el valor medio de oxígeno disuelto de los datos empíricos e indica si, según los papers, estos niveles son críticos para las especies bentónicas.",
        "A partir de la temperatura máxima registrada en los datos, investiga en la bibliografía el efecto del calentamiento en la dinámica de nutrientes.",
        "Compara el caudal máximo registrado en los datos con los impactos de escorrentía descritos en la literatura científica.",
        "Calcula la concentración media de clorofila-a en los datos empíricos y contrástala con los umbrales de eutrofización definidos en los artículos.",
        "Analiza las tendencias de nuestra tabla de datos y explica si concuerdan con los patrones documentados en los papers",
        "¿Justifica la literatura el comportamiento anómalo del pH observado en nuestra base de datos durante los episodios de lluvia intensa?"
    ]
}

# DATASET DE TEST (66 PREGUNTAS)
dataset_test = {
    "SQL": [

        "¿Cuál es el promedio de salinidad mensual en la estación E4 durante el año pasado?",
        "Extrae los datos de oxígeno disuelto de la boya de Lo Pagán para el mes de agosto.",
        "¿Qué día se midió el valor mínimo de pH en la Cubeta Norte?",
        "Filtra la tabla para mostrar solo los registros con turbidez superior a 15 NTU.",
        "¿Cuántas veces superó la temperatura los 29 grados en la boya de Los Alcázares?",
        "Calcula la precipitación total acumulada durante el episodio de DANA en San Javier.",
        "Muestra los registros de conductividad ordenados de mayor a menor.",
        "Dime el nivel de nitratos máximo detectado en la Rambla del Albujón.",
        "¿Cuál es la desviación estándar de la clorofila-a en todas las estaciones este año?",
        "Devuelve los datos de la boya de Islas Menores entre las 12:00 y las 16:00 horas de ayer.",
        "¿Cuántos registros nulos o vacíos hay en la columna de materia orgánica?",
        "Agrupa los datos por mes y calcula la temperatura máxima para cada uno.",
        "¿Cuál fue el último registro de nivel freático guardado en la base de datos?",
        "Busca las filas donde la salinidad sea menor a 35 PSU y el oxígeno menor a 3 mg/L.",
        "Obtén la media de todos los parámetros físicos medidos durante la primavera.",
        "Devuélveme la temperatura máxima y mínima registradas durante el mes de julio de cada año ",
        "Muestra un listado de todos los registros donde la precipitación acumulada fue mayor a 0 en 2025",
        "¿Cuál es la temperatura media registrada?",
        "Dime el valor máximo de salinidad.",
        "¿Cuántos registros hay en la tabla?",
        "¿Cuál es el valor mínimo de oxígeno disuelto?",
        "Muestra los datos de temperatura."
    ],

    "RAG": [

        "Según los artículos recientes, ¿cómo afecta el cambio climático a la estratificación de lagunas costeras?",
        "¿Qué dicen los informes sobre el impacto de los metales pesados en el bentos del Mar Menor?",
        "Explica la diferencia entre los algoritmos OC4 y C2RCC para la teledetección de clorofila.",
        "¿Qué papel juegan las praderas de Cymodocea nodosa en la oxigenación del agua según la literatura?",
        "Detalla la metodología de muestreo de campo recomendada por la Directiva Marco del Agua.",
        "¿Cuáles son los indicadores biológicos de eutrofización más utilizados en los estudios científicos?",
        "Según la teoría, ¿cómo influye el viento en la resuspensión de sedimentos y la turbidez?",
        "¿Qué estrategias de mitigación proponen los papers para reducir la entrada de nutrientes agrícolas?",
        "¿Cómo define la literatura científica el concepto de sopa verde en ecosistemas semicerrados?",
        "¿Qué precisión alcanzan los modelos de Random Forest para predecir la calidad del agua según los autores?",
        "Resume las conclusiones del estudio sobre la evolución histórica de la salinidad en el Mar Menor.",
        "¿Qué ventajas ofrece el satélite Sentinel-3 frente a Sentinel-2 para el estudio de aguas oceánicas según los expertos?",
        "Explica los mecanismos químicos de la liberación de fósforo desde el sedimento en condiciones de anoxia.",
        "¿Qué dice el CSIC sobre la capacidad de resiliencia del ecosistema lagunar tras una crisis distrófica?",
        "¿Cuáles son las fuentes principales de contaminación por nitrógeno documentadas en la bibliografía?",
        "¿Qué sensores se usan para la evaluar la presencia de clorofila-a?",
        "¿Cuál de los dos índices propuestos (IS2 o IS3) obtuvo un menor error (RMSE) en la validación y a qué se debe esto según los autores?",
        "¿Cómo influye el viento en la turbidez del agua?",
        "¿Qué causa la eutrofización en lagunas costeras?",
        "¿Qué factores afectan a la salinidad en ecosistemas costeros?",
        "¿Cómo se forman los episodios de anoxia en lagunas?",
        "¿Qué impacto tienen los nutrientes en la calidad del agua?"
    ],

    "HIBRIDO": [

        "Comprueba si la bajada de oxígeno de nuestra base de datos en agosto coincide con los eventos de mortandad descritos en los informes.",
        "Calcula la media de turbidez empírica y compárala con los límites de alerta establecidos por la legislación en los documentos.",
        "Aplica el umbral de clorofila-a sugerido por Delegido et al. a los registros de la boya E3 para ver cuántos días hubo riesgo de floración.",
        "¿Explican las teorías de estratificación térmica de los artículos los saltos bruscos de temperatura que vemos en la tabla?",
        "Cruza los picos de nitratos de nuestro CSV con los episodios de escorrentía agrícola documentados en la literatura.",
        "Verifica si la tendencia de salinidad de los últimos 5 años en los datos reales se ajusta a las predicciones de los modelos teóricos del paper.",
        "Basándote en los registros de la estación de San Javier, ¿es suficiente esta cantidad de lluvia para causar el fenómeno de la mancha blanca descrito por el CSIC?",
        "Analiza los valores extremos de pH en la tabla empírica e indica si la bibliografía reporta alteraciones similares en ecosistemas vecinos.",
        "¿Hay alguna correlación entre la conductividad de nuestros datos y las fases de eutrofización detalladas en el marco teórico?",
        "Extrae el mes con mayor concentración de nutrientes de la base de datos y busca en los artículos si es la época habitual de proliferación algal.",
        "Valida el algoritmo de teledetección del artículo usando las mediciones in situ de nuestra tabla de boyas.",
        "¿Justifica la teoría científica el aumento paulatino de materia orgánica que observamos en los registros de la Cubeta Sur?",
        "Compara el coeficiente de correlación entre temperatura y oxígeno de nuestros datos con el que obtuvieron los autores en su estudio.",
        "Según las definiciones del paper, clasifica el estado trófico actual del agua utilizando los valores promedios de nuestra base de datos.",
        "Investiga en la literatura si los niveles de metales pesados medidos empíricamente en nuestra tabla superan los umbrales de toxicidad.",
        "¿Estos niveles de oxígeno serían peligrosos para el ecosistema?",
        "¿Es normal esta temperatura del agua para esta época del año?",
        "¿Estos valores de salinidad indican algún problema ambiental?",
        "¿Podrían estos niveles de nutrientes causar eutrofización?",
        "¿Son preocupantes estos valores de turbidez observados en los datos?",
        "Calcula la humedad promedia diaria e investiga si tiene influencia en la presencia de clorofila-a",
        "Extrae los picos de temperatura máxima de nuestra base de datos empírica y explica, basándote en la literatura de Delegido, por qué es importante monitorizar la clorofila-a en episodios de calor extremo en aguas continentales."
    ]
}

if __name__ == "__main__":

    evaluador = EvaluadorEnrutador(dataset_tfg)

    ks = [1, 2, 3, 4, 5, 7, 10]

    campos = [
        "Pregunta", "Real", "Random", "Mejor_1NN", "Media_Total",
        "Score_SQL", "Score_RAG", "Score_HIBRIDO",
        "Correcto_Random", "Correcto_1NN", "Correcto_Media_Total" 
    ]

    # Añadimos las predicciones y sus columnas "Correcto_X"
    for k in ks:
        campos.append(f"Media_Top_{k}")
        campos.append(f"Correcto_Media_Top_{k}")
    for k in ks:
        campos.append(f"Voto_Top_{k}")
        campos.append(f"Correcto_Voto_Top_{k}")

    datos_csv = []

    total = sum(len(v) for v in dataset_test.values())
    contador = 1

    print(f"\nEvaluando {total} preguntas...\n")

    for categoria_real, preguntas in dataset_test.items():
        for pregunta in preguntas:

            print(f"[{contador}/{total}] {categoria_real}")

            (
                p1,
                p2,
                preds_media,
                preds_voto,
                p_random,
                scores
            ) = evaluador.evaluar_pregunta(pregunta)

            fila = {
                "Pregunta": pregunta,
                "Real": categoria_real,
                "Random": p_random,
                "Mejor_1NN": p1,
                "Media_Total": p2,
                "Score_SQL": scores.get("SQL", 0),
                "Score_RAG": scores.get("RAG", 0),
                "Score_HIBRIDO": scores.get("HIBRIDO", 0),
                
                # --- ACIERTOS BASE ---
                "Correcto_Random": 1 if p_random == categoria_real else 0,
                "Correcto_1NN": 1 if p1 == categoria_real else 0,
                "Correcto_Media_Total": 1 if p2 == categoria_real else 0
            }

            # Añadimos las predicciones literales de los K
            fila.update(preds_media)
            fila.update(preds_voto)

            # ACIERTOS TOP-K
            for k in ks:
                fila[f"Correcto_Media_Top_{k}"] = 1 if preds_media[f"Media_Top_{k}"] == categoria_real else 0
                fila[f"Correcto_Voto_Top_{k}"] = 1 if preds_voto[f"Voto_Top_{k}"] == categoria_real else 0

            datos_csv.append(fila)
            contador += 1


    nombre_archivo = "resultados_clasificador.csv"

    with open(nombre_archivo, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(datos_csv)

    print(f"\nResultados guardados en {nombre_archivo}")