from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

UMBRAL_CONFIANZA = 0.75
MARGEN_EMPATE = 0.012
TOP_K = 3
class ClasificadorIntencion:
    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True} 
        )

        self.ejemplos = {
            "SQL": [
                "¿Cuál fue el valor máximo o mínimo registrado?",
                "¿Cuántos registros superan este umbral en la tabla?",
                "Muéstrame los datos de la estación del mes de julio.",
                "Dame el promedio mensual y la desviación estándar.",
                "¿Qué día y hora exactos se registró el valor más extremo?",
                "Filtra los registros por variable y devuelve la tabla resultante."
            ],
            "RAG": [
                "¿Qué metodología, modelo empírico o algoritmo propone el estudio?",
                "Según el artículo, ¿cómo se define este fenómeno?",
                "¿Qué conclusiones o resultados principales extrae el paper?",
                "Según la literatura científica, ¿qué factores influyen en este proceso?",
                "¿Qué instrumentos o herramientas se recomiendan según el documento?",
                "Explícame la teoría descrita en el documento sobre este proceso."
            ],
            "HIBRIDO": [
                "¿Coinciden los datos medidos en nuestra tabla con lo que describe la literatura?",
                "Compara los registros reales de la estación con la teoría del artículo.",
                "¿Es normal el valor registrado hoy si nos basamos en los estudios científicos?",
                "¿Hay anomalías en nuestra base de datos respecto a los umbrales del paper?",
                "Valida los picos de nuestra tabla usando el modelo teórico del documento.",
                "¿Nuestra tabla empírica refleja los episodios descritos por los autores?",
                "¿Podemos usar la teoría del artículo para interpretar los datos de nuestra tabla?",
                "¿Los patrones de nuestra base de datos coinciden con las tendencias descritas en el paper?",
                "Cómo se relacionan los registros de nuestra tabla con los conceptos teóricos del documento?",
            ]
        }

        self.vectores_ejemplos = []
        for tipo, frases in self.ejemplos.items():
            for frase in frases:
                vector = self.embedder.embed_query("query: " + frase)
                self.vectores_ejemplos.append({
                    "tipo": tipo,
                    "texto": frase,
                    "vector": np.array(vector)
                })

    def similitud_coseno(self, a, b):
        """
        OPTIMIZACIÓN MATEMÁTICA: 
        Como usamos normalize_embeddings=True, la longitud de los vectores es 1.
        Por tanto,  np.dot equivale exactamente a la Similitud del Coseno,
        ahorrando ciclos de CPU.
        """
        return np.dot(a, b)

    def clasificar(self, pregunta):

        emb_pregunta = np.array(
            self.embedder.embed_query("query: " + pregunta)
        )

        resultados = []

        for item in self.vectores_ejemplos:
            score = self.similitud_coseno(
                emb_pregunta,
                item["vector"]
            )
            resultados.append({
                "tipo": item["tipo"],
                "score": float(score),
                "texto_match": item["texto"]
            })

        resultados = sorted(
            resultados,
            key=lambda x: x["score"],
            reverse=True
        )

        mejor_match = resultados[0]

        # ----------------------------------
        # calcular promedio por categoría
        # ----------------------------------

        scores_categoria = {"SQL": [], "RAG": [], "HIBRIDO": []}

        for r in resultados:
            scores_categoria[r["tipo"]].append(r["score"])

        scores = {}
        for categoria, lista in scores_categoria.items():
            top_k = sorted(lista, reverse=True)[:TOP_K]
            scores[categoria] = float(np.mean(top_k))

        tipo_detectado = max(scores, key=scores.get)
        mejor_score = scores[tipo_detectado]

        print(f"Match más cercano: '{mejor_match['texto_match']}' ({mejor_match['score']:.3f})")

        if mejor_score < UMBRAL_CONFIANZA:

            print (f"Baja confianza ({mejor_score:.2f}). Forzando ruta HÍBRIDA.")
        
            return "HIBRIDO", scores

        # ----------------------------------
        # empate SQL vs RAG (si no se ha detectado híbrido)
        # ----------------------------------
        if tipo_detectado != "HIBRIDO":
            if abs(scores["SQL"] - scores["RAG"]) < MARGEN_EMPATE:
                print(
                    f"Empate detectado "
                    f"(Dif: {abs(scores['SQL'] - scores['RAG']):.3f}). "
                    f"Forzando ruta HÍBRIDA."
                )
                return "HIBRIDO", scores
            
        return tipo_detectado, scores