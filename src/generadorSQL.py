import os

import duckdb
import re
from langchain_ollama import OllamaLLM

MAX_INTENTOS = 5
MODELO_LLM = "llama3"

class ConsultaSQL:
    def __init__(self, ruta_csv):
        """Inicializamos la conexión a DuckDB y el LLM."""
        self.ruta_csv = ruta_csv
        self.llm = OllamaLLM(model=MODELO_LLM, temperature=0.0, num_ctx=8192)
        self.con = duckdb.connect(database=':memory:')
        self.cargar_datos()
        self.esquema = self.obtener_esquema()
        self.ultimas_tablas_usadas = []
        self.system_prompt = self.construir_prompt_sistema()

    def construir_prompt_sistema(self):
        """
        Construye el system prompt aplicando técnicas de role prompting,
        ingeniería de contexto y few-shot prompting.
        """
        return f"""
        # IDENTIDAD
         Eres un Arquitecto de Bases de Datos de élite y un Ingeniero de Datos especializado en DuckDB y SQL OLAP de alto rendimiento. 
        Posees un conocimiento profundo de la ejecución de consultas vectorizadas, la optimización del almacenamiento en columnas y el dialecto SQL específico de DuckDB. 
        Tu objetivo principal es generar consultas SQL precisas, eficientes y robustas para responder a las preguntas de los usuarios basándote en los conjuntos de datos proporcionados. 
        Valoras la corrección técnica por encima de la velocidad y siempre verificas las suposiciones sobre el esquema antes de realizar consultas.
        
        # ESQUEMA:
        {self.esquema}

        # CAPACIDADES Y HERRAMIENTAS (ESPECÍFICAS DE DUCKDB)
        1.  **Conciencia del Esquema:** NUNCA adivines nombres. Las tablas disponibles son exactamente las del esquema inyectado.
        2.  **Restricción Absoluta de Columnas:** 
            - Si una columna mencionada en la pregunta NO existe en el esquema, NO generes SQL.
            - Explica claramente que la columna no está disponible.
            - NO inventes nombres similares.
            
        3.  **Sintaxis Avanzada:** Prefieres características de DuckDB:
            * * `GROUP BY ALL` (Úsalo SOLO, sin listar columnas después) para simplificar agregaciones complejas.
            * CTEs (`WITH ...`) para dividir lógica compleja y hacerla legible.
            * Funciones de ventana (`ROW_NUMBER`, `LEAD`) para análisis de series temporales.
            * Alias `AS` claros para las columnas resultantes.
        
        # REGLAS OPERATIVAS (CRÍTICAS)
        1.  **Piensa Antes de Codificar:** Usa un bloque <thought_process> para planificar la consulta.
        2.  **Identificadores:** SIEMPRE usa comillas dobles `""` para columnas con espacios o símbolos: "Temperature (°C)".
        3.  **Estrategia de Búsqueda:** - Si buscan "el día con mayor X", NO uses `WHERE col = (SELECT MAX...)`. 
            - USA: `ORDER BY col DESC LIMIT 1`.

        4.  **REGLA DE CONTEXTO COMPLETO:** - Para eventos extremos (máximos, mínimos), DEBES devolver toda la información de esa fila usando `SELECT *`.
            - Esto asegura que el orquestador final reciba la fecha, la hora y el valor exacto simultáneamente.

        5.  **Consistencia en Agregaciones (OBLIGATORIO):**
            - Si utilizas MAX, MIN, AVG, SUM o COUNT:
                * Todas las demás columnas seleccionadas deben estar en GROUP BY
                * O usa ORDER BY ... LIMIT 1
            - NUNCA mezcles columnas no agregadas con funciones agregadas sin GROUP BY.

        6.  **Manejo de Errores:** Si es una corrección, analiza el error paso a paso en el razonamiento.
        7.  **Seguridad:** PROHIBIDO usar DROP, DELETE, INSERT.

        # FORMATO DE SALIDA
        1. Bloque <thought_process> ... </thought_process>
        2. Bloque de código SQL dentro de ```sql ... ```
        3. Breve explicación técnica de *por qué* elegiste esa estrategia (fuera del bloque de código).
        """
    
    def cargar_datos(self):
        """Montamos las 5 tablas."""
        tablas = {
            "AEMET": "AEMET_unificada_limpia.csv",
            "Boyas_Nautilus": "Boyas_Nautilus_unificadas_limpias.csv",
            "Boyas_IMIDA_UPCT": "BoyasProf_IMIDA_UPCT_unificadas_limpias.csv",
            "SAIH_Piezometros": "SAIH_Piezometros_unificados_limpios.csv",
            "SAIH_Ramblas": "SAIH_ramblas_unificadas_limpias.csv"
        }
        
        for nombre_tabla, archivo in tablas.items():
            ruta = os.path.join(self.ruta_csv, archivo)
            if os.path.exists(ruta):
                self.con.execute(f"CREATE VIEW {nombre_tabla} AS SELECT * FROM read_csv_auto('{ruta}')")
            else:
                print(f" Advertencia: No se encontró {ruta}")

    def obtener_esquema(self):
        """Extraemos los esquemas de TODAS las tablas para que el LLM pueda hacer JOINs."""
        tablas = [
            "AEMET", 
            "Boyas_Nautilus", 
            "Boyas_IMIDA_UPCT", 
            "SAIH_Piezometros", 
            "SAIH_Ramblas"
        ]
        
        esquema_total = "BASE DE DATOS 'MAR MENOR':\n\n"
        
        for t in tablas:
            try:
                desc = self.con.execute(f"DESCRIBE {t}").df().to_string()
                esquema_total += f"--- TABLA: {t} ---\n{desc}\n\n"
            except Exception as e:
                print(f"Error al describir la tabla {t}: {e}")
                
        return esquema_total
    
    def limpiar_respuesta(self, respuesta):

        match = re.search(r'```sql\s*(.*?)(?:```|$)', respuesta, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            return match.group(1).strip()

        # Fallback: acepta WITH o SELECT, ; opcional
        match_sql = re.search(r'((?:WITH|SELECT)\b.*)', respuesta, re.DOTALL | re.IGNORECASE)
        if match_sql:
            sql = match_sql.group(1).strip()
            if ';' in sql:
                sql = sql[:sql.rfind(';') + 1]  # último ;, no el primero
            return sql
        return None

    def extraer_tablas(self, sql):
        """Extrae los nombres de las tablas usadas en la consulta SQL."""
        # Busca palabras que van después de FROM o JOIN
        tablas = re.findall(r"(?:FROM|JOIN)\s+[\"']?([a-zA-Z0-9_]+)[\"']?", sql, re.IGNORECASE)
        self.ultimas_tablas_usadas = list(set(tablas))
        return self.ultimas_tablas_usadas
    
    def generar_sql(self, pregunta, error_previo=None, sql_fallido=None):

        ejemplos = """
        EJEMPLOS DE REFERENCIA (ESTILO DE PENSAMIENTO Y CÓDIGO):

        Caso 1: Extremo, Contexto Completo y Selección de Tabla
        Usuario: "¿Cuál fue la temperatura máxima en la estación de la AEMET y cuándo ocurrió?"
        Asistente:
        <thought_process>
        El usuario pide el valor máximo de temperatura y su contexto temporal. Específicamente menciona "AEMET", así que usaré la tabla 'AEMET'.
        En lugar de usar MAX() que perdería el contexto temporal y daría error de GROUP BY, ordenaré por la columna "Temperature (°C)" de forma descendente y tomaré la primera fila con LIMIT 1. 
        Usaré SELECT * para traer todas las columnas de ese evento.
        </thought_process>
        ```sql
        SELECT * FROM AEMET ORDER BY "Temperature (°C)" DESC LIMIT 1;
        ```

        Caso 2: Agrupación Temporal
        Usuario: "¿Cuánto llovió en total cada día en la rambla?"
        Asistente:
        <thought_process>
        El usuario pide un sumatorio agrupado por día y menciona "rambla", por lo que usaré la tabla 'SAIH_Ramblas'. 
        Usaré SUM en la precipitación y agruparé por la columna "Date".
        </thought_process>
        ```sql
        SELECT "Date", SUM("Precipitation (mm)") AS total_lluvia FROM SAIH_Ramblas GROUP BY "Date" ORDER BY "Date";
        ```

        Caso 3: Múltiples extremos (UNION con contexto completo)
        Usuario: "Dime el registro del momento de mayor calor y el de mayor humedad en la boya nautilus"
        Asistente:
        <thought_process>
        Pide dos eventos extremos independientes de la tabla 'Boyas_Nautilus'. Necesito un UNION ALL.
        Para cumplir la regla de contexto completo, usaré SELECT * en ambos, añadiendo una columna 'Tipo' para diferenciarlos.
        Es obligatorio usar paréntesis ( ) alrededor de cada SELECT con LIMIT.
        </thought_process>
        ```sql
        (SELECT *, 'Mayor Calor' AS Tipo FROM Boyas_Nautilus ORDER BY "Temperature (°C)" DESC LIMIT 1) 
        UNION ALL 
        (SELECT *, 'Mayor Humedad' AS Tipo FROM Boyas_Nautilus ORDER BY "relative Humidity (%)" DESC LIMIT 1);
        ```
        """
        prompt = f"""

        {self.system_prompt}

        # Ejemplos de razonamiento y código SQL para diferentes tipos de preguntas
        {ejemplos}

        # PREGUNTA: {pregunta}
        """

        if error_previo:
            prompt += "\n\nTu último intento de responder falló.\n"
            if sql_fallido:
                prompt += (
                    "\n- Consulta SQL que escribiste:\n"
                    "```sql\n"
                    f"{sql_fallido}\n"
                    "```\n"
                )
            prompt += (
                "\n- Problema detectado:\n"
                f"{error_previo}\n\n"
                "Analiza la causa en tu <thought_process> y devuelve una consulta "
                "corregida usando los nombres de tabla EXACTOS del esquema.\n"
            )

        prompt += "\nTU RESPUESTA (Thinking + SQL + Explicación):"

        return self.llm.invoke(prompt)

    def es_seguro(self, sql):
        """
        Verifica la seguridad de la consulta mediante análisis de tokens.
        Extrae todas las palabras (alfanuméricas) de la consulta y comprueba
        mediante intersección de conjuntos que no haya comandos prohibidos.
        """
        if not sql:
            return False
            
        sql_limpio = re.sub(r'--.*|/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        sql_limpio = sql_limpio.replace("''", "") 
        sql_limpio = re.sub(r"'[^']*'", "", sql_limpio)
        
        tokens = re.findall(r'\w+', sql_limpio.upper())
        tokens_set = set(tokens)
        
        prohibidos = {"DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE"}
        
        tiene_select = "SELECT" in tokens_set
        tiene_prohibidos = bool(tokens_set & prohibidos)
        
        return tiene_select and not tiene_prohibidos

    def _detectar_truncamiento(self, respuesta):
        """Abrió ```sql pero no lo cerró -> respuesta cortada."""
        abre = re.search(r'```sql', respuesta, re.IGNORECASE)
        cierra = re.search(r'```sql\s*.*?```', respuesta, re.DOTALL | re.IGNORECASE)
        return bool(abre) and not bool(cierra)

    def _mostrar_razonamiento(self, respuesta, etiqueta):
        m = re.search(r'<thought_process>(.*?)</thought_process>', respuesta, re.DOTALL)
        if m:
            print(f"\n {etiqueta}:\n{m.group(1).strip()}\n")

    def generar_y_ejecutar(self, pregunta):
        self.ultimas_tablas_usadas = []

        max_intentos = MAX_INTENTOS
        error_previo = None     # feedback de texto para el siguiente intento
        sql_fallido = None      # SQL del intento anterior (solo si llegó a ejecutarse)
        ultimo_error = None

        for intento in range(max_intentos):
            respuesta_raw = self.generar_sql(pregunta, error_previo=error_previo, sql_fallido=sql_fallido)
            etiqueta = "RAZONAMIENTO" if intento == 0 else "CORRECCIÓN DEL RAZONAMIENTO"
            #self._mostrar_razonamiento(respuesta_raw, etiqueta)

            # Comprobamos si la respuesta ha sido cortada
            if self._detectar_truncamiento(respuesta_raw):
                ultimo_error = "Respuesta truncada: ```sql sin cerrar."
                error_previo = ("Tu respuesta anterior se cortó: el bloque ```sql quedó sin "
                                "cerrar. Sé más conciso en <thought_process> y devuelve la "
                                "consulta completa en un único bloque ```sql ... ``` cerrado.")
                sql_fallido = None
                continue

            # Extraer SQL
            sql = self.limpiar_respuesta(respuesta_raw)
            if not sql:
                ultimo_error = "No se devolvió SQL parseable."
                error_previo = ("No pude extraer ninguna consulta SQL de tu respuesta. "
                                "Devuélvela dentro de un bloque ```sql ... ``` cerrado.")
                sql_fallido = None
                continue

            #  Fallo de seguridad -> parada dura, sin reintento
            if not self.es_seguro(sql):
                raise ValueError("La consulta SQL generada no es segura. "
                                "Se detectaron comandos prohibidos.")

            # Ejecutar
            try:
                self.extraer_tablas(sql)
                return self.con.execute(sql).df()
            except Exception as e:
                ultimo_error = e
                error_previo = str(e)
                sql_fallido = sql
                continue

        raise RuntimeError(f"No se pudo resolver la consulta tras {max_intentos} "
                        f"intentos. Último error: {ultimo_error}")

    def generar_respuesta_natural(self, pregunta, df_resultado):
        """Genera una respuesta natural. """
        if isinstance(df_resultado, str): return df_resultado

        if df_resultado.empty:
            return "La consulta no devolvió datos (tabla vacía)."
        
        datos = df_resultado.to_string(index=False)
        return self.llm.invoke(f"Resume brevemente para el usuario: '{pregunta}'. Datos:\n{datos}").strip()