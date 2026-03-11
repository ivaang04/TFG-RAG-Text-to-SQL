import duckdb
import re
from langchain_ollama import OllamaLLM

class ConsultaSQL:
    def __init__(self, ruta_csv):
        """Inicializamos la conexión a DuckDB y el LLM."""
        self.ruta_csv = ruta_csv
        self.llm = OllamaLLM(model="llama3", temperature = 0.0)
        self.con = duckdb.connect(database=':memory:')
        self.cargar_datos()
        self.esquema = self.obtener_esquema()
        self.system_prompt = self.construir_prompt_sistema()

    def construir_prompt_sistema(self):
        """
        Sigue los principios de 'Anclaje de Persona' e 'Ingeniería de Contexto'.
        """
        return f"""
        # IDENTIDAD
         Eres un Arquitecto de Bases de Datos de élite y un Ingeniero de Datos especializado en DuckDB y SQL OLAP de alto rendimiento. 
        Posees un conocimiento profundo de la ejecución de consultas vectorizadas, la optimización del almacenamiento en columnas y el dialecto SQL específico de DuckDB. 
        Tu objetivo principal es generar consultas SQL precisas, eficientes y robustas para responder a las preguntas de los usuarios basándote en los conjuntos de datos proporcionados. 
        Valoras la corrección técnica por encima de la velocidad y siempre verificas las suposiciones sobre el esquema antes de realizar consultas.
        
        # TABLA: 'datos'
        # ESQUEMA:
        {self.esquema}

        # CAPACIDADES Y HERRAMIENTAS (ESPECÍFICAS DE DUCKDB)
        1.  **Conciencia del Esquema:** NUNCA adivines nombres. La tabla es 'datos'.
        2.  **Restricción Absoluta de Columnas:** 
            - Si una columna mencionada en la pregunta NO existe en el esquema, 
            NO generes SQL.
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
        """Creamos una vista para poder consultar el CSV usando consultas SQL."""
        try:
            self.con.execute(f"CREATE VIEW datos AS SELECT * FROM read_csv_auto('{self.ruta_csv}', skip=4)")
        except Exception as e:
            print(f"Error crítico cargando el CSV: {e}")

    def obtener_esquema(self):
        """Extraemos los nombres de columnas y tipos para que el LLM los conozca."""
        return self.con.execute("DESCRIBE datos").df().to_string()
    
    def limpiar_respuesta(self, respuesta):
        match = re.search(r'```sql\s*(.*?)\s*```', respuesta, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Buscar primer SELECT real
        match_select = re.search(r'(SELECT .*?;)', respuesta, re.DOTALL | re.IGNORECASE)
        if match_select:
            return match_select.group(1).strip()

        return None

    def generar_sql(self, pregunta, error_previo=None):

        ejemplos = """
        EJEMPLOS DE REFERENCIA (ESTILO DE PENSAMIENTO Y CÓDIGO):

        Caso 1: Extremo y Contexto Completo
        Usuario: "¿Cuál fue la temperatura máxima y cuándo ocurrió?"
        Asistente:
        <thought_process>
        El usuario pide el valor máximo y su contexto temporal (fecha y hora). 
        En lugar de usar MAX() que perdería el contexto, ordenaré por temperatura de forma descendente y tomaré la primera fila. 
        Usaré SELECT * para traer todas las columnas de ese evento.
        </thought_process>
        ```sql
        SELECT * FROM datos ORDER BY "Temperature (°C)" DESC LIMIT 1;
        ```

        Caso 2: Agrupación Temporal
        Usuario: "¿Cuánto llovió en total cada día?"
        Asistente:
        <thought_process>
        El usuario pide un sumatorio agrupado por día. 
        Usaré SUM en la precipitación y agruparé por la columna Date.
        </thought_process>
        ```sql
        SELECT "Date", SUM("accumulated Precipitation (0.1 mm)") AS total_lluvia FROM datos GROUP BY "Date" ORDER BY "Date";
        ```

        Caso 3: Múltiples extremos (UNION con contexto completo)
        Usuario: "Dime el registro del momento de mayor calor y el de mayor humedad"
        Asistente:
        <thought_process>
        Pide dos eventos extremos independientes. Necesito un UNION ALL.
        Para cumplir la regla de contexto completo, usaré SELECT * en ambos, añadiendo una columna 'Tipo' para diferenciarlos.
        Es obligatorio usar paréntesis ( ) alrededor de cada SELECT con LIMIT.
        </thought_process>
        ```sql
        (SELECT *, 'Mayor Calor' AS Tipo FROM datos ORDER BY "Temperature (°C)" DESC LIMIT 1) 
        UNION ALL 
        (SELECT *, 'Mayor Humedad' AS Tipo FROM datos ORDER BY "relative Humidity (%)" DESC LIMIT 1);
        ```
        """
        prompt = f"""

        {self.system_prompt}

        # Ejemplos de razonamiento y código SQL para diferentes tipos de preguntas
        {ejemplos}

        # PREGUNTA: {pregunta}
        """
        
        if error_previo:
            prompt += f"\n Tu intento anterior falló con: {error_previo}. Analiza el error en tu razonamiento y corrige el SQL."

        prompt += "\nTU RESPUESTA (Thinking + SQL + Explicación):"
                
        return self.llm.invoke(prompt)

    def es_seguro(self, sql):
        if not sql:
            return False
        sql_clean = sql.upper()
        prohibidos = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE"]
        return "SELECT" in sql_clean and not any(p in sql_clean for p in prohibidos)

    def generar_y_ejecutar(self, pregunta):
    
        
        print(f"\n Se está procesando la pregunta: '{pregunta}'...")
        respuesta_raw = self.generar_sql(pregunta)
        
        #Se muestra el razonamiento
        razonamiento = re.search(r'<thought_process>(.*?)</thought_process>', respuesta_raw, re.DOTALL)
        if razonamiento:
            print(f"\n RAZONAMIENTO:\n{razonamiento.group(1).strip()}\n")
            
        sql = self.limpiar_respuesta(respuesta_raw)
    
        if not sql:
            raise ValueError("El agente no pudo generar una consulta SQL válida.")

        # Bucle de intentos
        max_intentos = 3
        ultimo_error = None
        for intento in range(max_intentos):
            try:
                if not self.es_seguro(sql): 
                    raise ValueError("La consulta SQL generada no es segura. Se detectaron comandos prohibidos.")
                
                # Ejecución exitosa
                return self.con.execute(sql).df()
            
            except ValueError as ve:
                # Si es un error de seguridad, lo re-lanzamos hacia arriba (no se intenta corregir)
                raise ve
            
            except Exception as e:
                # Aquí capturamos los errores que no han sido capturados en el anterior catch, es decir, errores de DuckDB.
                # Sintaxis, columnas que no existen, etc.
                # Estos errores los intentaremos corregir. 
                print(f"Intento {intento+1} fallido: {e}")
                ultimo_error = e
                if intento < max_intentos - 1:
                    print("Auto-corrigiendo...")
                    #Enviamos el error de vuelta
                    # anadir consulta que ha generado error
                    respuesta_corr = self.generar_sql(pregunta, error_previo=str(e))
                    sql = self.limpiar_respuesta(respuesta_corr)

                    if not sql:
                        continue # Si no se pudo generar un SQL válido, intentamos de nuevo sin enviar el error (reiniciando el proceso)
                    
                    # Mostrar razonamiento de corrección
                    razonamiento_correcto = re.search(r'<thought_process>(.*?)</thought_process>', respuesta_corr, re.DOTALL)
                    if razonamiento_correcto:
                        print(f"CORRECCIÓN DEL RAZONAMIENTO: {razonamiento_correcto.group(1).strip()}")
                
        raise RuntimeError(f"No se pudo resolver la consulta tras {max_intentos} intentos. Último error: {ultimo_error}")

    def generar_respuesta_natural(self, pregunta, df_resultado):
        """Genera una respuesta natural. """
        if isinstance(df_resultado, str): return df_resultado

        if df_resultado.empty:
            return "La consulta no devolvió datos (tabla vacía)."
        
        datos = df_resultado.to_string(index=False)
        return self.llm.invoke(f"Resume brevemente para el usuario: '{pregunta}'. Datos:\n{datos}").strip()