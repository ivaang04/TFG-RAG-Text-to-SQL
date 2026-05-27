# Sistema híbrido SQL+RAG para consultas sobre fuentes de datos abiertas

Trabajo Fin de Grado · Grado en Ingeniería Informática · Universidad de Murcia · 2026

Autor: **Iván Guillamón Baños**
Tutores: **Aurora González Vidal**, **Antonio Martínez Ibarra**

---

## Descripción

Este repositorio contiene el código fuente y los materiales experimentales del Trabajo Fin de Grado titulado *"Fine-tuning de Modelos de Lenguaje Grandes (LLMs) para las consultas sobre fuentes de datos abiertas: un sistema híbrido basado en RAG, embeddings y generación de SQL"*.

El sistema integra dos paradigmas en un único pipeline ejecutado de forma local:

- **Text-to-SQL** para consultar series temporales de monitorización ambiental almacenadas en DuckDB.
- **Retrieval-Augmented Generation (RAG)** para recuperar fragmentos relevantes de un corpus de literatura científica.

Un clasificador de intención basado en similitud semántica enruta cada consulta del usuario hacia el módulo adecuado (SQL, RAG o ambos), y un orquestador central coordina los componentes y genera la respuesta final.

La validación se ha llevado a cabo sobre el ecosistema del Mar Menor, combinando datos empíricos de las redes de sensores desplegadas en la laguna y su cuenca con un corpus de literatura científica que incluye tanto estudios específicos de la laguna como publicaciones del ámbito de las ciencias marinas.

## Arquitectura

```
Consulta del usuario
        │
        ▼
┌─────────────────────┐
│ Clasificador de     │
│ intención           │
│ (embeddings + KNN)  │
└──────────┬──────────┘
           │
   ┌───────┴───────┐
   │               │
   ▼               ▼
┌──────┐       ┌──────┐
│ SQL  │       │ RAG  │
└──┬───┘       └──┬───┘
   │              │
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ Orquestador  │
   │ (Llama 3)    │
   └──────┬───────┘
          ▼
     Respuesta
```

## Requisitos

- **Sistema operativo**: Linux (probado en Ubuntu sobre WSL). Funciona también en macOS.
- **Python**: 3.12 o superior.
- **Memoria RAM**: 16 GB recomendados (el modelo Llama 3 8B requiere ~5 GB).
- **Disco**: ~10 GB libres para modelos, base de datos vectorial y datos.
- **Ollama**: instalado y en ejecución. [Instrucciones aquí](https://ollama.com/download).

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/ivaang04/TFG-Fine-Tunning.git
cd TFG-Fine-Tunning
```

### 2. Crear un entorno virtual e instalar dependencias

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Descargar el modelo Llama 3 vía Ollama

```bash
ollama pull llama3
```

### 4. Verificar el modelo

```bash
ollama list
```

Deberías ver `llama3:latest` en la lista.

## Estructura del repositorio

```
TFG-Fine-Tunning/
├── limpiar_datos/                       # Scripts ETL de preprocesamiento de datos
│   ├── limpiar_datos_boyas_Nautilus.py
│   ├── limpiar_datos_boyas_prof_IMIDA.py
│   ├── limpiar_datos_boyas_UPCT.py
│   ├── limpiar_datos_ramblas.py
│   ├── limpiar_datos_SAIH_piezometros.py
│   └── union_Prof_IMIDA_UPCT.py
├── resultados/                          # Gráficas y resultados experimentales
│   ├── GRAFICAS-CLASIFICADOR/
│   └── GRAFICAS-RAG/
├── src/                                 # Código fuente principal
│   ├── evaluadores/                     # Modelos juez y scripts de evaluación
│   ├── clasificadorIntencion.py         # Clasificador de intención
│   ├── generadorSQL.py                  # Módulo SQL
│   ├── orquestador.py                   # Orquestador central
│   └── RAG.py                           # Módulo RAG
├── datos/                               # Datos empíricos y corpus documental (no incluidos)
├── chroma_db/                           # Base de datos vectorial (generada automáticamente)
├── requirements.txt                     # Dependencias Python
├── LICENSE                              # Licencia MIT
└── README.md                            # Este fichero
```

## Datos

Por motivos de tamaño y de las restricciones de difusión asociadas a las fuentes originales, los datos empíricos y el corpus de literatura científica **no se incluyen en este repositorio**. Para reproducir los experimentos:

- **Datos empíricos**: provienen del proyecto OMM-Azul del IMIDA y de las redes públicas de AEMET y SAIH. Los scripts en `limpiar_datos/` permiten regenerar los ficheros CSV unificados a partir de los datos en bruto.
- **Corpus documental**: 36 documentos científico-técnicos en formato PDF, procedentes de bases de datos académicas y repositorios institucionales (IEO-CSIC, CARM, entre otros). Las fuentes concretas se detallan en la memoria del TFG.

Una vez obtenidos los datos, deben colocarse en una carpeta `datos/` con la siguiente estructura:

```
datos/
├── csv/                    # Ficheros CSV resultantes del preprocesamiento
└── Docs/                   # PDFs del corpus de literatura científica
```

## Uso

### Ejecutar el sistema completo

```bash
python src/orquestador.py
```

El sistema lanzará una interfaz por terminal donde podrás formular preguntas en lenguaje natural. Para salir, escribe `salir`.

### Ejemplo de uso

```
> ¿Cuál fue el pH medio de la estación E3 en 2023?
[Modo SQL]
Respuesta: El pH medio de la estación E3 en 2023 fue de 8.64...

> ¿Qué dicen los autores sobre la eutrofización del Mar Menor?
[Modo RAG]
Respuesta: Según los estudios consultados, la eutrofización...
```

### Ejecutar módulos individualmente

```bash
python src/RAG.py                       # Solo el módulo RAG
python src/generadorSQL.py              # Solo el módulo SQL
python src/clasificadorIntencion.py     # Solo el clasificador
```

## Reproducir los experimentos

Los scripts de evaluación se encuentran en `src/evaluadores/`. Los resultados generados se almacenan en `resultados/` como gráficas, organizadas en dos carpetas:

- `resultados/GRAFICAS-CLASIFICADOR/` — Resultados del clasificador de intención.
- `resultados/GRAFICAS-RAG/` — Resultados del módulo RAG.

Para reproducir los resultados presentados en el capítulo 5 de la memoria, ejecuta los scripts correspondientes desde la raíz del repositorio.

## Preprocesamiento de datos

Los scripts de la carpeta `limpiar_datos/` implementan el proceso ETL descrito en la sección 4.2 de la memoria. Cada script procesa una fuente de datos distinta:

| Script | Fuente de datos |
|---|---|
| `limpiar_datos_boyas_Nautilus.py` | Boyas NAUTILUS de la UPCT |
| `limpiar_datos_boyas_prof_IMIDA.py` | Boyas a profundidad fija del IMIDA |
| `limpiar_datos_boyas_UPCT.py` | Boyas de la UPCT |
| `limpiar_datos_ramblas.py` | Aforadores de ramblas del SAIH |
| `limpiar_datos_SAIH_piezometros.py` | Piezómetros del SAIH |
| `union_Prof_IMIDA_UPCT.py` | Integración de boyas IMIDA y UPCT |

## Tecnologías utilizadas

| Componente | Tecnología | Versión |
|---|---|---|
| Lenguaje | Python | 3.12 |
| Base de datos analítica | DuckDB | 1.4.4 |
| Base de datos vectorial | ChromaDB | 1.4.1 |
| Framework de orquestación | LangChain | 1.2.9 |
| Modelo de embeddings | intfloat/multilingual-e5-small | — |
| Modelo de lenguaje | Llama 3 (8B) | latest |
| Plataforma de inferencia | Ollama | 0.15.4 |

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Consulta el fichero [LICENSE](LICENSE) para más detalles.
