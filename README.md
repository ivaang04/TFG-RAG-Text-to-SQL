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
│ (embeddings)        │
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

- **Sistema operativo**: Linux (probado en Ubuntu 22.04+). Funciona también en macOS.
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
python3.12 -m venv .venv
source .venv/bin/activate
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
├── src/
│   ├── clasificador.py        # Clasificador de intención
│   ├── consulta_sql.py        # Módulo SQL
│   ├── rag.py                 # Módulo RAG
│   ├── orquestador.py         # Orquestador central
│   └── juez_rag.py            # Modelo juez para evaluación
├── datos/
│   ├── csv/                   # Datos empíricos preprocesados
│   └── docs/                  # Corpus de literatura científica (PDFs)
├── evaluacion/
│   ├── conjuntos_prueba/      # Test sets del clasificador y RAG
│   └── scripts/               # Scripts de evaluación experimental
├── requirements.txt           # Dependencias Python
├── LICENSE                    # Licencia MIT
└── README.md                  # Este fichero
```

> Ajusta esta sección según tu estructura real de carpetas.

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

## Reproducir los experimentos

Los scripts de evaluación del clasificador y del módulo RAG se encuentran en `evaluacion/scripts/`. Para reproducir los resultados presentados en el capítulo 5 de la memoria:

```bash
python evaluacion/scripts/eval_clasificador.py
python evaluacion/scripts/eval_rag.py
```

Los resultados se generan en formato CSV y se guardan en `evaluacion/resultados/`.

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

## Memoria del TFG

La memoria completa del trabajo está disponible en formato PDF en [enlace al PDF si lo subes al repo o al repositorio institucional de la Universidad].

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Consulta el fichero [LICENSE](LICENSE) para más detalles.
