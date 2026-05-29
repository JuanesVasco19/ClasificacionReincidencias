# Predicción de Reincidencia en Conducta Suicida — Municipio de Tunja, Boyacá

Proyecto de minería de datos y despliegue de modelo de Machine Learning para predecir
la reincidencia en intentos de suicidio (clasificación binaria: SI / NO).
Desarrollado como herramienta de apoyo a la decisión clínica para la Secretaría de Salud
del Municipio de Tunja, Boyacá.

---

## Estructura del repositorio

```
Despliegue_Reincidencias/
│
├── Preprocesamiento_De_Datos_Minería_CasoReincidenciaSuicidios.ipynb
├── Modelos_Predictivos_Reincidencias_Final.ipynb
├── Pipeline_Reincidencias.ipynb
│
└── proyecto_reincidencia/          # Aplicación de despliegue
    ├── app.py                      # Interfaz Streamlit
    ├── pipeline_reincidencia.py    # Módulo de preprocesamiento y predicción
    ├── database.py                 # Persistencia de evaluaciones en Supabase
    ├── supabase_client.py          # Cliente Supabase (local y Cloud)
    ├── modelo_reincidencia.pkl     # Modelo entrenado (no incluido en el repo)
    ├── requirements.txt            # Dependencias Python
    ├── .env                        # Variables de entorno (no incluido en el repo)
    └── assets/
        ├── OIP.jpg                 # Logo barra lateral
        └── upb-550x212.png         # Logo encabezado
```

---

## Notebooks

### 1. Preprocesamiento
`Preprocesamiento_De_Datos_Minería_CasoReincidenciaSuicidios.ipynb`

Limpieza y transformación del dataset original:
- Tratamiento de valores nulos y duplicados
- Codificación de variables categóricas
- Análisis exploratorio y distribución de clases

### 2. Entrenamiento y evaluación del modelo
`Modelos_Predictivos_Reincidencias_Final.ipynb`

Construcción y selección del modelo final:
- Comparación de clasificadores (Regresión Logística, Naive Bayes, Random Forest)
- Ensamblado con `VotingClassifier` (soft voting)
- Métricas de evaluación y validación cruzada
- Exportación del artefacto como `modelo_reincidencia.pkl`

### 3. Pipeline completo
`Pipeline_Reincidencias.ipynb`

Integración del flujo completo de extremo a extremo:
- Carga de datos crudos → preprocesamiento → predicción
- Verificación del artefacto guardado

---

## Artefacto del modelo

El archivo `modelo_reincidencia.pkl` se genera con `pickle` y contiene una lista de
**3 elementos**:

```python
artefacto = [modelo, labelencoder, variables]
pickle.dump(artefacto, open('modelo_reincidencia.pkl', 'wb'))
```

| Elemento       | Tipo                        | Descripción                                        |
|----------------|-----------------------------|----------------------------------------------------|
| `modelo`       | `VotingClassifier` (soft)   | Ensamble: LR + Naive Bayes calibrado + RF          |
| `labelencoder` | `LabelEncoder`              | Clases: `['NO', 'SI']`                             |
| `variables`    | `list` / `numpy.ndarray`    | Nombres de columnas después de aplicar get_dummies |

> El modelo no incluye scaler porque el dataset final no contiene variables numéricas
> continuas que requieran normalización.

---

## Aplicación de despliegue

### Requisitos previos
- Python 3.8 o superior
- Cuenta en [Supabase](https://supabase.com) con una tabla `evaluaciones_reincidencia`

### Instalación local

```bash
# 1. Ir a la carpeta de la app
cd proyecto_reincidencia

# 2. Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Variables de entorno

Crea el archivo `proyecto_reincidencia/.env` con las credenciales de Supabase:

```
SUPABASE_URL=https://<tu-proyecto>.supabase.co
SUPABASE_KEY=<tu-anon-key>
```

### Archivos que debes colocar manualmente

| Archivo | Descripción |
|---|---|
| `proyecto_reincidencia/modelo_reincidencia.pkl` | Artefacto generado por el notebook de entrenamiento |
| `proyecto_reincidencia/.env` | Credenciales de Supabase (ejecución local) |

### Ejecutar localmente

```bash
# Desde dentro de proyecto_reincidencia/
streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`.

---

## Despliegue en Streamlit Cloud

1. Sube el repositorio a GitHub (sin incluir `.env` ni `modelo_reincidencia.pkl`).
2. En [share.streamlit.io](https://share.streamlit.io), crea una nueva app apuntando a
   `proyecto_reincidencia/app.py`.
3. En **Settings → Secrets**, agrega las credenciales de Supabase en formato TOML:

```toml
SUPABASE_URL = "https://<tu-proyecto>.supabase.co"
SUPABASE_KEY = "<tu-anon-key>"
```

4. Para incluir el modelo, sube `modelo_reincidencia.pkl` al repositorio o a un
   bucket de Supabase Storage y ajusta `MODEL_PATH` en `app.py`.

---

## Modos de uso de la aplicación

### Pestaña 1 — Evaluación Individual

Formulario clínico organizado en cuatro secciones:

| Sección | Campos |
|---|---|
| I. Datos Sociodemográficos | Sexo, área de residencia, estrato, estado civil, escolaridad, régimen de seguridad social, gestante, población a cargo ICBF |
| II. Factores de Riesgo Clínico | Trastornos mentales (depresivo, personalidad, bipolar, esquizofrenia), antecedentes familiares, antecedente de violencia |
| III. Factores de Riesgo Psicosocial | Conflictos, problemas económicos/laborales/jurídicos/familiares, maltrato, consumo de sustancias, alcohol |
| IV. Características del Evento | Ideación suicida, plan organizado, método utilizado |

Al presionar **GENERAR EVALUACIÓN DE RIESGO** se muestra:
- Nivel de riesgo: **ALTO** / **MODERADO** / **BAJO**
- Probabilidad de reincidencia en porcentaje
- Recomendación clínica según el nivel de riesgo
- El registro se persiste automáticamente en Supabase

### Pestaña 2 — Evaluación por Lote

- Carga de archivo `.csv` con múltiples registros
- Las columnas deben coincidir con el formato del dataset de entrenamiento
- Muestra métricas generales, tabla de resultados y gráfico de distribución
- Permite descargar el reporte completo como CSV
- Los registros se persisten automáticamente en Supabase

---

## Notas técnicas

- El modelo se carga una sola vez con `@st.cache_resource`.
- Las rutas de archivos se resuelven con `pathlib.Path(__file__).parent` para
  funcionar tanto en local como en Streamlit Cloud.
- Las variables booleanas (SI/NO) se transforman con `pd.get_dummies(drop_first=True)`
  usando categorías fijas (`pd.Categorical`) para garantizar columnas estables
  independientemente del orden de los datos.
- Las variables multicategoría se transforman con `pd.get_dummies(drop_first=False)`.
- Las columnas faltantes en el input se rellenan con `0` vía `reindex`.
- El módulo `pipeline_reincidencia.py` puede usarse de forma independiente:

```python
from pipeline_reincidencia import predecir
import pandas as pd

df = pd.read_csv("nuevos_datos.csv")
resultado = predecir(df)
print(resultado[["PREDICCION", "PROBABILIDAD_REINCIDENCIA"]])
```
