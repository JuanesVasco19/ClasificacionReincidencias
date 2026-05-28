# Sistema de Prediccion de Reincidencia en Intentos de Suicidio

Aplicacion Streamlit para desplegar el modelo de Machine Learning que predice
reincidencia en intentos de suicidio (clasificacion binaria: SI / NO).

---

## Estructura de archivos requerida

```
proyecto_reincidencia/
├── app.py                    # Aplicacion Streamlit
├── pipeline_reincidencia.py  # Modulo de preprocesamiento y prediccion
├── modelo_reincidencia.pkl   # <-- COLOCA AQUI tu modelo entrenado
├── requirements.txt          # Dependencias Python
└── README.md                 # Este archivo
```

> **Importante:** el archivo `modelo_reincidencia.pkl` NO esta incluido en el
> repositorio. Debes copiarlo manualmente a esta carpeta antes de ejecutar la app.

---

## Instalacion

### 1. Requisitos previos
- Python 3.8 o superior
- pip

### 2. Crear entorno virtual (recomendado)

```bash
# Crear entorno
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en Linux / Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Uso

### Ejecutar la aplicacion

Desde la carpeta `proyecto_reincidencia/`:

```bash
streamlit run app.py
```

La aplicacion se abre automaticamente en el navegador en
`http://localhost:8501`.

---

## Modos de uso

### Modo 1 — Prediccion por Lote (CSV)

1. Abre la pestana **"Prediccion por Lote (CSV)"**.
2. Sube un archivo `.csv` con los datos de los pacientes.
3. El CSV debe tener las mismas columnas que el dataset de entrenamiento
   (los nombres exactos, incluyendo espacios).
4. La app mostrara:
   - Tabla de resultados con columnas `PREDICCION` y `PROBABILIDAD_REINCIDENCIA`.
   - Metricas: total de casos, reincidentes predichos y porcentaje.
   - Grafico de barras con la distribucion de predicciones.
   - Boton para descargar los resultados como CSV.

### Modo 2 — Prediccion Individual

1. Abre la pestana **"Prediccion Individual"**.
2. Completa todos los campos del formulario:
   - Datos demograficos (sexo, area de residencia, etc.)
   - Factores psicosociales (problemas economicos, conflictos, etc.)
   - Diagnosticos psiquiatricos
   - Metodo del intento
3. Haz clic en **Predecir**.
4. La app mostrara:
   - Etiqueta **REINCIDENTE** (rojo) o **NO REINCIDENTE** (verde).
   - Barra de progreso con la probabilidad de reincidencia.
   - Texto de interpretacion segun el nivel de riesgo:
     - >= 70 %: alto riesgo — intervencion inmediata.
     - 50–70 %: riesgo moderado — seguimiento.
     - < 50 %: riesgo bajo — monitoreo rutinario.

---

## Modelo incluido

El artefacto `modelo_reincidencia.pkl` contiene una lista de 4 elementos:

```python
[modelo, labelencoder, variables, scaler]
```

| Elemento       | Descripcion                                             |
|----------------|---------------------------------------------------------|
| `modelo`       | VotingClassifier (soft): LR + NB calibrado + RF         |
| `labelencoder` | LabelEncoder con clases `['NO', 'SI']`                  |
| `variables`    | Array con los nombres de columnas despues de dummies    |
| `scaler`       | MinMaxScaler ajustado sobre variables numericas, o None |

---

## Notas tecnicas

- El modelo se carga una sola vez gracias a `@st.cache_resource`.
- Los valores faltantes en el CSV se rellenan con 0.
- Las columnas extra en el CSV se ignoran; las faltantes se completan con 0.
- El modulo `pipeline_reincidencia.py` puede usarse de forma independiente:

```python
from pipeline_reincidencia import predecir
import pandas as pd

df = pd.read_csv("nuevos_datos.csv")
resultados = predecir(df)
print(resultados[["PREDICCION", "PROBABILIDAD_REINCIDENCIA"]])
```
