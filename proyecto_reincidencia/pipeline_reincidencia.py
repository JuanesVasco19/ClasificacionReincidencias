import pickle
import pandas as pd

MODEL_PATH = 'modelo_reincidencia.pkl'

COLS_BOOLEANAS = [
    'GESTANTE', 'POBLACION_ A _CARGO_ ICBF', 'CONFLICTO_ CON_ PAREJA_ O _EX_ PAREJA',
    'ENFERMEDAD_ CRONICA_ DOLOROSA_ O_ DISCAPACITANTE', 'PROBLEMAS_ ECONOMICOS',
    'MUERTE_ DE_ UN_ FAMILIAR', 'PROBLEMAS_ JURIDICOS', 'SUICIDIO_ DE_ UN_ FAMILIAR',
    'MALTRATO_ FISICO_SICOLOGICO_SEXUAL', 'PROBLEMAS_ LABORALES', 'PROBLEMAS_ FAMILIARES',
    'CONSUMO_ DE_ SPA', 'ANTECEDENTES_ FAMILIARES_ DE_ CONDUCTA_ SUICIDA',
    'IDEACION_ SUICIDA_ PERSISTENTE_', 'PLAN_ ORGANIZADO_ DE _SUICIDIO',
    'TRASTORNO_ DEPRESIVO', 'TRASTORNO_ DE_ PERSONALIDAD', 'TRASTORNO_ BIPOLAR',
    'ESQUIZOFRENIA', 'ANTECEDENTE_ VIOLENCIA _O_ ABUSO', 'ABUSO_ DE_  ALCOHOL',
    'AHORCAMIENTO', 'ARMA_ CORTOPUNZANTE', 'LANZAMIENTO_ AL_ VACIO',
    'LANZAMIENTO_ A_ VEHICULO', 'INTOXICACIONES',
]

# >2 categorías → drop_first=False; exactamente 2 → drop_first=True
COLS_MULTICAT     = ['SEGURIDAD_ SOCIAL', 'ESTRATO_SOCIOECONOMICO', 'ESTADO_CIVIL', 'ESCOLARIDAD']
COLS_BINARIAS_CAT = ['SEXO', 'AREA_ DE_ RESIDENCIA']


def cargar_modelo(path=MODEL_PATH):
    with open(path, 'rb') as f:
        modelo, labelencoder, variables = pickle.load(f)
    return modelo, labelencoder, variables


CATS_BINARIAS = {
    'SEXO':                ['FEMENINO', 'MASCULINO'],
    'AREA_ DE_ RESIDENCIA': ['CABECERA MUNICIPAL', 'CENTRO POBLADO', 'RURAL DISPERSO'],
}


def preprocesar(df_raw, variables):
    df = df_raw.copy()

    # Booleanas: codificación directa COLUMN_SI=1/0 para evitar el bug de
    # get_dummies(drop_first=True) que elimina la única categoría en filas únicas
    cols_bool = [c for c in COLS_BOOLEANAS if c in df.columns]
    for col in cols_bool:
        df[col + '_SI'] = (
            df[col].astype(str).str.strip().str.upper() == 'SI'
        ).astype(int)
    df = df.drop(columns=cols_bool, errors='ignore')

    # Multicategoría: drop_first=False, sin problema de categoría única
    cols_mc = [c for c in COLS_MULTICAT if c in df.columns]
    if cols_mc:
        df = pd.get_dummies(df, columns=cols_mc, drop_first=False, dtype=int)

    # Binarias categóricas: categorías fijas para que drop_first sea estable
    cols_bc = [c for c in COLS_BINARIAS_CAT if c in df.columns]
    for col in cols_bc:
        df[col] = pd.Categorical(
            df[col].astype(str).str.strip().str.upper(),
            categories=CATS_BINARIAS[col],
        )
    if cols_bc:
        df = pd.get_dummies(df, columns=cols_bc, drop_first=True, dtype=int)

    return df.fillna(0).reindex(columns=variables, fill_value=0)


def predecir(df_raw, model_path=MODEL_PATH):
    modelo, labelencoder, variables = cargar_modelo(model_path)
    X = preprocesar(df_raw, variables)
    si_idx = list(labelencoder.classes_).index('SI')
    resultado = df_raw.copy()
    resultado['PREDICCION'] = labelencoder.inverse_transform(modelo.predict(X))
    resultado['PROBABILIDAD_REINCIDENCIA'] = modelo.predict_proba(X)[:, si_idx].round(4)
    return resultado
