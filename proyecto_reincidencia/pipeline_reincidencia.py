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


def preprocesar(df_raw, variables):
    df = df_raw.copy()

    cols_bool = [c for c in COLS_BOOLEANAS if c in df.columns]
    if cols_bool:
        df = pd.get_dummies(df, columns=cols_bool, drop_first=True, dtype=int)

    cols_mc = [c for c in COLS_MULTICAT if c in df.columns]
    if cols_mc:
        df = pd.get_dummies(df, columns=cols_mc, drop_first=False, dtype=int)

    cols_bc = [c for c in COLS_BINARIAS_CAT if c in df.columns]
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
