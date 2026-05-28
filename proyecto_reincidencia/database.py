from datetime import datetime, timezone
import math
import pandas as pd
from supabase_client import get_client

TABLE = "evaluaciones_reincidencia"


def _serializable(value):
    """Convierte tipos numpy/pandas a tipos nativos de Python para JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _limpiar_dict(d: dict) -> dict:
    return {k: _serializable(v) for k, v in d.items()}


def guardar_individual(datos: dict, prediccion: str, probabilidad: float) -> None:
    get_client().table(TABLE).insert({
        "fuente": "INDIVIDUAL",
        "prediccion": prediccion,
        "probabilidad_reincidencia": round(float(probabilidad), 4),
        "datos_clinicos": _limpiar_dict(datos),
        "fecha_registro": datetime.now(timezone.utc).isoformat(),
    }).execute()


def guardar_lote(df: pd.DataFrame) -> None:
    cols_meta = {"PREDICCION", "PROBABILIDAD_REINCIDENCIA"}
    records = []
    for _, row in df.iterrows():
        datos = {col: _serializable(row[col]) for col in df.columns if col not in cols_meta}
        records.append({
            "fuente": "LOTE",
            "prediccion": str(row["PREDICCION"]),
            "probabilidad_reincidencia": round(float(row["PROBABILIDAD_REINCIDENCIA"]), 4),
            "datos_clinicos": datos,
            "fecha_registro": datetime.now(timezone.utc).isoformat(),
        })
    get_client().table(TABLE).insert(records).execute()
