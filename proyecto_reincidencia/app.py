import base64
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

from pipeline_reincidencia import cargar_modelo, preprocesar
from database import guardar_individual, guardar_lote

BASE_DIR = Path(__file__).parent
MODEL_PATH = str(BASE_DIR / 'modelo_reincidencia.pkl')

# ── CSS INSTITUCIONAL ──────────────────────────────────────────────────────────
CSS = """
<style>
/* Fondo general */
.stApp { background-color: #eef1f5; }
.main .block-container { padding: 1.5rem 2.5rem; max-width: 1150px; }

/* Ocultar elementos de Streamlit */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { visibility: visible !important; }

/* Barra lateral */
[data-testid="stSidebar"] { background-color: #1a3a5c; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] .stMarkdown li { color: #c0d4e8 !important; }

/* Pestañas */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 2px solid #1a3a5c;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    background-color: #dde4ec;
    color: #1a3a5c;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 10px 28px;
    border-radius: 0;
    border: 1px solid #bbc8d6;
    border-bottom: none;
}
.stTabs [aria-selected="true"] {
    background-color: #1a3a5c !important;
    color: white !important;
    border-color: #1a3a5c !important;
}

/* Etiquetas de campos */
.stRadio > label,
.stSelectbox > label,
.stFileUploader > label {
    font-size: 10px !important;
    font-weight: 700 !important;
    color: #2a3a4a !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* Opciones de radio */
.stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 13px !important;
    color: #2a3a4a !important;
}

/* Botón de envío del formulario */
.stFormSubmitButton > button {
    background-color: #1a3a5c !important;
    color: white !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 2.5px !important;
    padding: 14px !important;
    border-radius: 0 !important;
    border: none !important;
    text-transform: uppercase;
    width: 100%;
}
.stFormSubmitButton > button:hover {
    background-color: #0d2640 !important;
}

/* Métricas */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #ccd4de;
    padding: 14px 18px;
    border-radius: 0;
}
[data-testid="metric-container"] label {
    font-size: 10px !important;
    color: #5a6a7a !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* Separadores */
hr { border-color: #ccd4de !important; }

/* Contenedor del resultado de evaluación */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
    background-color: transparent !important;
    border: 1px solid #d0d7de !important;
    box-shadow: none !important;
}
</style>
"""

# ── CONFIGURACIÓN DE PÁGINA ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Evaluación de Riesgo — Conducta Suicida Reincidente",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ── BARRA LATERAL ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(str(BASE_DIR / "assets" / "OIP.jpg"), use_container_width=True)
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <p style="font-size:14px; letter-spacing:2px; color:#6a90b0; margin:0;
                  text-transform:uppercase;">Sistema de Apoyo a la Decisión Clínica</p>
        <p style="font-size:20px; font-weight:700; color:white; line-height:1.4; margin:8px 0 0 0;">
            Conducta Suicida<br>Reincidente
        </p>
    </div>
    <hr style="border:none; border-top:1px solid #2a5a8c; margin:0 0 20px 0;">
    <p style="font-size:14px; letter-spacing:2px; color:#6a90b0;
              text-transform:uppercase; margin:0 0 6px 0;">Institución</p>
    <p style="font-size:16px; color:#b0cce0; line-height:1.7; margin:0 0 24px 0;">
        Secretaría de Salud<br>Municipio de Tunja, Boyacá
    </p>
    <hr style="border:none; border-top:1px solid #2a5a8c; margin:0 0 20px 0;">
    <p style="font-size:14px; letter-spacing:2px; color:#6a90b0;
              text-transform:uppercase; margin:0 0 8px 0;">Advertencia</p>
    <p style="font-size:15px; color:#8aaec8; line-height:1.7; margin:0;">
        Esta herramienta es un instrumento de apoyo a la decisión clínica.
        Los resultados no reemplazan el criterio del profesional de salud
        ni constituyen un diagnóstico definitivo.
    </p>
    """, unsafe_allow_html=True)

# ── CARGA DEL MODELO ───────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return cargar_modelo(MODEL_PATH)

try:
    modelo, labelencoder, variables = get_model()
    SI_IDX = list(labelencoder.classes_).index('SI')
    _ok = True
except FileNotFoundError:
    st.error(
        f"Archivo de modelo no encontrado: '{MODEL_PATH}'. "
        "Colóquelo en la misma carpeta que app.py y reinicie la aplicación."
    )
    _ok = False
except Exception as exc:
    st.error(f"Error al cargar el modelo: {exc}")
    _ok = False

# ── ENCABEZADO PRINCIPAL ───────────────────────────────────────────────────────
with open(BASE_DIR / "assets" / "upb-550x212.png", "rb") as _f:
    _logo_b64 = base64.b64encode(_f.read()).decode()

st.markdown(
    f'<div style="position:relative;border-bottom:2px solid #1a3a5c;padding-bottom:14px;margin-bottom:24px;">'
    f'<img src="data:image/png;base64,{_logo_b64}" '
    f'style="position:absolute;top:-4px;right:-8px;height:80px;width:auto;opacity:0.55;">'
    f'<p style="font-size:12px;letter-spacing:2.5px;color:#6a7a8a;text-transform:uppercase;margin:0;">'
    f'Herramienta de apoyo a la decisión clínica</p>'
    f'<h1 style="font-size:28px;color:#1a3a5c;margin:6px 0 0 0;font-weight:700;'
    f'font-family:\'Segoe UI\',Arial,sans-serif;">'
    f'Evaluación de Riesgo de Reincidencia en Conducta Suicida</h1>'
    f'</div>',
    unsafe_allow_html=True,
)

if not _ok:
    st.stop()

# ── FUNCIONES AUXILIARES ───────────────────────────────────────────────────────
def _encabezado_seccion(numero, titulo):
    st.markdown(
        f'<div style="background:#1a3a5c; color:white; padding:7px 16px; '
        f'font-size:13px; font-weight:700; letter-spacing:2px; text-transform:uppercase; '
        f'margin:28px 0 16px 0;">'
        f'{numero}. {titulo}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_resultado(prob):
    if prob >= 0.70:
        nivel = "ALTO"
        recomendacion = (
            "Se recomienda activar el protocolo de atención prioritaria. "
            "El perfil del paciente indica un riesgo elevado de reincidencia en conducta suicida. "
            "Iniciar seguimiento intensivo e intervención inmediata en coordinación con el equipo "
            "de salud mental."
        )
    elif prob >= 0.50:
        nivel = "MODERADO"
        recomendacion = (
            "Se recomienda seguimiento activo a corto plazo. "
            "El perfil del paciente presenta factores de riesgo que requieren atención. "
            "Programar consulta de control en los próximos días e implementar estrategias "
            "de apoyo psicosocial."
        )
    else:
        nivel = "BAJO"
        recomendacion = (
            "Mantener monitoreo rutinario según el protocolo establecido. "
            "El perfil actual no indica un riesgo inmediato elevado de reincidencia. "
            "Continuar con el plan de atención vigente y realizar seguimiento periódico "
            "conforme a los lineamientos institucionales."
        )

    if prob >= 0.70:
        card_bg, card_border = "#fde8e8", "#e53935"
    elif prob >= 0.50:
        card_bg, card_border = "#fff4e0", "#fb8c00"
    else:
        card_bg, card_border = "#e8f5e9", "#43a047"

    barra = int(prob * 100)

    st.markdown(
        f'<div style="background:{card_bg};border:1px solid {card_border};'
        f'border-left:6px solid {card_border};border-radius:6px;padding:24px 28px;margin-top:20px;">'
        f'<p style="font-size:11px;letter-spacing:2px;color:#5a6a7a;text-transform:uppercase;'
        f'margin:0 0 16px 0;padding-bottom:10px;border-bottom:1px solid {card_border};">'
        f'Resultado de la Evaluación</p>'
        f'<div style="display:flex;gap:60px;margin-bottom:20px;">'
        f'<div><p style="font-size:11px;color:#5a6a7a;text-transform:uppercase;'
        f'letter-spacing:1px;margin:0 0 4px 0;">Nivel de riesgo estimado</p>'
        f'<p style="font-size:28px;font-weight:700;color:{card_border};margin:0;">{nivel}</p></div>'
        f'<div><p style="font-size:11px;color:#5a6a7a;text-transform:uppercase;'
        f'letter-spacing:1px;margin:0 0 4px 0;">Probabilidad de reincidencia</p>'
        f'<p style="font-size:28px;font-weight:700;color:{card_border};margin:0;">{prob:.1%}</p></div>'
        f'</div>'
        f'<div style="background:#00000020;height:8px;border-radius:4px;margin-bottom:20px;">'
        f'<div style="background:{card_border};width:{barra}%;height:100%;border-radius:4px;"></div></div>'
        f'<p style="font-size:11px;color:#5a6a7a;text-transform:uppercase;'
        f'letter-spacing:1px;margin:0 0 6px 0;">Recomendación clínica</p>'
        f'<p style="font-size:14px;color:#2a3a4a;line-height:1.8;margin:0;">{recomendacion}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── PESTAÑAS ───────────────────────────────────────────────────────────────────
tab_ind, tab_lote = st.tabs(["EVALUACIÓN INDIVIDUAL", "EVALUACIÓN POR LOTE"])


# ══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 1 — EVALUACIÓN INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_ind:
    st.markdown("""
    <p style="font-size:15px; color:#5a6a7a; margin:8px 0 4px 0;">
        Complete todos los campos de la ficha clínica y presione el botón
        <strong>Generar evaluación de riesgo</strong> para obtener el resultado.
    </p>
    """, unsafe_allow_html=True)

    with st.form("ficha_clinica"):

        # ── I. DATOS SOCIODEMOGRÁFICOS ────────────────────────────────────────
        _encabezado_seccion("I", "Datos Sociodemográficos")

        c1, c2, c3 = st.columns(3)
        sexo = c1.selectbox("Sexo", ["FEMENINO", "MASCULINO"])
        area_residencia = c2.selectbox(
            "Área de residencia",
            ["CABECERA MUNICIPAL", "RURAL DISPERSO", "CENTRO POBLADO"],
        )
        seguridad_social = c3.selectbox(
            "Régimen de seguridad social",
            ["CONTRIBUTIVO", "SUBSIDIADO", "EXCEPCION",
             "ESPECIAL", "NO ASEGURADO", "INDETERMINADO"],
        )

        c4, c5, c6 = st.columns(3)
        estrato = c4.selectbox(
            "Estrato socioeconómico", ["1", "2", "3", "4", "5", "6"]
        )
        estado_civil = c5.selectbox(
            "Estado civil",
            ["SOLTERO/A", "CASADO/A", "UNION LIBRE",
             "DIVORCIADO/A", "VIUDO(A)", "SEPARADO/A"],
        )
        escolaridad = c6.selectbox(
            "Nivel de escolaridad",
            ["NINGUNO", "PREESCOLAR", "BASICA PRIMARIA", "BASICA SECUNDARIA",
             "MEDIA TECNICA", "TECNOLOGIA O TECNICA", "PROFESIONAL",
             "ESPECIALIZACION", "MAESTRIA", "SIN INFORMACION"],
        )

        c7, c8 = st.columns(2)
        gestante = c7.radio(
            "Estado gestacional", ["NO", "SI"], horizontal=True, key="r_gestante"
        )
        pob_icbf = c8.radio(
            "Población a cargo del ICBF", ["NO", "SI"], horizontal=True, key="r_icbf"
        )

        # ── II. FACTORES DE RIESGO CLÍNICO ────────────────────────────────────
        _encabezado_seccion("II", "Factores de Riesgo Clínico")

        c9, c10 = st.columns(2)
        tras_depresivo = c9.radio(
            "Trastorno depresivo", ["NO", "SI"], horizontal=True, key="r_tdep"
        )
        tras_personalidad = c10.radio(
            "Trastorno de personalidad", ["NO", "SI"], horizontal=True, key="r_tper"
        )

        c11, c12 = st.columns(2)
        tras_bipolar = c11.radio(
            "Trastorno bipolar", ["NO", "SI"], horizontal=True, key="r_tbip"
        )
        esquizofrenia = c12.radio(
            "Esquizofrenia", ["NO", "SI"], horizontal=True, key="r_esq"
        )

        c13, c14 = st.columns(2)
        ant_fam = c13.radio(
            "Antecedentes familiares de conducta suicida",
            ["NO", "SI"], horizontal=True, key="r_antfam"
        )
        ant_violencia = c14.radio(
            "Antecedente de violencia o abuso",
            ["NO", "SI"], horizontal=True, key="r_antv"
        )

        # ── III. FACTORES DE RIESGO PSICOSOCIAL ──────────────────────────────
        _encabezado_seccion("III", "Factores de Riesgo Psicosocial")

        c15, c16 = st.columns(2)
        conflicto_pareja = c15.radio(
            "Conflicto con pareja o ex-pareja",
            ["NO", "SI"], horizontal=True, key="r_conf"
        )
        enf_cronica = c16.radio(
            "Enfermedad crónica, dolorosa o discapacitante",
            ["NO", "SI"], horizontal=True, key="r_enf"
        )

        c17, c18 = st.columns(2)
        prob_economicos = c17.radio(
            "Problemas económicos", ["NO", "SI"], horizontal=True, key="r_eco"
        )
        muerte_familiar = c18.radio(
            "Muerte de un familiar", ["NO", "SI"], horizontal=True, key="r_mf"
        )

        c19, c20 = st.columns(2)
        prob_juridicos = c19.radio(
            "Problemas jurídicos", ["NO", "SI"], horizontal=True, key="r_jur"
        )
        suicidio_familiar = c20.radio(
            "Suicidio de un familiar", ["NO", "SI"], horizontal=True, key="r_sf"
        )

        c21, c22 = st.columns(2)
        maltrato = c21.radio(
            "Maltrato físico, psicológico o sexual",
            ["NO", "SI"], horizontal=True, key="r_malt"
        )
        prob_laborales = c22.radio(
            "Problemas laborales", ["NO", "SI"], horizontal=True, key="r_lab"
        )

        c23, c24 = st.columns(2)
        prob_familiares = c23.radio(
            "Problemas familiares", ["NO", "SI"], horizontal=True, key="r_pfam"
        )
        consumo_spa = c24.radio(
            "Consumo de sustancias psicoactivas",
            ["NO", "SI"], horizontal=True, key="r_spa"
        )

        c25, _ = st.columns(2)
        abuso_alcohol = c25.radio(
            "Abuso de alcohol", ["NO", "SI"], horizontal=True, key="r_alc"
        )

        # ── IV. CARACTERÍSTICAS DEL EVENTO ───────────────────────────────────
        _encabezado_seccion("IV", "Características del Evento")

        c26, c27 = st.columns(2)
        ideacion = c26.radio(
            "Ideación suicida persistente",
            ["NO", "SI"], horizontal=True, key="r_idea"
        )
        plan_suicidio = c27.radio(
            "Plan organizado de suicidio",
            ["NO", "SI"], horizontal=True, key="r_plan"
        )

        metodo = st.selectbox(
            "Método utilizado en el intento",
            ["NINGUNO", "AHORCAMIENTO", "ARMA CORTOPUNZANTE",
             "LANZAMIENTO AL VACÍO", "LANZAMIENTO A VEHÍCULO", "INTOXICACIONES"],
        )

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "GENERAR EVALUACIÓN DE RIESGO",
            use_container_width=True,
        )

    # ── RESULTADO INDIVIDUAL ──────────────────────────────────────────────────
    if submitted:
        datos = {
            'SEXO':                                             sexo,
            'AREA_ DE_ RESIDENCIA':                            area_residencia,
            'SEGURIDAD_ SOCIAL':                               seguridad_social,
            'ESTRATO_SOCIOECONOMICO':                          estrato,
            'GESTANTE':                                        gestante,
            'POBLACION_ A _CARGO_ ICBF':                      pob_icbf,
            'ESTADO_CIVIL':                                    estado_civil,
            'ESCOLARIDAD':                                     escolaridad,
            'CONFLICTO_ CON_ PAREJA_ O _EX_ PAREJA':          conflicto_pareja,
            'ENFERMEDAD_ CRONICA_ DOLOROSA_ O_ DISCAPACITANTE': enf_cronica,
            'PROBLEMAS_ ECONOMICOS':                           prob_economicos,
            'MUERTE_ DE_ UN_ FAMILIAR':                        muerte_familiar,
            'PROBLEMAS_ JURIDICOS':                            prob_juridicos,
            'SUICIDIO_ DE_ UN_ FAMILIAR':                      suicidio_familiar,
            'MALTRATO_ FISICO_SICOLOGICO_SEXUAL':              maltrato,
            'PROBLEMAS_ LABORALES':                            prob_laborales,
            'PROBLEMAS_ FAMILIARES':                           prob_familiares,
            'CONSUMO_ DE_ SPA':                                consumo_spa,
            'ANTECEDENTES_ FAMILIARES_ DE_ CONDUCTA_ SUICIDA': ant_fam,
            'IDEACION_ SUICIDA_ PERSISTENTE_':                 ideacion,
            'PLAN_ ORGANIZADO_ DE _SUICIDIO':                  plan_suicidio,
            'TRASTORNO_ DEPRESIVO':                            tras_depresivo,
            'TRASTORNO_ DE_ PERSONALIDAD':                     tras_personalidad,
            'TRASTORNO_ BIPOLAR':                              tras_bipolar,
            'ESQUIZOFRENIA':                                    esquizofrenia,
            'ANTECEDENTE_ VIOLENCIA _O_ ABUSO':                ant_violencia,
            'ABUSO_ DE_  ALCOHOL':                             abuso_alcohol,
            'AHORCAMIENTO':                                    'SI' if metodo == 'AHORCAMIENTO' else 'NO',
            'ARMA_ CORTOPUNZANTE':                             'SI' if metodo == 'ARMA CORTOPUNZANTE' else 'NO',
            'LANZAMIENTO_ AL_ VACIO':                          'SI' if metodo == 'LANZAMIENTO AL VACÍO' else 'NO',
            'LANZAMIENTO_ A_ VEHICULO':                        'SI' if metodo == 'LANZAMIENTO A VEHÍCULO' else 'NO',
            'INTOXICACIONES':                                  'SI' if metodo == 'INTOXICACIONES' else 'NO',
        }
        try:
            X = preprocesar(pd.DataFrame([datos]), variables)
            prob = float(modelo.predict_proba(X)[0, SI_IDX])
            prediccion = labelencoder.inverse_transform(modelo.predict(X))[0]
            st.session_state['prob_individual'] = prob
            try:
                guardar_individual(datos, prediccion, prob)
                st.toast("Registro guardado en Supabase.", icon="✅")
            except Exception as db_exc:
                st.warning(f"Evaluación generada, pero no se pudo guardar en Supabase: {db_exc}")
        except Exception as exc:
            st.session_state['prob_individual'] = None
            st.error(f"Error al generar la evaluación: {exc}")

    if st.session_state.get('prob_individual') is not None:
        _render_resultado(st.session_state['prob_individual'])


# ══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 2 — EVALUACIÓN POR LOTE
# ══════════════════════════════════════════════════════════════════════════════
with tab_lote:
    st.markdown("""
    <div style="border-bottom:1px solid #ccd4de; padding-bottom:12px; margin-bottom:20px;">
        <h2 style="font-size:17px; color:#1a3a5c; margin:0; font-weight:700;">
            Evaluación por Lote de Pacientes
        </h2>
        <p style="font-size:15px; color:#5a6a7a; margin:6px 0 0 0;">
            Cargue un archivo CSV con los registros de los pacientes. Las columnas deben
            coincidir con el formato del dataset de entrenamiento.
        </p>
    </div>
    """, unsafe_allow_html=True)

    archivo = st.file_uploader("Seleccionar archivo de datos (.csv)", type=["csv"])

    if archivo is not None:
        try:
            df_entrada = pd.read_csv(archivo, encoding="utf-8")
        except UnicodeDecodeError:
            archivo.seek(0)
            df_entrada = pd.read_csv(archivo, encoding="latin-1")

        st.markdown(
            f'<p style="font-size:11px; color:#5a6a7a; margin-bottom:12px;">'
            f'Registros cargados: <strong>{len(df_entrada)}</strong> &nbsp;|&nbsp; '
            f'Columnas detectadas: <strong>{len(df_entrada.columns)}</strong></p>',
            unsafe_allow_html=True,
        )

        with st.spinner("Procesando registros..."):
            try:
                X = preprocesar(df_entrada, variables)
                preds_enc = modelo.predict(X)
                probs = modelo.predict_proba(X)[:, SI_IDX]
                df_resultado = df_entrada.copy()
                df_resultado['PREDICCION'] = labelencoder.inverse_transform(preds_enc)
                df_resultado['PROBABILIDAD_REINCIDENCIA'] = probs.round(4)
                try:
                    guardar_lote(df_resultado)
                    st.toast(f"{len(df_resultado)} registros guardados en Supabase.", icon="✅")
                except Exception as db_exc:
                    st.warning(f"Lote procesado, pero no se pudo guardar en Supabase: {db_exc}")
            except Exception as exc:
                st.error(f"Error al procesar el archivo: {exc}")
                st.info(
                    "Verifique que el archivo CSV contenga las columnas del dataset original."
                )
                st.stop()

        total = len(df_resultado)
        n_si = int((df_resultado['PREDICCION'] == 'SI').sum())
        pct_si = n_si / total * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Total de registros evaluados", total)
        c2.metric("Casos predichos como reincidentes", n_si)
        c3.metric("Porcentaje de reincidencia", f"{pct_si:.1f} %")

        # Tabla de resultados
        st.markdown("""
        <p style="font-size:9px; letter-spacing:2px; color:#5a6a7a; text-transform:uppercase;
                  margin:28px 0 8px 0; padding-top:16px; border-top:1px solid #ccd4de;">
            Tabla de resultados
        </p>
        """, unsafe_allow_html=True)

        cols_inicio = ['PREDICCION', 'PROBABILIDAD_REINCIDENCIA']
        cols_resto = [c for c in df_resultado.columns if c not in cols_inicio]
        st.dataframe(
            df_resultado[cols_inicio + cols_resto],
            use_container_width=True,
        )

        # Gráfico (paleta institucional, sin colores llamativos)
        conteo = df_resultado['PREDICCION'].value_counts()
        df_conteo = pd.DataFrame({
            'Prediccion': conteo.index,
            'Cantidad': conteo.values,
        })
        fig = px.bar(
            df_conteo,
            x='Prediccion', y='Cantidad',
            color='Prediccion',
            color_discrete_map={'SI': '#1a3a5c', 'NO': '#8ea8c0'},
            text='Cantidad',
            title='Distribución de predicciones de reincidencia',
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Segoe UI, Arial', size=12, color='#2a3a4a'),
            title_font=dict(size=13, color='#1a3a5c'),
            xaxis=dict(title='Predicción', gridcolor='#e4e8ec', linecolor='#ccd4de'),
            yaxis=dict(title='Número de registros', gridcolor='#e4e8ec', linecolor='#ccd4de'),
            margin=dict(t=50, b=40, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Descarga
        st.download_button(
            label="Descargar reporte completo (.csv)",
            data=df_resultado.to_csv(index=False).encode('utf-8'),
            file_name="reporte_evaluacion_riesgo.csv",
            mime="text/csv",
        )
