import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import google.generativeai as genai


st.set_page_config(
    page_title="Riesgo Actuarial con K-means",
    layout="centered"
)

st.title("Clasificación de riesgo actuarial con K-means")
st.write("Ingrese los datos del cliente para estimar su nivel de riesgo y generar recomendaciones actuariales.")

MODEL_PATH = Path("models/kmeans_riesgo_actuarial.pkl")
META_PATH = Path("models/model_metadata.json")


@st.cache_resource
def cargar_modelo():
    modelo_cargado = joblib.load(MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata_cargada = json.load(f)

    return modelo_cargado, metadata_cargada


modelo, metadata = cargar_modelo()


def predecir_cluster(cliente_df):
    try:
        return int(modelo.predict(cliente_df)[0])
    except Exception:
        columnas = ["age", "bmi", "children", "charges"]
        X = cliente_df[columnas]

        scaler = modelo["scaler"]
        kmeans = modelo["model"]

        X_scaled = scaler.transform(X)
        return int(kmeans.predict(X_scaled)[0])


def generar_recomendacion_ia(cliente_df, riesgo, cluster):
    if "GEMINI_API_KEY" not in st.secrets:
        return "La API de Gemini no está configurada. Agregue GEMINI_API_KEY en Streamlit Secrets."

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        modelo_ia = genai.GenerativeModel("gemini-2.5-flash")

        cliente = cliente_df.iloc[0].to_dict()

        prompt = f"""
Eres un consultor actuarial. Analiza el siguiente perfil de cliente de seguro médico.

Datos del cliente:
- Edad: {cliente["age"]}
- Sexo: {cliente["sex"]}
- BMI: {cliente["bmi"]}
- Hijos/dependientes: {cliente["children"]}
- Fumador: {cliente["smoker"]}
- Región: {cliente["region"]}
- Cargos médicos estimados: {cliente["charges"]}
- Cluster asignado por K-means: {cluster}
- Nivel de riesgo actuarial: {riesgo}

Genera una recomendación breve, clara y profesional en español con:
1. Interpretación del riesgo.
2. Factores principales que podrían influir en el riesgo.
3. Recomendaciones preventivas.
4. Recomendación gerencial para una aseguradora.

Aclara que esto es una orientación analítica y no una decisión actuarial definitiva.
"""

        respuesta = modelo_ia.generate_content(prompt)
        return respuesta.text

    except Exception as e:
        return f"No se pudo generar la recomendación con IA. Detalle técnico: {e}"


st.subheader("Datos del cliente")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad", min_value=18, max_value=100, value=40)
    bmi = st.number_input("Índice de masa corporal (BMI)", min_value=10.0, max_value=60.0, value=28.0)
    children = st.number_input("Número de hijos/dependientes", min_value=0, max_value=10, value=1)

with col2:
    sex = st.selectbox("Sexo", ["male", "female"])
    smoker = st.selectbox("¿Fuma?", ["no", "yes"])
    region = st.selectbox("Región", ["southwest", "southeast", "northwest", "northeast"])
    charges = st.number_input("Cargos médicos estimados", min_value=0.0, max_value=100000.0, value=10000.0)


if st.button("Evaluar riesgo"):
    cliente = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "charges": charges
    }])

    cluster = predecir_cluster(cliente)

    mapa_riesgo = metadata.get("mapa_riesgo", {})
    riesgo = mapa_riesgo.get(str(cluster), "No definido")

    st.subheader("Resultado")

    if riesgo == "Alto":
        st.error(f"Riesgo actuarial: {riesgo}")
    elif riesgo == "Medio":
        st.warning(f"Riesgo actuarial: {riesgo}")
    elif riesgo == "Bajo":
        st.success(f"Riesgo actuarial: {riesgo}")
    else:
        st.info(f"Riesgo actuarial: {riesgo}")

    st.write(f"Cluster asignado: {cluster}")

    st.subheader("Recomendación actuarial con IA")
    recomendacion = generar_recomendacion_ia(cliente, riesgo, cluster)
    st.write(recomendacion)

    st.subheader("Datos evaluados")
    st.dataframe(cliente)

st.divider()

st.caption(
    "Nota: K-means es un modelo no supervisado. La clasificación es orientativa y debe validarse con criterio actuarial profesional."
)
