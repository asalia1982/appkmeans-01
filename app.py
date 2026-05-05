import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# API Gemini
import google.generativeai as genai

# Configuración
st.set_page_config(page_title="Riesgo Actuarial", layout="centered")

st.title("Clasificación de riesgo actuarial + IA")
st.write("Modelo K-means + recomendaciones inteligentes")

# Rutas
MODEL_PATH = Path("models/kmeans_riesgo_actuarial.pkl")
META_PATH = Path("models/model_metadata.json")

# Cargar modelo
modelo = joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Configurar API
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    modelo_ia = genai.GenerativeModel("gemini-pro")
    usar_ia = True
except:
    usar_ia = False

# Formulario
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad", 18, 100, 40)
    bmi = st.number_input("BMI", 10.0, 60.0, 28.0)
    children = st.number_input("Hijos", 0, 10, 1)

with col2:
    sex = st.selectbox("Sexo", ["male", "female"])
    smoker = st.selectbox("¿Fuma?", ["no", "yes"])
    region = st.selectbox("Región", ["southwest", "southeast", "northwest", "northeast"])
    charges = st.number_input("Cargos médicos", 0.0, 50000.0, 10000.0)

# Botón
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

    # Predicción
    try:
        cluster = int(modelo.predict(cliente)[0])
    except:
        columnas = ["age", "bmi", "children", "charges"]
        X = cliente[columnas]

        scaler = modelo["scaler"]
        kmeans = modelo["model"]

        X_scaled = scaler.transform(X)
        cluster = int(kmeans.predict(X_scaled)[0])

    mapa_riesgo = metadata["mapa_riesgo"]
    riesgo = mapa_riesgo[str(cluster)]

    # Mostrar resultado
    st.subheader("Resultado")

    if riesgo == "Alto":
        st.error(f"Riesgo: {riesgo}")
    elif riesgo == "Medio":
        st.warning(f"Riesgo: {riesgo}")
    else:
        st.success(f"Riesgo: {riesgo}")

    st.write(f"Cluster: {cluster}")

    # IA ACTUARIAL
    st.subheader("Recomendación actuarial")

    if usar_ia:
        prompt = f"""
        Eres un actuario profesional.

        Cliente:
        Edad: {age}
        BMI: {bmi}
        Hijos: {children}
        Fumador: {smoker}
        Región: {region}
        Cargos médicos: {charges}

        Clasificación de riesgo: {riesgo}

        Genera recomendaciones actuariales claras:
        - Evaluación del riesgo
        - Posibles decisiones de prima
        - Acciones preventivas
        """

        try:
            respuesta = modelo_ia.generate_content(prompt)
            st.write(respuesta.text)
        except:
            st.warning("Error generando recomendación IA")

    else:
        st.info("IA no configurada (falta API KEY)")

    st.write("Datos evaluados:")
    st.dataframe(cliente)
