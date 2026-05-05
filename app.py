import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# Configuración de la app
st.set_page_config(page_title="Riesgo Actuarial", layout="centered")

st.title("Clasificación de riesgo actuarial (K-means)")
st.write("Ingrese los datos del cliente para estimar su nivel de riesgo.")

# Rutas
MODEL_PATH = Path("models/kmeans_riesgo_actuarial.pkl")
META_PATH = Path("models/model_metadata.json")

# Cargar modelo y metadata
modelo = joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

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

    # Crear DataFrame (IMPORTANTE: nombres exactos)
    cliente = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "charges": charges
    }])

    # Predicción (maneja ambos casos)
    try:
        # Caso 1: modelo tipo pipeline
        cluster = int(modelo.predict(cliente)[0])

    except:
        # Caso 2: modelo guardado como dict (model + scaler)
        columnas = ["age", "bmi", "children", "charges"]
        X = cliente[columnas]

        scaler = modelo["scaler"]
        kmeans = modelo["model"]

        X_scaled = scaler.transform(X)
        cluster = int(kmeans.predict(X_scaled)[0])

    # Obtener riesgo
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

    st.write(f"Cluster asignado: {cluster}")

    # Mostrar datos
    st.write("Datos evaluados:")
    st.dataframe(cliente)
