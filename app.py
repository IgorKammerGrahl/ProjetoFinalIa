import streamlit as st
import numpy as np
import joblib

# Carrega artefatos pré‑treinados
@st.cache_resource
def load_models():
    scaler_full  = joblib.load('scaler_full.pkl')
    model_full   = joblib.load('model_full.pkl')
    scaler_ch    = joblib.load('scaler_chuvas.pkl')
    model_ch     = joblib.load('model_chuvas.pkl')
    return scaler_full, model_full, scaler_ch, model_ch

scaler_full, model_full, scaler_ch, model_ch = load_models()

st.title("💧 Previsão do Nível do Rio de Sul")

modo = st.selectbox("Modelo:", ["Completo (4 inputs)", "Só Chuvas (2 inputs)"])
if modo == "Completo (4 inputs)":
    ni = st.number_input("Nível em Ituporanga (m)", value=0.0)
    ci = st.number_input("Chuva em Ituporanga (mm)", value=0.0)
    nt = st.number_input("Nível em Taió (m)", value=0.0)
    ct = st.number_input("Chuva em Taió (mm)", value=0.0)
else:
    ci = st.number_input("Chuva em Ituporanga (mm)", value=0.0)
    ct = st.number_input("Chuva em Taió (mm)", value=0.0)

if st.button("Prever"):
    if modo == "Completo (4 inputs)":
        x = np.array([ni, ci, nt, ct]).reshape(1, -1)
        xs = scaler_full.transform(x)
        pred = model_full.predict(xs)[0]
    else:
        x = np.array([ci, ct]).reshape(1, -1)
        xs = scaler_ch.transform(x)
        pred = model_ch.predict(xs)[0]
    st.success(f"Nível previsto: {pred:.2f} m")

st.markdown("---")
st.markdown("[Repositório no GitHub](https://github.com/IgorKammerGrahl/ProjetoFinalIa)")
