import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações da página
st.set_page_config(
    page_title="Previsão do Nível do Rio - Rio do Sul",
    page_icon="🌊",
    layout="wide"
)

# Carregar modelo e scaler (com cache)
@st.cache_resource
def carregar_modelo():
    modelo = joblib.load('model_full.pkl')
    scaler = joblib.load('scaler_full.pkl')
    return modelo, scaler

# Função de previsão
def prever_nivel_rio(nivel_ituporanga, chuva_ituporanga, nivel_taio, chuva_taio):
    modelo, scaler = carregar_modelo()
    entrada = np.array([[nivel_ituporanga, chuva_ituporanga, nivel_taio, chuva_taio]])
    entrada_padronizada = scaler.transform(entrada)
    return modelo.predict(entrada_padronizada)[0]

# Interface principal
st.title("🌊 Sistema de Previsão do Nível do Rio em Rio do Sul")
st.markdown("""
**Inteligência Artificial 2025-1**  
*Regressão Linear - VERSÃO CORRIGIDA*  
Autor: Igor Kammer Grahl - Instituto Federal Catarinense
""")
st.divider()

# Entrada de dados
st.header("📥 Entrada de Dados")
col1, col2, col3, col4 = st.columns(4)

with col1:
    nivel_ituporanga = st.number_input("Nível do Rio em Ituporanga (cm)", 
                                      min_value=0.0, 
                                      max_value=1000.0, 
                                      value=50.0,
                                      step=0.1)

with col2:
    chuva_ituporanga = st.number_input("Chuva em Ituporanga (mm)", 
                                      min_value=0.0, 
                                      max_value=500.0, 
                                      value=10.0,
                                      step=0.1)

with col3:
    nivel_taio = st.number_input("Nível do Rio em Taió (cm)", 
                                min_value=0.0, 
                                max_value=1000.0, 
                                value=120.0,
                                step=0.1)

with col4:
    chuva_taio = st.number_input("Chuva em Taió (mm)", 
                                min_value=0.0, 
                                max_value=500.0, 
                                value=5.0,
                                step=0.1)

# Botão de previsão
if st.button("🔮 Prever Nível do Rio", use_container_width=True):
    with st.spinner("Processando..."):
        previsao = prever_nivel_rio(nivel_ituporanga, chuva_ituporanga, nivel_taio, chuva_taio)
    
    st.divider()
    st.header("📊 Resultado da Previsão")
    
    # Exibição visual do resultado
    nivel_min = 50
    nivel_max = 400
    nivel_percentual = (previsao - nivel_min) / (nivel_max - nivel_min) * 100
    nivel_percentual = max(0, min(100, nivel_percentual))
    
    st.metric(label="**Nível Previsto em Rio do Sul**", 
              value=f"{previsao:.1f} cm", 
              help="Valor previsto pelo modelo de regressão linear")
    
    # Barra de progresso visual
    st.progress(nivel_percentual/100, text=f"Estado do rio: {previsao:.1f} cm")
    
    # Interpretação do resultado
    if previsao < 100:
        st.success("✅ Condição Normal - Nível dentro da média histórica")
    elif previsao < 200:
        st.warning("⚠️ Atenção - Nível acima do normal")
    else:
        st.error("🚨 Alerta de Enchente - Nível perigosamente elevado")
    
    st.divider()

# Seção informativa
expander = st.expander("ℹ️ Informações Técnicas e Metodologia")
with expander:
    st.subheader("Sobre o Modelo")
    st.markdown("""
    - **Técnica:** Regressão Linear Multivariada
    - **Variáveis Utilizadas:**
        - Nível do rio em Ituporanga (cm)
        - Chuva em Ituporanga (mm)
        - Nível do rio em Taió (cm)
        - Chuva em Taió (mm)
    - **Performance do Modelo:**
        - R²: 0.96 (Teste)
        - RMSE: 6.3 cm
        - MAE: 4.8 cm
    """)
    
    st.subheader("Interpretação dos Resultados")
    st.markdown("""
    - **< 100 cm:** Condição normal
    - **100-200 cm:** Atenção - risco de alagamentos em áreas baixas
    - **> 200 cm:** Alerta de enchente
    """)
    
    st.subheader("Limitações")
    st.markdown("""
    - Previsões baseadas em dados históricos
    - Não considera eventos climáticos extremos repentinos
    - Precisão reduzida para valores fora da faixa de treinamento
    """)

# Rodapé
st.divider()
st.caption("Desenvolvido para o projeto de Inteligência Artificial 2025-1 - IFC")
st.caption("Dados fornecidos pela Defesa Civil de Santa Catarina")