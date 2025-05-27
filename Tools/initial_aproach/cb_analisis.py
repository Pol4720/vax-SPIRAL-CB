import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Configuración de la página
st.set_page_config(page_title="Análisis Costo-Beneficio Leptospirosis", layout="wide")

# Título
st.title("Modelo Costo-Beneficio para Vacunación contra Leptospirosis")

# Sidebar para parámetros de entrada
with st.sidebar:
    st.header("Parámetros de Entrada")
    
    # Costos de vacunación
    st.subheader("Costos de Vacunación")
    P = st.number_input("Población objetivo", min_value=1000, value=100000, step=1000)
    D = st.number_input("Dosis por persona", min_value=1, value=2)
    Cv = st.number_input("Costo por dosis ($)", min_value=0.0, value=5.0)
    Ca = st.number_input("Costo de aplicación ($/persona)", min_value=0.0, value=2.0)
    C_alm = st.number_input("Costo almacenamiento (% de Cv)", min_value=0, value=10)
    tasa_ea = st.number_input("Tasa efectos adversos (%)", min_value=0.0, value=0.1)
    costo_ea = st.number_input("Costo por efecto adverso ($)", min_value=0, value=200)
    
    # Parámetros de enfermedad
    st.subheader("Parámetros de Enfermedad")
    I = st.number_input("Incidencia anual (casos)", min_value=1, value=1000)
    E = st.slider("Efectividad vacuna (%)", 0.0, 100.0, 80.0)
    Ch = st.number_input("Costo hospitalización por caso ($)", value=700)
    Cm = st.number_input("Costo medicamentos ($)", value=50)
    Cd = st.number_input("Costo diagnóstico ($)", value=30)
    Ci = st.number_input("Costo indirectos ($)", value=200)
    tasa_graves = st.slider("Tasa casos graves (%)", 0.0, 100.0, 10.0)
    Oc = st.number_input("Otros costos por caso grave ($)", value=500)

# Funciones del modelo
def calcular_modelo():
    # Costos totales vacunación
    C_alm_total = Cv * (C_alm/100)
    CTV = P * D * (Cv + Ca + C_alm_total) + (P * (tasa_ea/100) * costo_ea)
    
    # Beneficios
    CE = I * (E/100)
    Ct = Ch + Cm + Cd + Ci
    B = CE * Ct + (CE * (tasa_graves/100) * Oc)
    
    # Métricas
    BN = B - CTV
    RCB = B / CTV if CTV != 0 else 0
    CCE = CTV / CE if CE != 0 else 0
    
    return CTV, B, BN, RCB, CCE

# Simulación Monte Carlo
def monte_carlo_sim(n_sim=1000):
    inputs = []
    outputs = []
    
    for _ in range(n_sim):
        # Generar valores aleatorios
        E_sim = np.random.normal(loc=E, scale=5)  # Efectividad ±5%
        I_sim = np.random.poisson(lam=I)          # Incidencia
        Cv_sim = np.random.uniform(Cv*0.8, Cv*1.2) # Costo vacuna ±20%
        
        # Calcular modelo
        CE_sim = I_sim * (E_sim/100)
        Ct_sim = Ch + Cm + Cd + Ci
        B_sim = CE_sim * Ct_sim + (CE_sim * (tasa_graves/100) * Oc)
        CTV_sim = P * D * (Cv_sim + Ca + Cv_sim*(C_alm/100)) + (P * (tasa_ea/100) * costo_ea)
        BN_sim = B_sim - CTV_sim
        
        inputs.append((E_sim, I_sim, Cv_sim))
        outputs.append(BN_sim)
    
    return inputs, outputs

# Análisis de sensibilidad
def sensitivity_analysis(base_params):
    variables = ['E', 'I', 'Cv', 'Ch', 'tasa_graves']
    results = []
    
    for var in variables:
        # Variación ±20%
        original_value = base_params[var]
        
        # High
        modified_params = base_params.copy()
        modified_params[var] = original_value * 1.2
        BN_high = calcular_modelo_modified(modified_params)[2]
        
        # Low
        modified_params[var] = original_value * 0.8
        BN_low = calcular_modelo_modified(modified_params)[2]
        
        results.append({
            'Variable': var,
            'Impacto': (BN_high - BN_low)/abs(base_BN) if base_BN !=0 else 0
        })
    
    return pd.DataFrame(results)

# Ejecutar modelo base
CTV, B, BN, RCB, CCE = calcular_modelo()
base_params = {
    'E': E,
    'I': I,
    'Cv': Cv,
    'Ch': Ch,
    'tasa_graves': tasa_graves
}
base_BN = BN

# Resultados principales
st.header("Resultados Principales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Costo Total Vacunación", f"${CTV:,.2f}")
col2.metric("Beneficios Totales", f"${B:,.2f}")
col3.metric("Beneficio Neto", f"${BN:,.2f}", delta_color="inverse")
col4.metric("Relación Costo-Beneficio", f"{RCB:.2f}")

# Gráfico de barras comparativo
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['Costos Vacunación', 'Beneficios'],
    y=[CTV, B],
    marker_color=['#FF4B4B', '#0068C9']
))
fig.update_layout(
    title="Comparación Costos vs Beneficios",
    yaxis_title="Monto ($)",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# Análisis de sensibilidad
st.header("Análisis de Sensibilidad")
sens_df = sensitivity_analysis(base_params)
fig2 = px.bar(sens_df, x='Impacto', y='Variable', orientation='h',
             title='Impacto en Beneficio Neto (Variación ±20%)')
st.plotly_chart(fig2, use_container_width=True)

# Simulación Monte Carlo
st.header("Simulación de Monte Carlo")
inputs, outputs = monte_carlo_sim(1000)

fig3 = go.Figure()
fig3.add_trace(go.Histogram(x=outputs, nbinsx=50, 
                          marker_color='#00CC96',
                          name='Beneficio Neto'))
fig3.add_vline(x=0, line_dash="dash", line_color="red")
fig3.update_layout(
    title="Distribución del Beneficio Neto (1000 simulaciones)",
    xaxis_title="Beneficio Neto ($)",
    yaxis_title="Frecuencia",
    template="plotly_white"
)
st.plotly_chart(fig3, use_container_width=True)

# Mostrar probabilidad de rentabilidad
prob_rentable = sum(np.array(outputs) > 0) / len(outputs) * 100
st.metric("Probabilidad de ser rentable", f"{prob_rentable:.1f}%")

# Explicación de resultados
with st.expander("Interpretación de Resultados"):
    st.markdown("""
    - **Beneficio Neto Positivo:** La vacunación es económicamente favorable
    - **RCB > 1:** Los beneficios superan los costos
    - **Gráfico de Sensibilidad:** Muestra qué variables tienen mayor impacto
    - **Histograma:** Distribución de posibles resultados considerando incertidumbre
    """)