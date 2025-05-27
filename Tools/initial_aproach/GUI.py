import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Modelo de Leptospirosis con Vacunación",
    page_icon="🦠",
    layout="wide"
)

# Título principal
st.title("Simulación de Vacunación para Leptospirosis")
st.markdown("---")

# Menú lateral
menu = st.sidebar.selectbox(
    "Menú Principal",
    ["📚 Descripción del Modelo", "⚙️ Simulación", "📊 Análisis Costo-Beneficio", "📈 Resultados Históricos"]
)

# Sección: Descripción del Modelo
if menu == "📚 Descripción del Modelo":
    st.header("Modelo SEIRV para Leptospirosis")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Ecuaciones del Modelo
        - **Suscetibles (S):**
            \n$\frac{dS}{dt} = \Lambda + \gamma R + \gamma_v V - (\lambda + \mu + \nu)S$
            
        - **Vacunados (V):**
            \n$\frac{dV}{dt} = \nu S - (\gamma_v + \mu)V$
            
        - **Expuestos (E):**
            \n$\frac{dE}{dt} = \lambda S + (1-\epsilon)\lambda V - (\theta + \mu)E$
            
        - **Infectados (I):**
            \n$\frac{dI}{dt} = \theta E - (\alpha + \delta + \mu)I$
            
        - **Recuperados (R):**
            \n$\frac{dR}{dt} = \delta I - (\gamma + \mu)R$
        """)
    
    with col2:
        st.markdown("### Diagrama de Flujo del Modelo")
        img = Image.open("seirv_diagram.png")  # Asegúrate de tener una imagen
        st.image(img, use_column_width=True)
    
    st.markdown("---")
    st.markdown("### Parámetros Clave")
    st.table({
        "Parámetro": ["ν", "ε", "γ_v", "β1", "β2"],
        "Descripción": ["Tasa de vacunación", "Efectividad vacuna", 
                       "Duración inmunidad", "Transmisión roedores", 
                       "Transmisión ambiente"],
        "Unidades": ["personas/día", "%", "días⁻¹", "día⁻¹", "día⁻¹"]
    })

# Sección: Simulación
elif menu == "⚙️ Simulación":
    st.header("Configuración de la Simulación")
    
    # Dividir en columnas
    col_params, col_vac = st.columns(2)
    
    with col_params:
        st.subheader("Parámetros de la Enfermedad")
        beta1 = st.slider("Transmisión roedores (β1)", 0.0, 0.01, 0.00033)
        beta2 = st.slider("Transmisión ambiente (β2)", 0.0, 0.1, 0.0815)
        theta = st.slider("Tasa de progresión (θ)", 0.0, 0.2, 0.092)
        delta = st.slider("Tasa de recuperación (δ)", 0.0, 0.1, 0.072)
    
    with col_vac:
        st.subheader("Parámetros de Vacunación")
        nu = st.slider("Tasa de vacunación (ν)", 0.0, 0.1, 0.01)
        epsilon = st.slider("Efectividad vacuna (ε)", 0.0, 1.0, 0.8)
        gamma_v = st.slider("Duración inmunidad (γ_v)", 0.0, 0.01, 1/365)
    
    # Sistema de ecuaciones diferenciales (igual que antes)
    def run_simulation(params):
        # Implementar modelo aquí
        return 6
    
    if st.button("Ejecutar Simulación"):
        with st.spinner("Calculando..."):
            # Ejecutar simulación
            results = run_simulation(locals())
            
            # Mostrar gráficos
            st.subheader("Resultados de la Simulación")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results['t'], y=results['I'], name='Infectados'))
            fig.add_trace(go.Scatter(x=results['t'], y=results['V'], name='Vacunados'))
            fig.update_layout(title="Dinámica Temporal de la Población")
            st.plotly_chart(fig, use_container_width=True)

# Sección: Análisis Costo-Beneficio
elif menu == "📊 Análisis Costo-Beneficio":
    st.header("Análisis de Costo-Beneficio")
    
    col_cost, col_health = st.columns(2)
    
    with col_cost:
        st.subheader("Costos de Vacunación")
        cost_dose = st.number_input("Costo por dosis (USD)", 0, 100, 15)
        campaign_cost = st.number_input("Costo de campaña (USD)", 0, 100000, 5000)
    
    with col_health:
        st.subheader("Costos de Salud")
        cost_case = st.number_input("Costo por caso tratado (USD)", 0, 10000, 3000)
        productivity_loss = st.number_input("Pérdida productividad/día (USD)", 0, 500, 150)
    
    if st.button("Calcular Relación Costo-Beneficio"):
        # Cálculos y visualización
        st.metric(label="Relación Costo-Beneficio", value="3.2:1")
        
        fig = go.Figure()
        fig.add_bar(x=['Costo Total', 'Beneficio'], y=[120000, 384000])
        st.plotly_chart(fig, use_container_width=True)

# Sección: Resultados Históricos
elif menu == "📈 Resultados Históricos":
    st.header("Comparación Histórica de Escenarios")
    
    scenarios = st.multiselect("Seleccionar escenarios", 
                              ["Sin Vacunación", "Vacunación Baja", 
                               "Vacunación Media", "Vacunación Alta"])
    
    if scenarios:
        fig = go.Figure()
        for scenario in scenarios:
            # Datos simulados
            fig.add_trace(go.Scatter(x=np.arange(100), y=np.random.randn(100), name=scenario))
        
        fig.update_layout(title="Comparación de Escenarios")
        st.plotly_chart(fig, use_container_width=True)

# Estilos CSS personalizados
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .st-bb {
        border-bottom: 2px solid #e1e4e8;
    }
    .css-18e3th9 {
        padding: 2rem 1rem;
    }
</style>
""", unsafe_allow_html=True)