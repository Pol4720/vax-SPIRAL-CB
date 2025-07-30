import streamlit as st
import numpy as np

from src.model_vaccine import LeptospirosisVaccineModel
from src.model import LeptospirosisModel
from src.parameters import show_parameters
from src.simulation import run_simulation
from src.cost_benefit import analyze_cost_benefit
from src.sensitivity import run_sensitivity_analysis
from src.parameter_fitting.main import parameter_fitting_section
from src.initial_conditions_analysis import initial_conditions_experiment

st.set_page_config(layout="wide")
st.title("Leptospirosis Vaccination Simulator")

# Instanciar modelos
vaccine_model_obj = LeptospirosisVaccineModel()
no_vaccine_model_obj = LeptospirosisModel()

params = vaccine_model_obj.params.copy()
initial_conditions = vaccine_model_obj.initial_conditions.copy()

# Reordenar las secciones: primero experimentación de condiciones iniciales
sections = [
    "Parameter Fitting",
    "Model Parameters",
    "Initial Conditions Analysis",  
    "Run Simulation",
    "Cost-Benefit Analysis",
    "Sensitivity Analysis"
]
with st.sidebar:
    st.markdown(
        """
        <style>
        .sidebar-title {
            font-size: 1.5em;
            font-weight: bold;
            color: #4F8BF9;
            margin-bottom: 20px;
        }
        .sidebar-radio label {
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
    section = st.radio(
        "Select Section",
        sections,
        key="sidebar_radio",
        format_func=lambda x: f"🔹 {x}" if x != "Run Simulation" else f"🚀 {x}",
        help="Choose a section to explore"
    )


# Inicializar condiciones iniciales en session_state según el modelo seleccionado
if "initial_conditions" not in st.session_state:
    # Por defecto, usar las condiciones iniciales del modelo con vacuna (9 compartimentos)
    st.session_state["initial_conditions"] = vaccine_model_obj.initial_conditions.copy()

if section == "Parameter Fitting":
    parameter_fitting_section(no_vaccine_model_obj)

# Model Parameters Section
elif section == "Model Parameters":
    params = show_parameters(vaccine_model_obj, no_vaccine_model_obj)

# Initial Conditions Analysis Section
elif section == "Initial Conditions Analysis":
    # Detectar el modelo seleccionado y ajustar el tamaño de las condiciones iniciales
    def update_initial_conditions(new_ic):
        st.session_state["initial_conditions"] = new_ic
        # Ajustar condiciones iniciales según el número de compartimentos de cada modelo
        if len(new_ic) == 9:
            vaccine_model_obj.initial_conditions = new_ic.copy()
            # Para el modelo sin vacuna, usar solo los primeros 8 compartimentos
            no_vaccine_model_obj.initial_conditions = new_ic[:8]
        elif len(new_ic) == 8:
            no_vaccine_model_obj.initial_conditions = new_ic.copy()
            # Para el modelo con vacuna, agregar 0 para Vh si falta
            vaccine_model_obj.initial_conditions = new_ic + [0.0]
    initial_conditions_experiment(update_callback=update_initial_conditions)

# Run Simulation Section
elif section == "Run Simulation":
    # Usar las condiciones iniciales seleccionadas en la sección previa
    initial_conditions = st.session_state["initial_conditions"]
    # Ajustar condiciones iniciales según el modelo
    if len(initial_conditions) == 9:
        vaccine_model_obj.initial_conditions = initial_conditions.copy()
        no_vaccine_model_obj.initial_conditions = initial_conditions[:8]
    elif len(initial_conditions) == 8:
        no_vaccine_model_obj.initial_conditions = initial_conditions.copy()
        vaccine_model_obj.initial_conditions = initial_conditions + [0.0]
    run_simulation(vaccine_model_obj, no_vaccine_model_obj, params, initial_conditions)

# Cost-Benefit Analysis Section
elif section == "Cost-Benefit Analysis":
    initial_conditions = st.session_state["initial_conditions"]
    if len(initial_conditions) == 9:
        vaccine_model_obj.initial_conditions = initial_conditions.copy()
    elif len(initial_conditions) == 8:
        vaccine_model_obj.initial_conditions = initial_conditions + [0.0]
    analyze_cost_benefit(vaccine_model_obj, params, vaccine_model_obj.initial_conditions)

# Sensitivity Analysis Section
elif section == "Sensitivity Analysis":
    initial_conditions = st.session_state["initial_conditions"]
    if len(initial_conditions) == 9:
        vaccine_model_obj.initial_conditions = initial_conditions.copy()
    elif len(initial_conditions) == 8:
        vaccine_model_obj.initial_conditions = initial_conditions + [0.0]
    run_sensitivity_analysis(vaccine_model_obj, params, vaccine_model_obj.initial_conditions)
