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

# --- Inicializar session_state para params y initial_conditions ---
if "params" not in st.session_state:
    st.session_state["params"] = vaccine_model_obj.params.copy()
if "initial_conditions" not in st.session_state:
    st.session_state["initial_conditions"] = vaccine_model_obj.initial_conditions.copy()

# --- Secciones ---
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

# --- Sección Parameter Fitting ---
if section == "Parameter Fitting":
    # Usar los parámetros y condiciones iniciales actuales
    no_vaccine_model_obj.params = st.session_state["params"].copy()
    no_vaccine_model_obj.initial_conditions = st.session_state["initial_conditions"][:8]
    parameter_fitting_section(no_vaccine_model_obj)

# --- Sección Model Parameters ---
elif section == "Model Parameters":
    # Pasar los modelos y actualizar session_state con los valores editados
    params = show_parameters(vaccine_model_obj, no_vaccine_model_obj)
    st.session_state["params"] = params.copy()
    st.session_state["initial_conditions"] = vaccine_model_obj.initial_conditions.copy()

# --- Sección Initial Conditions Analysis ---
elif section == "Initial Conditions Analysis":
    def update_initial_conditions(new_ic):
        st.session_state["initial_conditions"] = new_ic
        # Ajustar condiciones iniciales según el número de compartimentos de cada modelo
        if len(new_ic) == 9:
            vaccine_model_obj.initial_conditions = new_ic.copy()
            no_vaccine_model_obj.initial_conditions = new_ic[:8]
        elif len(new_ic) == 8:
            no_vaccine_model_obj.initial_conditions = new_ic.copy()
            vaccine_model_obj.initial_conditions = new_ic + [0.0]
    initial_conditions_experiment(update_callback=update_initial_conditions)

# --- Sección Run Simulation ---
elif section == "Run Simulation":
    # Usar los parámetros y condiciones iniciales actuales
    params = st.session_state["params"].copy()
    initial_conditions = st.session_state["initial_conditions"]
    # Ajustar condiciones iniciales según el modelo
    if len(initial_conditions) == 9:
        vaccine_model_obj.initial_conditions = initial_conditions.copy()
        no_vaccine_model_obj.initial_conditions = initial_conditions[:8]
        ic_for_vax = initial_conditions.copy()
    elif len(initial_conditions) == 8:
        no_vaccine_model_obj.initial_conditions = initial_conditions.copy()
        vaccine_model_obj.initial_conditions = initial_conditions + [0.0]
        ic_for_vax = initial_conditions + [0.0]
    else:
        st.error("Las condiciones iniciales deben tener 8 (sin vacuna) o 9 (con vacuna) elementos.")
        st.stop()
    # Pasar los parámetros editados a ambos modelos
    vaccine_model_obj.params = params.copy()
    no_vaccine_model_obj.params = {k: v for k, v in params.items() if k in no_vaccine_model_obj.params}
    run_simulation(vaccine_model_obj, no_vaccine_model_obj, params, ic_for_vax)

# --- Sección Cost-Benefit Analysis ---
elif section == "Cost-Benefit Analysis":
    params = st.session_state["params"].copy()
    initial_conditions = st.session_state["initial_conditions"]
    if len(initial_conditions) == 9:
        vaccine_model_obj.initial_conditions = initial_conditions.copy()
    elif len(initial_conditions) == 8:
        vaccine_model_obj.initial_conditions = initial_conditions + [0.0]
    vaccine_model_obj.params = params.copy()
    analyze_cost_benefit(vaccine_model_obj, params, vaccine_model_obj.initial_conditions)

# --- Sección Sensitivity Analysis ---
elif section == "Sensitivity Analysis":
    params = st.session_state["params"].copy()
    initial_conditions = st.session_state["initial_conditions"]
    if len(initial_conditions) == 9:
        vaccine_model_obj.initial_conditions = initial_conditions.copy()
    elif len(initial_conditions) == 8:
        vaccine_model_obj.initial_conditions = initial_conditions + [0.0]
    vaccine_model_obj.params = params.copy()
    run_sensitivity_analysis(vaccine_model_obj, params, vaccine_model_obj.initial_conditions)
