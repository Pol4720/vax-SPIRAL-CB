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


if section == "Parameter Fitting":
    parameter_fitting_section(no_vaccine_model_obj)

# Model Parameters Section
elif section == "Model Parameters":
    params = show_parameters(vaccine_model_obj, no_vaccine_model_obj)

# Initial Conditions Analysis Section
elif section == "Initial Conditions Analysis":
    # Permite modificar las condiciones iniciales y las guarda en session_state
    def update_initial_conditions(new_ic):
        st.session_state["initial_conditions"] = new_ic
        vaccine_model_obj.initial_conditions = new_ic.copy()
        no_vaccine_model_obj.initial_conditions = new_ic.copy()
    initial_conditions_experiment(update_callback=update_initial_conditions)

# Run Simulation Section
elif section == "Run Simulation":
    # Usar las condiciones iniciales seleccionadas en la sección previa
    initial_conditions = st.session_state["initial_conditions"]
    vaccine_model_obj.initial_conditions = initial_conditions.copy()
    no_vaccine_model_obj.initial_conditions = initial_conditions.copy()
    run_simulation(vaccine_model_obj, no_vaccine_model_obj, params, initial_conditions)

# Cost-Benefit Analysis Section
elif section == "Cost-Benefit Analysis":
    analyze_cost_benefit(vaccine_model_obj, params, st.session_state["initial_conditions"])

# Sensitivity Analysis Section
elif section == "Sensitivity Analysis":
    run_sensitivity_analysis(vaccine_model_obj, params, st.session_state["initial_conditions"])
