import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.model_vaccine import LeptospirosisVaccineModel
from src.model import LeptospirosisModel

st.set_page_config(layout="wide")
st.title("Leptospirosis Vaccination Simulator")

# Instanciar modelos
vaccine_model_obj = LeptospirosisVaccineModel()
no_vaccine_model_obj = LeptospirosisModel()

params = vaccine_model_obj.params.copy()
initial_conditions = vaccine_model_obj.initial_conditions.copy()

sections = ["Model Parameters", "Run Simulation", "Cost-Benefit Analysis", "Sensitivity Analysis"]
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

# Model Parameters Section
if section == "Model Parameters":
    st.header("Model Parameters")
    st.markdown("Set the parameters below. Default values are pre-filled.")

    # Mostrar nombre y descripción usando param_comments
    for key in params:
        comment = vaccine_model_obj.param_comments.get(key, "")
        label = f"{key} ({comment})" if comment else key
        if isinstance(params[key], float):
            params[key] = st.number_input(label, value=params[key])

    # Actualizar parámetros en los modelos
    vaccine_model_obj.params = params.copy()
    no_vaccine_model_obj.params = {k: v for k, v in params.items() if k in no_vaccine_model_obj.params}

# Run Simulation Section
elif section == "Run Simulation":
    st.header("Simulation Results")

    vaccine_model_obj.params = params.copy()
    vaccine_model_obj.initial_conditions = initial_conditions.copy()

    sol_vax = vaccine_model_obj.solve()
    sol_no_vax = no_vaccine_model_obj.solve()

    t_eval = vaccine_model_obj.t_eval

    fig, ax = plt.subplots()
    ax.plot(t_eval, sol_vax.y[2], label="With Vaccine", linewidth=2)
    ax.plot(t_eval, sol_no_vax.y[2], label="Without Vaccine", linestyle='--', linewidth=2)
    ax.set_title("Infectious Humans Over Time")
    ax.set_xlabel("Days")
    ax.set_ylabel("Infectious Humans")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Mostrar resultados de los métodos plot() y plot_infectious_humans() en la interfaz
    st.subheader("Resultados detallados del modelo sin vacuna")
    no_vaccine_model_obj.plot()

    st.subheader("Resultados detallados del modelo con vacuna")
    vaccine_model_obj.plot_compartment()


# Cost-Benefit Analysis Section
elif section == "Cost-Benefit Analysis":
    st.header("Cost-Benefit Analysis")

    dose_cost = st.number_input("Cost per Vaccine Dose ($)", 0.0, 500.0, 20.0)
    doses_per_person = st.number_input("Doses per Person", 1, 3, 1)

    prop_mild = st.slider("Proportion Mild", 0.0, 1.0, 0.7)
    prop_mod = st.slider("Proportion Moderate", 0.0, 1.0, 0.2)
    prop_sev = st.slider("Proportion Severe", 0.0, 1.0, 0.1)
    cost_mild = st.number_input("Cost Mild Case ($)", 0.0, 1000.0, 50.0)
    cost_mod = st.number_input("Cost Moderate Case ($)", 0.0, 5000.0, 500.0)
    cost_sev = st.number_input("Cost Severe Case ($)", 0.0, 20000.0, 5000.0)

    vaccine_model_obj.params = params.copy()
    vaccine_model_obj.initial_conditions = initial_conditions.copy()
    sol_vax = vaccine_model_obj.solve(with_vaccine=True)
    sol_no_vax = vaccine_model_obj.solve(with_vaccine=False)
    t_eval = vaccine_model_obj.t_eval

    # Calcular el número total de personas que han pasado por el compartimento de infectados (incidencia acumulada)
    total_infected_no_vax = sol_no_vax.y[2, -1]  
    total_infected_vax = sol_vax.y[2, -1] 
    avoided_cases = total_infected_no_vax - total_infected_vax

    cases_mild = avoided_cases * prop_mild
    cases_mod = avoided_cases * prop_mod
    cases_sev = avoided_cases * prop_sev

    # El costo de vacunación depende de la cobertura y la población susceptible inicial
    costs = dose_cost * doses_per_person * params['ϕ'] * initial_conditions[0]
    savings = (cases_mild * cost_mild) + (cases_mod * cost_mod) + (cases_sev * cost_sev)
    net_benefit = savings - costs

    st.markdown(f"**Avoided Infections**: {avoided_cases:.0f}")
    st.markdown(f"**Healthcare Savings**: ${savings:,.2f}")
    st.markdown(f"**Vaccination Cost**: ${costs:,.2f}")
    st.markdown(f"**Net Benefit**: ${net_benefit:,.2f}")

# Sensitivity Analysis Section
elif section == "Sensitivity Analysis":
    st.header("Sensitivity Analysis: Vaccine Coverage")
    coverages = np.linspace(0, 0.5, 20)
    results = []

    for phi in coverages:
        p = params.copy()
        p['ϕ'] = phi
        vaccine_model_obj.params = p
        vaccine_model_obj.initial_conditions = initial_conditions.copy()
        sol = vaccine_model_obj.solve(with_vaccine=True)
        inf = np.trapz(sol.y[2], vaccine_model_obj.t_eval)
        results.append(inf)

    fig2, ax2 = plt.subplots()
    ax2.plot(coverages, results, marker='o')
    ax2.set_xlabel("Vaccine Coverage Rate (ϕ)")
    ax2.set_ylabel("Total Infections (AUC)")
    ax2.set_title("Sensitivity to Vaccine Coverage")
    ax2.grid()
    st.pyplot(fig2)
