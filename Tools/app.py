import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.model_vaccine import LeptospirosisVaccineModel
from src.model import LeptospirosisModel
from scipy.integrate import simpson
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sampler

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

    # Selector de periodo de simulación
    period_options = {
        "6 months": 182,
        "1 year": 365,
        "2 years": 730,
        "5 years": 1825,
        "10 years": 3650
    }
    selected_period = st.selectbox(
        "Select simulation period",
        list(period_options.keys()),
        index=1
    )
    t_end = period_options[selected_period]

    # Actualizar t_span en los modelos
    t_eval = np.linspace(0, t_end, t_end + 1)
    vaccine_model_obj.t_span = (0, t_end)
    vaccine_model_obj.t_eval = t_eval
    no_vaccine_model_obj.t_span = (0, t_end)
    no_vaccine_model_obj.t_eval = t_eval

    vaccine_model_obj.params = params.copy()
    vaccine_model_obj.initial_conditions = initial_conditions.copy()

    sol_vax = vaccine_model_obj.solve()
    sol_no_vax = no_vaccine_model_obj.solve()

    t_eval = vaccine_model_obj.t_eval

    # fig, ax = plt.subplots()
    # ax.plot(t_eval, sol_vax.y[2], label="With Vaccine", linewidth=2)
    # ax.plot(t_eval, sol_no_vax.y[2], label="Without Vaccine", linestyle='--', linewidth=2)
    # ax.set_title("Infectious Humans Over Time")
    # ax.set_xlabel("Days")
    # ax.set_ylabel("Infectious Humans")
    # ax.legend()
    # ax.grid()
    # st.pyplot(fig)

    # Mostrar resultados de los métodos plot() y plot_infectious_humans() en la interfaz
    st.subheader("Resultados detallados del modelo sin vacuna")
    no_vaccine_model_obj.plot()

    st.subheader("Resultados detallados del modelo con vacuna")
    vaccine_model_obj.plot_compartment()


# Cost-Benefit Analysis Section
elif section == "Cost-Benefit Analysis":
    st.header("Cost-Benefit Analysis")

    dose_cost = st.number_input("Cost per Vaccine Dose ($)", 0.0, 500.0, 15.0)
    doses_per_person = st.number_input("Doses per Person", 1, 3, 1)
    prop_mild = st.slider("Proportion Mild", 0.0, 1.0, 0.85)
    prop_mod = st.slider("Proportion Moderate", 0.0, 1.0, 0.05)
    prop_sev = st.slider("Proportion Severe", 0.0, 1.0, 0.10)
    cost_mild = st.number_input("Cost Mild Case ($)", 0.0, 1000.0, 159.0)
    cost_mod = st.number_input("Cost Moderate Case ($)", 0.0, 5000.0, 1996.0)
    cost_sev = st.number_input("Cost Severe Case ($)", 0.0, 40000.0, 33260.0)


    vaccine_model_obj.params = params.copy()
    vaccine_model_obj.initial_conditions = initial_conditions.copy()
    sol_vax = vaccine_model_obj.solve(with_vaccine=True)
    sol_no_vax = vaccine_model_obj.solve(with_vaccine=False)
    t_eval = vaccine_model_obj.t_eval

    # Calcular el número total de personas que han pasado por el compartimento de infectados (incidencia acumulada)
    # Usar la integral de theta * expuestos (E) sobre el tiempo
    theta = params['θ']
    total_infected_no_vax = simpson(theta * sol_no_vax.y[1], t_eval)
    total_infected_vax = simpson(theta * sol_vax.y[1], t_eval)
    avoided_cases = total_infected_no_vax - total_infected_vax

    cases_mild = avoided_cases * prop_mild
    cases_mod = avoided_cases * prop_mod
    cases_sev = avoided_cases * prop_sev

    # Calcular la cantidad total de vacunados como la integral de la tasa de vacunación (ϕ * S) en el tiempo
    # Calcular la cantidad total de vacunados usando la tasa de vacunación dinámica nu_dinamica
    Ih = sol_vax.y[2]  # Compartimento de infectados humanos en el modelo con vacuna
    susceptibles = sol_vax.y[0]  # S(t) del modelo con vacuna

    # Calcular la tasa de vacunación dinámica en cada punto de tiempo
    nu_dyn = np.array([
        vaccine_model_obj.nu_dinamica(Ih[i], t_eval[i])
        for i in range(len(t_eval))
    ])
    total_vaccinated = simpson(nu_dyn * susceptibles, t_eval)
    costs = dose_cost * doses_per_person * total_vaccinated
    savings = (cases_mild * cost_mild) + (cases_mod * cost_mod) + (cases_sev * cost_sev)
    net_benefit = savings - costs

    st.markdown(f"**Avoided Infections**: {avoided_cases:.0f}")
    st.markdown(f"**Healthcare Savings**: ${savings:,.2f}")
    st.markdown(f"**Vaccination Cost**: ${costs:,.2f}")
    st.markdown(f"**Net Benefit**: ${net_benefit:,.2f}")
    
    # Calcular el coeficiente de costo-beneficio
    if costs > 0:
        cost_benefit_ratio = savings / costs
    else:
        cost_benefit_ratio = float('inf')

    st.markdown(f"**Cost-Benefit Ratio**: {cost_benefit_ratio:.2f}")

    # Determinar si es rentable
    if cost_benefit_ratio > 1:
        st.success("Vaccination is cost-effective (savings exceed costs).")
    else:
        st.warning("Vaccination is NOT cost-effective (costs exceed savings).")

# Sensitivity Analysis Section
elif section == "Sensitivity Analysis":
    st.header("Sensitivity Analysis: Key Parameters using Sobol Indices")

    # Definir el problema para SALib con los parámetros solicitados
    problem = {
        'num_vars': 5,
        'names': ['ϕ', 'β1', 'β2', 'β3', 'ε'],
        'bounds': [
            [0.0, 0.5],      # ϕ: tasa máxima de vacunación
            [0.0, 1.0],      # β1: tasa de transmisión 1
            [0.0, 1.0],      # β2: tasa de transmisión 2
            [0.0, 1.0],      # β3: tasa de transmisión 3
            [0.0, 1.0],      # ε: eficacia de la vacuna
        ]
    }

    # Permitir al usuario elegir el número de muestras (N) para acelerar el análisis
    N = st.slider("Number of samples (higher = slower, more accurate)", 4, 32, 8, help="Reduce for faster analysis")
    param_values = sobol_sampler.sample(problem, N)
    st.write("Sampled parameter values (first 10 rows):")
    st.dataframe(pd.DataFrame(param_values, columns=problem['names']).head(10))
    Y = np.zeros(param_values.shape[0])
    progress = st.progress(0, text="Running simulations...")
    for i, vals in enumerate(param_values):
        phi, beta1, beta2, beta3, eficacia = vals
        p = params.copy()
        p['ϕ'] = phi
        p['β1'] = beta1
        p['β2'] = beta2
        p['β3'] = beta3
        p['ε'] = eficacia
        vaccine_model_obj.params = p
        vaccine_model_obj.initial_conditions = initial_conditions.copy()
        sol = vaccine_model_obj.solve(with_vaccine=True)
        auc_inf = simpson(sol.y[2], vaccine_model_obj.t_eval)
        Y[i] = auc_inf
        progress.progress((i + 1) / len(param_values), text=f"Running simulations... ({i+1}/{len(param_values)})")
    progress.empty()

    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)

    st.subheader("Sobol Sensitivity Indices for Key Parameters")
    indices = ['ϕ', 'β1', 'β2', 'β3', 'ε']
    df = pd.DataFrame({
        'Parameter': indices,
        'S1': Si['S1'],
        'ST': Si['ST']
    })
    st.dataframe(df)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df['Parameter'], df['S1'], label='S1', alpha=0.7)
    ax.bar(df['Parameter'], df['ST'], label='ST', alpha=0.7, bottom=df['S1'])
    ax.set_ylabel("Sobol Index")
    ax.set_title("Sobol Sensitivity Indices for Key Parameters")
    ax.legend()
    st.pyplot(fig)
