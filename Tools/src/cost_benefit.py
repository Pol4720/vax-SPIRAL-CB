import streamlit as st
import numpy as np
from scipy.integrate import simpson

def analyze_cost_benefit(vaccine_model_obj, params, initial_conditions):
    """
    Realiza el análisis de costo-beneficio de la vacunación.
    
    Args:
        vaccine_model_obj: Instancia del modelo de vacunación
        params: Parámetros del modelo
        initial_conditions: Condiciones iniciales
    """
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

    # Calcular el número total de personas que han pasado por el compartimento de infectados
    theta = params['θ']
    total_infected_no_vax = simpson(theta * sol_no_vax.y[1], t_eval)
    total_infected_vax = simpson(theta * sol_vax.y[1], t_eval)
    avoided_cases = total_infected_no_vax - total_infected_vax

    cases_mild = avoided_cases * prop_mild
    cases_mod = avoided_cases * prop_mod
    cases_sev = avoided_cases * prop_sev

    # Calcular la cantidad total de vacunados
    Ih = sol_vax.y[2]  # Compartimento de infectados humanos
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
