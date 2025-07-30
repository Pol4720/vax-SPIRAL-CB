import streamlit as st
import numpy as np

def run_simulation(vaccine_model_obj, no_vaccine_model_obj, params, initial_conditions):
    """
    Ejecuta la simulación y muestra los resultados.
    
    Args:
        vaccine_model_obj: Instancia del modelo de vacunación
        no_vaccine_model_obj: Instancia del modelo sin vacunación
        params: Parámetros del modelo
        initial_conditions: Condiciones iniciales
    """
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

    # --- Selección de compartimento para graficar ---
    st.subheader("Resultados detallados del modelo sin vacuna")
    compartimentos_nv = {
        "Susceptibles Humanos (Sh)": 0,
        "Expuestos Humanos (Eh)": 1,
        "Infectados Humanos (Ih)": 2,
        "Recuperados Humanos (Rh)": 3,
        "Vectores (Sv, Iv, Rv)": "vectores",
        "Bacterias en el ambiente (Bl)": 7,
    }
    selected_nv = st.selectbox(
        "Compartimento a visualizar (sin vacuna)",
        list(compartimentos_nv.keys()),
        index=2,
        key="sim_nv_comp"
    )
    no_vaccine_model_obj.plot(selected=selected_nv)

    st.subheader("Resultados detallados del modelo con vacuna")
    compartimentos_vax = {
        "Susceptible (Sh)": 0,
        "Exposed (Eh)": 1,
        "Infectious (Ih)": 2,
        "Recovered (Rh)": 3,
        "Vaccinated (Vh)": 8,
        "Bacterias en ambiente (Bl)": 7,
        "Todos los compartimentos de vectores": "all_vectors",
        "Vaccination rate (personas/día)": "vaccination_rate"
    }
    selected_vax = st.selectbox(
        "Compartimento a visualizar (con vacuna)",
        list(compartimentos_vax.keys()),
        index=2,
        key="sim_vax_comp"
    )
    vaccine_model_obj.plot_compartment(selected=selected_vax)
    
    return sol_vax, sol_no_vax
