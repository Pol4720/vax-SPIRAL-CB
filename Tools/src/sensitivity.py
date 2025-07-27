import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol

def run_sensitivity_analysis(vaccine_model_obj, params, initial_conditions):
    """
    Realiza el análisis de sensibilidad para los parámetros clave.
    
    Args:
        vaccine_model_obj: Instancia del modelo de vacunación
        params: Parámetros del modelo
        initial_conditions: Condiciones iniciales
    """
    st.header("Sensitivity Analysis: Key Parameters using Sobol Indices")

    # Definir el problema para SALib
    problem = {
        'num_vars': 5,
        'names': ['ϕ', 'β1', 'β2', 'β3', 'ε'],
        'bounds': [
            [0.0, 0.5],     # ϕ: tasa máxima de vacunación
            [0.0, 0.25],    # β1: tasa de transmisión 1
            [0.0, 0.25],    # β2: tasa de transmisión 2
            [0.0, 0.25],    # β3: tasa de transmisión 3
            [0.0, 1.0],     # ε: eficacia de la vacuna
        ]
    }

    # Permitir al usuario elegir el número de muestras
    N = st.slider("Number of samples (higher = slower, more accurate)", 4, 1024, 256, 
                  help="Reduce for faster analysis")
    param_values = sobol_sampler.sample(problem, N, calc_second_order=False)
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
        
        progress.progress((i + 1) / len(param_values), 
                          text=f"Running simulations... ({i+1}/{len(param_values)})")
    
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
