import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def plot_comparison(model_obj, real_data, data_type):
    """
    Compara los datos reales con la simulación del modelo.
    
    Args:
        model_obj: Modelo de leptospirosis
        real_data: DataFrame con datos reales
        data_type: Tipo de datos
    """
    # Evaluar en un rango más amplio para visualización
    t_span = (0, max(real_data['day']))
    t_eval = np.linspace(*t_span, int(t_span[1]) + 1)
    
    sol = solve_ivp(
        lambda t, y: model_obj.model(t, y),
        t_span=t_span,
        y0=model_obj.initial_conditions,
        t_eval=t_eval,
        method='RK45'
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if data_type == "casos_diarios":
        # Graficar casos diarios (Ih)
        ax.plot(t_eval, sol.y[2], 'b-', label='Modelo ajustado (Ih)')
        ax.scatter(real_data['day'], real_data['cases'], color='red', label='Datos reales')
        ax.set_ylabel("Casos")
    
    elif data_type == "casos_acumulados":
        # Calcular casos acumulados del modelo
        cases_cumulative = np.cumsum(sol.y[2])
        ax.plot(t_eval, cases_cumulative, 'b-', label='Modelo ajustado (acumulado)')
        ax.scatter(real_data['day'], real_data['cumulative_cases'], color='red', label='Datos reales')
        ax.set_ylabel("Casos acumulados")
    
    elif data_type == "compartimentos":
        # Graficar múltiples compartimentos
        compartments = {
            'Sh': 0, 'Eh': 1, 'Ih': 2, 'Rh': 3,
            'Sv': 4, 'Iv': 5, 'Rv': 6, 'Bl': 7
        }
        
        # Determinar qué compartimentos mostrar
        compartments_to_show = [col for col in real_data.columns if col in compartments]
        
        if not compartments_to_show:
            st.error("No se encontraron compartimentos válidos en los datos")
            return
        
        # Crear una figura con múltiples subplots
        fig, axs = plt.subplots(len(compartments_to_show), 1, figsize=(12, 4*len(compartments_to_show)))
        
        for i, comp in enumerate(compartments_to_show):
            if len(compartments_to_show) > 1:
                ax = axs[i]
            else:
                ax = axs
            
            idx = compartments[comp]
            ax.plot(t_eval, sol.y[idx], 'b-', label=f'Modelo ajustado ({comp})')
            ax.scatter(real_data['day'], real_data[comp], color='red', label='Datos reales')
            ax.set_ylabel(comp)
            ax.set_xlabel("Días" if i == len(compartments_to_show)-1 else "")
            ax.legend()
            ax.grid(True)
        
        # Ajustar el layout
        plt.tight_layout()
        st.pyplot(fig)
        return
    
    ax.set_xlabel("Días")
    ax.set_title("Comparación entre modelo ajustado y datos reales")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_parameter_importance(param_names, sensitivity_values):
    """
    Visualiza la importancia relativa de los parámetros.
    
    Args:
        param_names: Lista de nombres de parámetros
        sensitivity_values: Valores de sensibilidad/importancia
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ordenar por importancia
    sorted_indices = np.argsort(sensitivity_values)
    sorted_params = [param_names[i] for i in sorted_indices]
    sorted_values = [sensitivity_values[i] for i in sorted_indices]
    
    # Crear barras horizontales
    ax.barh(sorted_params, sorted_values, color='skyblue')
    ax.set_xlabel("Importancia relativa")
    ax.set_title("Importancia de los parámetros en el ajuste")
    ax.grid(axis='x')
    
    st.pyplot(fig)

def plot_convergence_diagnostics(samples, param_names):
    """
    Visualiza diagnósticos de convergencia para MCMC.
    
    Args:
        samples: Muestras del MCMC
        param_names: Nombres de los parámetros
    """
    n_params = len(param_names)
    
    fig, axs = plt.subplots(n_params, 1, figsize=(10, 3*n_params))
    
    for i, param in enumerate(param_names):
        if n_params > 1:
            ax = axs[i]
        else:
            ax = axs
        
        # Trazar la cadena (trace plot)
        ax.plot(samples[:, i], 'b-', alpha=0.5)
        ax.set_ylabel(param)
        ax.set_xlabel("Muestra" if i == n_params-1 else "")
        ax.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
