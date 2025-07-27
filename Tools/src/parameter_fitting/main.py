import streamlit as st
import pandas as pd

from .data_loader import load_data_ui, prepare_data_for_fitting
from .mcmc import run_mcmc, display_mcmc_results
from .metaheuristic import run_metaheuristic, display_metaheuristic_results
from .utils import get_model_parameter_bounds

def parameter_fitting_section(model_obj):
    """
    Sección principal para el ajuste de parámetros del modelo.
    
    Args:
        model_obj: Modelo de leptospirosis sin vacuna
    """
    st.header("Ajuste de Parámetros")
    
    # Cargar datos
    real_data, data_type = load_data_ui()
    
    # Continuar solo si tenemos datos
    if real_data is not None:
        # Preparar datos para el ajuste
        prepared_data = prepare_data_for_fitting(real_data, data_type, model_obj)
        
        if prepared_data is not None or data_type == "condiciones_iniciales":
            # Seleccionar parámetros a ajustar
            st.subheader("Selección de parámetros a ajustar")
            
            param_category = st.selectbox(
                "Categoría de parámetros:",
                ["Transmisión (β)", "Tasas de transición", "Reclutamiento", "Patógeno", "Todos"],
                help="Seleccione la categoría de parámetros que desea ajustar"
            )
            
            # Mapear categorías a tipos de parámetros
            category_to_type = {
                "Transmisión (β)": "beta",
                "Tasas de transición": "rates",
                "Reclutamiento": "recruitment",
                "Patógeno": "pathogen",
                "Todos": "all"
            }
            
            param_bounds = get_model_parameter_bounds(model_obj, category_to_type[param_category])
            
            # Permitir seleccionar parámetros específicos
            st.write("Seleccione los parámetros que desea ajustar:")
            
            selected_params = {}
            for param, bounds in param_bounds.items():
                if st.checkbox(f"Ajustar {param}", value=True if param in ['β1', 'β2', 'β3'] else False):
                    col1, col2 = st.columns(2)
                    with col1:
                        min_val = st.number_input(f"Valor mínimo para {param}", value=bounds['min'], format="%.6f")
                    with col2:
                        max_val = st.number_input(f"Valor máximo para {param}", value=bounds['max'], format="%.6f")
                    selected_params[param] = (min_val, max_val)
            
            if not selected_params:
                st.warning("Por favor, seleccione al menos un parámetro para ajustar.")
                return
            
            # Seleccionar método de ajuste
            st.subheader("Método de Ajuste")
            
            method = st.radio(
                "Seleccione el método de ajuste:",
                ["MCMC (Markov Chain Monte Carlo)", "Optimización Metaheurística"],
                key="fitting_method"
            )
            
            if method == "MCMC (Markov Chain Monte Carlo)":
                # Configuración de MCMC
                st.subheader("Configuración de MCMC")
                
                n_walkers = st.slider("Número de caminantes", 10, 100, 20)
                n_steps = st.slider("Número de pasos", 100, 2000, 500)
                burn_in = st.slider("Burn-in (pasos a descartar)", 50, 500, 100)
                
                if st.button("Ejecutar MCMC"):
                    with st.spinner("Ejecutando MCMC..."):
                        results = run_mcmc(model_obj, real_data, selected_params, data_type, n_walkers, n_steps, burn_in)
                        
                        # Mostrar resultados
                        if results:
                            samples, param_names, log_prob_samples, best_params = results
                            
                            # Actualizar el modelo con los mejores parámetros
                            updated_params = model_obj.params.copy()
                            for i, param in enumerate(param_names):
                                updated_params[param] = best_params[i]
                            
                            # Mostrar los resultados del MCMC
                            display_mcmc_results(model_obj, real_data, samples, param_names, log_prob_samples, best_params, updated_params, data_type)
            
            elif method == "Optimización Metaheurística":
                # Configuración de Metaheurísticas
                st.subheader("Configuración de Metaheurísticas")
                
                optimizer = st.selectbox(
                    "Seleccione el algoritmo:",
                    ["Evolución Diferencial", "Basin-Hopping", "Nelder-Mead"]
                )
                
                max_iter = st.slider("Máximo de iteraciones", 10, 200, 50)
                
                if st.button("Ejecutar Optimización"):
                    with st.spinner(f"Ejecutando {optimizer}..."):
                        results = run_metaheuristic(model_obj, real_data, selected_params, optimizer, data_type, max_iter)
                        
                        if results:
                            best_params, best_score, convergence = results
                            
                            # Actualizar el modelo con los mejores parámetros
                            updated_params = model_obj.params.copy()
                            for i, (param, _) in enumerate(selected_params.items()):
                                updated_params[param] = best_params[i]
                            
                            # Mostrar los resultados
                            display_metaheuristic_results(model_obj, real_data, best_params, best_score, convergence, list(selected_params.keys()), updated_params, data_type)
