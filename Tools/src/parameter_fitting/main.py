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
    
    # Inicializar session state para datos si no existe
    if 'fitted_data' not in st.session_state:
        st.session_state.fitted_data = None
        st.session_state.fitted_data_type = None
        st.session_state.fitted_prepared_data = None
    
    # Botón para recargar datos
    if st.session_state.fitted_data is not None:
        if st.button("Cargar nuevos datos"):
            st.session_state.fitted_data = None
            st.session_state.fitted_data_type = None
            st.session_state.fitted_prepared_data = None
            st.experimental_rerun()
    
    # Cargar datos solo si no están ya cargados
    if st.session_state.fitted_data is None:
        real_data, data_type = load_data_ui()
        if real_data is not None:
            st.session_state.fitted_data = real_data
            st.session_state.fitted_data_type = data_type
            prepared_data = prepare_data_for_fitting(real_data, data_type, model_obj)
            st.session_state.fitted_prepared_data = prepared_data
    else:
        real_data = st.session_state.fitted_data
        data_type = st.session_state.fitted_data_type
        prepared_data = st.session_state.fitted_prepared_data
    
    # Continuar solo si tenemos datos
    if real_data is not None:
        # Solo necesitamos verificar prepared_data si no es condiciones_iniciales
        if prepared_data is not None or data_type == "condiciones_iniciales":
            st.subheader("Selección de parámetros a ajustar o fijar")

            param_bounds = get_model_parameter_bounds(model_obj, "all")
            param_names = list(param_bounds.keys())

            # Inicializar session state para los parámetros si no existe
            if 'param_choices' not in st.session_state:
                st.session_state.param_choices = {}
                
            # Inicializar valores para cada parámetro si no existen
            for param in param_names:
                if param not in st.session_state.param_choices:
                    st.session_state.param_choices[param] = {
                        'mode': 'Ajustar',
                        'fixed_value': float(model_obj.params[param]),
                        'min_value': param_bounds[param]['min'],
                        'max_value': param_bounds[param]['max']
                    }

            st.write("Seleccione para cada parámetro si desea fijarlo o ajustarlo:")

            fixed_params = {}
            adjustable_params = {}

            # Crear un contenedor para todos los parámetros
            params_container = st.container()
            
            with params_container:
                for i, param in enumerate(param_names):
                    # Añadir línea separadora entre parámetros
                    if i > 0:
                        st.markdown("<hr style='margin: 15px 0; border: 0; height: 1px; background-color: #e0e0e0;'>", unsafe_allow_html=True)
                    
                    st.markdown(f"**Parámetro: {param}**")
                    col1, col2 = st.columns([1, 3])
                    
                    # Función para actualizar el modo del parámetro
                    def create_mode_callback(param_name):
                        def callback():
                            current_mode = st.session_state[f"fix_{param_name}"]
                            st.session_state.param_choices[param_name]['mode'] = current_mode
                        return callback
                    
                    with col1:
                        mode = st.radio(
                            f"",
                            ["Ajustar", "Fijar"],
                            key=f"fix_{param}",
                            index=0 if st.session_state.param_choices[param]['mode'] == 'Ajustar' else 1,
                            on_change=create_mode_callback(param)
                        )
                    
                    if mode == "Fijar":
                        with col2:
                            # Función para actualizar el valor fijo
                            def create_value_callback(param_name):
                                def callback():
                                    st.session_state.param_choices[param_name]['fixed_value'] = st.session_state[f"val_{param_name}"]
                                return callback
                            
                            val = st.number_input(
                                f"Valor fijo para {param}:", 
                                value=st.session_state.param_choices[param]['fixed_value'],
                                format="%.6f", 
                                key=f"val_{param}",
                                on_change=create_value_callback(param)
                            )
                        fixed_params[param] = val
                    else:  # Ajustar
                        with col2:
                            min_col, max_col = st.columns(2)
                            
                            # Funciones para actualizar min/max
                            def create_min_callback(param_name):
                                def callback():
                                    st.session_state.param_choices[param_name]['min_value'] = st.session_state[f"min_{param_name}"]
                                return callback
                            
                            def create_max_callback(param_name):
                                def callback():
                                    st.session_state.param_choices[param_name]['max_value'] = st.session_state[f"max_{param_name}"]
                                return callback
                            
                            with min_col:
                                min_val = st.number_input(
                                    f"Valor mínimo:", 
                                    value=st.session_state.param_choices[param]['min_value'], 
                                    format="%.6f", 
                                    key=f"min_{param}",
                                    on_change=create_min_callback(param)
                                )
                            
                            with max_col:
                                max_val = st.number_input(
                                    f"Valor máximo:", 
                                    value=st.session_state.param_choices[param]['max_value'], 
                                    format="%.6f", 
                                    key=f"max_{param}",
                                    on_change=create_max_callback(param)
                                )
                        
                        adjustable_params[param] = (min_val, max_val)
            
            st.markdown("<hr style='margin: 30px 0; border: 0; height: 2px; background-color: #bbb;'>", unsafe_allow_html=True)

            if not adjustable_params:
                st.warning("Por favor, seleccione al menos un parámetro para ajustar.")
                return

            st.markdown("---")
            st.subheader("Método de Ajuste")
            method = st.radio(
                "Seleccione el método de ajuste:",
                ["MCMC (Markov Chain Monte Carlo)", "Optimización Metaheurística"],
                key="fitting_method"
            )

            if method == "MCMC (Markov Chain Monte Carlo)":
                st.subheader("Configuración de MCMC")
                n_walkers = st.slider("Número de caminantes", 10, 100, 20)
                n_steps = st.slider("Número de pasos", 100, 2000, 500)
                burn_in = st.slider("Burn-in (pasos a descartar)", 50, 500, 100)
                if st.button("Ejecutar MCMC"):
                    with st.spinner("Ejecutando MCMC..."):
                        results = run_mcmc(model_obj, real_data, adjustable_params, fixed_params, data_type, n_walkers, n_steps, burn_in)
                        if results:
                            samples, param_names_adj, log_prob_samples, best_params = results
                            updated_params = model_obj.params.copy()
                            # Actualizar los parámetros ajustados
                            for i, param in enumerate(param_names_adj):
                                updated_params[param] = best_params[i]
                            # Actualizar los parámetros fijados
                            for param, val in fixed_params.items():
                                updated_params[param] = val
                            display_mcmc_results(model_obj, real_data, samples, param_names_adj, log_prob_samples, best_params, updated_params, data_type, fixed_params)

            elif method == "Optimización Metaheurística":
                st.subheader("Configuración de Metaheurísticas")
                optimizer = st.selectbox(
                    "Seleccione el algoritmo:",
                    ["Evolución Diferencial", "Basin-Hopping", "Nelder-Mead"]
                )
                max_iter = st.slider("Máximo de iteraciones", 10, 200, 50)
                if st.button("Ejecutar Optimización"):
                    with st.spinner(f"Ejecutando {optimizer}..."):
                        results = run_metaheuristic(model_obj, real_data, adjustable_params, fixed_params, optimizer, data_type, max_iter)
                        if results:
                            best_params, best_score, convergence = results
                            updated_params = model_obj.params.copy()
                            # Actualizar los parámetros ajustados
                            for i, param in enumerate(adjustable_params.keys()):
                                updated_params[param] = best_params[i]
                            # Actualizar los parámetros fijados
                            for param, val in fixed_params.items():
                                updated_params[param] = val
                            display_metaheuristic_results(model_obj, real_data, best_params, best_score, convergence, list(adjustable_params.keys()), updated_params, data_type, fixed_params)
