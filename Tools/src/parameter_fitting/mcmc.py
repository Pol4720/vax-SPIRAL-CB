import numpy as np
import streamlit as st
import emcee
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import corner
from .utils import create_model_copy, evaluate_model, calculate_metrics

def run_mcmc(model_obj, real_data, selected_params, data_type, n_walkers=20, n_steps=500, burn_in=100):
    """
    Ejecuta el algoritmo MCMC para ajustar los parámetros.
    
    Args:
        model_obj: Modelo de leptospirosis
        real_data: DataFrame con datos reales
        selected_params: Diccionario de parámetros a ajustar con sus límites
        data_type: Tipo de datos (casos, compartimentos, etc.)
        n_walkers: Número de walkers
        n_steps: Número de pasos
        burn_in: Número de pasos a descartar al inicio
        
    Returns:
        tuple: (samples, param_names, log_prob_samples, best_params)
    """
    param_names = list(selected_params.keys())
    ndim = len(param_names)
    
    # Función de verosimilitud dependiente del tipo de datos
    def log_likelihood(theta):
        # Actualizar parámetros del modelo
        params = model_obj.params.copy()
        for i, param in enumerate(param_names):
            params[param] = theta[i]
        
        # Ejecutar modelo con los parámetros actualizados
        model_copy = create_model_copy(model_obj, params)
        
        # Evaluar el modelo según el tipo de datos
        try:
            model_data = evaluate_model(model_copy, real_data, data_type)
            
            # Extraer los datos observados según el tipo
            if data_type == "casos_diarios":
                observed = real_data['cases'].values
                predicted = model_data
            elif data_type == "casos_acumulados":
                observed = real_data['cumulative_cases'].values
                predicted = model_data
            elif data_type == "compartimentos":
                # Aquí asumimos que real_data tiene columnas para cada compartimento
                observed_columns = [col for col in real_data.columns if col not in ['day', 'time']]
                observed = real_data[observed_columns].values.flatten()
                predicted = model_data.flatten()
            else:
                # Por defecto, asumimos casos diarios
                observed = real_data['cases'].values
                predicted = model_data
            
            # Calcular log-likelihood (suponiendo error gaussiano)
            sigma = np.std(observed - predicted) or 1.0
            return -0.5 * np.sum((observed - predicted)**2 / sigma**2 + np.log(2 * np.pi * sigma**2))
        except Exception as e:
            # Penalizar fuertemente si hay error en la integración
            return -np.inf
    
    # Prior uniforme
    def log_prior(theta):
        for i, param in enumerate(param_names):
            if not (selected_params[param][0] <= theta[i] <= selected_params[param][1]):
                return -np.inf
        return 0.0
    
    # Probabilidad posterior
    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)
    
    # Posición inicial para los walkers
    initial_positions = []
    for param in param_names:
        min_val, max_val = selected_params[param]
        initial_positions.append(np.random.uniform(min_val, max_val, n_walkers))
    
    pos = np.array(initial_positions).T
    
    # Ejecutar MCMC
    progress_bar = st.progress(0)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability)
    
    # Track progress
    for i, _ in enumerate(sampler.sample(pos, iterations=n_steps)):
        progress_bar.progress((i + 1) / n_steps)
    
    # Obtener muestras (descartando burn-in)
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_prob_samples = sampler.get_log_prob(discard=burn_in, flat=True)
    
    # Encontrar los mejores parámetros (máxima probabilidad posterior)
    best_idx = np.argmax(log_prob_samples)
    best_params = samples[best_idx]
    
    return samples, param_names, log_prob_samples, best_params

def display_mcmc_results(model_obj, real_data, samples, param_names, log_prob_samples, best_params, updated_params, data_type):
    """
    Muestra los resultados del ajuste MCMC.
    
    Args:
        model_obj: Modelo de leptospirosis
        real_data: DataFrame con datos reales
        samples: Muestras del MCMC
        param_names: Nombres de los parámetros ajustados
        log_prob_samples: Log-probabilidades de las muestras
        best_params: Mejores parámetros encontrados
        updated_params: Parámetros actualizados del modelo
        data_type: Tipo de datos utilizados para el ajuste
    """
    st.subheader("Resultados del MCMC")
    
    # Tabla de mejores parámetros
    results_df = pd.DataFrame({
        "Parámetro": param_names,
        "Valor Original": [model_obj.params[p] for p in param_names],
        "Valor Ajustado": best_params,
        "Diferencia (%)": [(best_params[i] - model_obj.params[p]) / model_obj.params[p] * 100 for i, p in enumerate(param_names)]
    })
    
    st.write("Mejores parámetros encontrados:")
    st.dataframe(results_df)
    
    # Visualización de la distribución posterior
    if len(param_names) > 1:
        try:
            fig = corner.corner(
                samples, 
                labels=param_names,
                truths=best_params,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12}
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error al generar gráfico de corner: {e}")
    
    # Visualización de las cadenas de Markov
    for i, param in enumerate(param_names):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(samples[:, i], bins=50, alpha=0.7)
        ax.axvline(best_params[i], color="red", linestyle="--", label="Mejor valor")
        ax.set_xlabel(param)
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribución posterior de {param}")
        ax.legend()
        st.pyplot(fig)
    
    # Calcular y mostrar las métricas de ajuste
    metrics = calculate_metrics(model_obj, real_data, updated_params, data_type)
    
    # Mostrar métricas
    st.subheader("Métricas de ajuste")
    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{metrics['mse']:.4f}")
    col2.metric("RMSE", f"{metrics['rmse']:.4f}")
    col3.metric("MAE", f"{metrics['mae']:.4f}")
    
    # Visualizar la comparación
    st.subheader("Comparación con datos reales")
    
    # Configurar el modelo con los parámetros actualizados
    model_copy = create_model_copy(model_obj, updated_params)
    
    # Generar la simulación para comparar
    from .visualization import plot_comparison
    plot_comparison(model_copy, real_data, data_type)
