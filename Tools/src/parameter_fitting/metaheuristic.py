import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import differential_evolution, minimize, basinhopping
from .utils import create_model_copy, evaluate_model, calculate_metrics

def run_metaheuristic(model_obj, real_data, selected_params, optimizer, data_type, max_iter=50):
    """
    Ejecuta algoritmos metaheurísticos para ajustar los parámetros.
    
    Args:
        model_obj: Modelo de leptospirosis
        real_data: DataFrame con datos reales
        selected_params: Diccionario de parámetros a ajustar con sus límites
        optimizer: Algoritmo a utilizar
        data_type: Tipo de datos (casos, compartimentos, etc.)
        max_iter: Máximo de iteraciones
        
    Returns:
        tuple: (best_params, best_score, convergence)
    """
    param_names = list(selected_params.keys())
    bounds = [selected_params[param] for param in param_names]
    
    # Convergencia
    convergence = []
    
    # Función objetivo adaptada según el tipo de datos
    def objective_function(theta):
        # Actualizar parámetros del modelo
        params = model_obj.params.copy()
        for i, param in enumerate(param_names):
            params[param] = theta[i]
        
        # Ejecutar modelo con los parámetros actualizados
        model_copy = create_model_copy(model_obj, params)
        
        try:
            # Evaluar el modelo según el tipo de datos
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
            
            # Calcular MSE
            mse = np.mean((observed - predicted)**2)
            convergence.append(mse)
            return mse
        except Exception as e:
            # Penalizar fuertemente si hay error en la integración
            return 1e10
    
    # Punto inicial
    x0 = np.array([model_obj.params[param] for param in param_names])
    
    # Ejecutar optimización según el algoritmo elegido
    if optimizer == "Evolución Diferencial":
        result = differential_evolution(
            objective_function, 
            bounds, 
            maxiter=max_iter, 
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=True
        )
    elif optimizer == "Basin-Hopping":
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        result = basinhopping(
            objective_function,
            x0,
            niter=max_iter,
            minimizer_kwargs=minimizer_kwargs,
            stepsize=0.1
        )
    elif optimizer == "Nelder-Mead":
        result = minimize(
            objective_function,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": max_iter, "disp": True}
        )
    
    return result.x, result.fun, convergence

def display_metaheuristic_results(model_obj, real_data, best_params, best_score, convergence, param_names, updated_params, data_type):
    """
    Muestra los resultados del ajuste metaheurístico.
    
    Args:
        model_obj: Modelo de leptospirosis
        real_data: DataFrame con datos reales
        best_params: Mejores parámetros encontrados
        best_score: Mejor puntuación (MSE)
        convergence: Lista de valores de convergencia
        param_names: Nombres de los parámetros ajustados
        updated_params: Parámetros actualizados del modelo
        data_type: Tipo de datos utilizados para el ajuste
    """
    st.subheader("Resultados de la Optimización")
    
    # Tabla de mejores parámetros
    results_df = pd.DataFrame({
        "Parámetro": param_names,
        "Valor Original": [model_obj.params[p] for p in param_names],
        "Valor Ajustado": best_params,
        "Diferencia (%)": [(best_params[i] - model_obj.params[p]) / model_obj.params[p] * 100 for i, p in enumerate(param_names)]
    })
    
    st.write("Mejores parámetros encontrados:")
    st.dataframe(results_df)
    
    # Gráfico de convergencia
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(convergence, 'o-', alpha=0.7)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Error Cuadrático Medio (MSE)")
    ax.set_title("Convergencia del algoritmo")
    ax.grid(True)
    st.pyplot(fig)
    
    st.write(f"Mejor MSE encontrado: {best_score:.6f}")
    
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
