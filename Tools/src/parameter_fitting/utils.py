import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def create_model_copy(model_obj, params=None):
    """
    Crea una copia del modelo con parámetros actualizados.
    
    Args:
        model_obj: Modelo original
        params: Nuevos parámetros (opcional)
        
    Returns:
        Copia del modelo con parámetros actualizados
    """
    model_copy = type(model_obj)()
    if params:
        model_copy.params = params
    model_copy.initial_conditions = model_obj.initial_conditions.copy()
    return model_copy

def evaluate_model(model_obj, real_data, data_type):
    """
    Evalúa el modelo según el tipo de datos y devuelve los valores predichos.
    
    Args:
        model_obj: Modelo de leptospirosis
        real_data: DataFrame con datos reales
        data_type: Tipo de datos
        
    Returns:
        ndarray: Valores predichos por el modelo
    """
    # Evaluar en los días de los datos reales
    t_span = (0, max(real_data['day']))
    t_eval = real_data['day'].values
    
    sol = solve_ivp(
        lambda t, y: model_obj.model(t, y),
        t_span=t_span,
        y0=model_obj.initial_conditions,
        t_eval=t_eval,
        method='RK45'
    )
    
    # Extraer predicciones según el tipo de datos
    if data_type == "casos_diarios":
        # Devolver compartimento Ih (infectados humanos)
        return sol.y[2]
    
    elif data_type == "casos_acumulados":
        # Calcular casos acumulados
        # Nota: esto es una aproximación, para casos acumulados reales
        # se debería integrar la tasa de infección a lo largo del tiempo
        return np.cumsum(sol.y[2])
    
    elif data_type == "compartimentos":
        # Devolver todos los compartimentos
        compartments = {
            'Sh': 0, 'Eh': 1, 'Ih': 2, 'Rh': 3,
            'Sv': 4, 'Iv': 5, 'Rv': 6, 'Bl': 7
        }
        
        # Extraer los compartimentos presentes en los datos reales
        columns = [col for col in real_data.columns if col in compartments]
        
        if not columns:
            raise ValueError("No se encontraron compartimentos válidos en los datos")
        
        # Extraer los valores de los compartimentos correspondientes
        values = np.array([sol.y[compartments[col]] for col in columns])
        
        return values
    
    else:
        # Por defecto, devolver el compartimento Ih
        return sol.y[2]

def calculate_metrics(model_obj, real_data, updated_params, data_type):
    """
    Calcula métricas de error entre los datos reales y las predicciones del modelo.
    
    Args:
        model_obj: Modelo de leptospirosis
        real_data: DataFrame con datos reales
        updated_params: Parámetros actualizados
        data_type: Tipo de datos
        
    Returns:
        dict: Diccionario con métricas (MSE, RMSE, MAE)
    """
    # Crear una copia del modelo con los parámetros actualizados
    model_copy = create_model_copy(model_obj, updated_params)
    
    # Evaluar el modelo
    model_predictions = evaluate_model(model_copy, real_data, data_type)
    
    # Extraer los datos observados según el tipo
    if data_type == "casos_diarios":
        observed = real_data['cases'].values
    elif data_type == "casos_acumulados":
        observed = real_data['cumulative_cases'].values
    elif data_type == "compartimentos":
        # Extraer los compartimentos que coinciden con los predichos
        compartments = {
            'Sh': 0, 'Eh': 1, 'Ih': 2, 'Rh': 3,
            'Sv': 4, 'Iv': 5, 'Rv': 6, 'Bl': 7
        }
        columns = [col for col in real_data.columns if col in compartments]
        observed = real_data[columns].values.flatten()
        model_predictions = model_predictions.flatten()
    else:
        # Por defecto, usar casos diarios
        observed = real_data['cases'].values
    
    # Calcular métricas
    mse = np.mean((observed - model_predictions)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(observed - model_predictions))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }

def get_model_parameter_bounds(model_obj, param_type="all"):
    """
    Devuelve límites razonables para los parámetros del modelo.
    
    Args:
        model_obj: Modelo de leptospirosis
        param_type: Tipo de parámetros ("all", "beta", "rates", etc.)
        
    Returns:
        dict: Diccionario de parámetros con sus límites
    """
    # Límites por defecto para diferentes tipos de parámetros
    param_bounds = {
        # Parámetros beta (transmisión)
        'β1': {'min': 0.00001, 'max': 0.001, 'default': model_obj.params['β1']},
        'β2': {'min': 0.01, 'max': 0.5, 'default': model_obj.params['β2']},
        'β3': {'min': 0.0001, 'max': 0.01, 'default': model_obj.params['β3']},
        
        # Tasas de recuperación/muerte
        'γ': {'min': 0.01, 'max': 0.2, 'default': model_obj.params['γ']},
        'μ': {'min': 0.0001, 'max': 0.01, 'default': model_obj.params['μ']},
        'μv': {'min': 0.0001, 'max': 0.01, 'default': model_obj.params['μv']},
        'μb': {'min': 0.01, 'max': 0.2, 'default': model_obj.params['μb']},
        'θ': {'min': 0.01, 'max': 0.2, 'default': model_obj.params['θ']},
        'α': {'min': 0.01, 'max': 0.1, 'default': model_obj.params['α']},
        'δ': {'min': 0.01, 'max': 0.2, 'default': model_obj.params['δ']},
        'ρ': {'min': 0.01, 'max': 0.2, 'default': model_obj.params['ρ']},
        'σ': {'min': 0.01, 'max': 0.2, 'default': model_obj.params['σ']},
        
        # Otros parámetros
        'Λ': {'min': 1, 'max': 10, 'default': model_obj.params['Λ']},
        'Π': {'min': 0.5, 'max': 5, 'default': model_obj.params['Π']},
        'κ': {'min': 1000, 'max': 50000, 'default': model_obj.params['κ']},
        'τ1': {'min': 0.01, 'max': 0.2, 'default': model_obj.params['τ1']},
        'τ2': {'min': 0.05, 'max': 0.5, 'default': model_obj.params['τ2']}
    }
    
    # Filtrar según el tipo solicitado
    if param_type == "beta":
        return {k: v for k, v in param_bounds.items() if k in ['β1', 'β2', 'β3']}
    elif param_type == "rates":
        return {k: v for k, v in param_bounds.items() if k in ['γ', 'μ', 'μv', 'μb', 'θ', 'α', 'δ', 'ρ', 'σ']}
    elif param_type == "recruitment":
        return {k: v for k, v in param_bounds.items() if k in ['Λ', 'Π']}
    elif param_type == "pathogen":
        return {k: v for k, v in param_bounds.items() if k in ['κ', 'τ1', 'τ2']}
    else:
        return param_bounds
