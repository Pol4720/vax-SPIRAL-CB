import streamlit as st

def show_parameters(vaccine_model_obj, no_vaccine_model_obj):
    """
    Muestra y permite editar los parámetros del modelo.
    
    Args:
        vaccine_model_obj: Instancia del modelo de vacunación
        no_vaccine_model_obj: Instancia del modelo sin vacunación
    
    Returns:
        dict: Parámetros actualizados
    """
    st.header("Model Parameters")
    st.markdown("Set the parameters below. Default values are pre-filled.")

    params = vaccine_model_obj.params.copy()
    
    # Mostrar nombre y descripción usando param_comments
    for key in params:
        comment = vaccine_model_obj.param_comments.get(key, "")
        label = f"{key} ({comment})" if comment else key
        if isinstance(params[key], float):
            params[key] = st.number_input(label, value=params[key])

    # Actualizar parámetros en los modelos
    vaccine_model_obj.params = params.copy()
    no_vaccine_model_obj.params = {k: v for k, v in params.items() if k in no_vaccine_model_obj.params}
    
    return params
