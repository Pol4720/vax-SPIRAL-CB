import streamlit as st

def show_parameters(vaccine_model_obj, no_vaccine_model_obj):
    """
    Muestra y permite editar los parámetros y condiciones iniciales del modelo.
    """
    st.header("Model Parameters")
    st.markdown("Set the parameters and initial conditions below. Default values are pre-filled.")

    # --- Formulario para condiciones iniciales ---
    st.subheader("Initial Conditions")
    # Definir el orden correcto de los compartimentos
    ic_labels = ["Sh", "Eh", "Ih", "Rh", "Sv", "Iv", "Rv"]
    ic_current = vaccine_model_obj.initial_conditions.copy()
    has_vaccine = len(ic_current) == 9
    if has_vaccine:
        ic_labels.append("Vh")  # Vh antes de Bl
    # Mostrar inputs para cada compartimento excepto Bl
    cols = st.columns(len(ic_labels))
    ic_values = []
    for i, label in enumerate(ic_labels):
        # Si hay vacuna, el valor de Vh debe ser el penúltimo elemento de initial_conditions
        if label == "Vh" and has_vaccine:
            val = cols[i].number_input(
                f"Initial {label}", min_value=0.0, value=float(ic_current[8]), step=1.0, key=f"ic_{label}_param"
            )
        else:
            val = cols[i].number_input(
                f"Initial {label}", min_value=0.0, value=float(ic_current[i]), step=1.0, key=f"ic_{label}_param"
            )
        ic_values.append(val)
    # Calcular Bl* automáticamente
    params = vaccine_model_obj.params.copy()
    tau1 = params.get('τ1', 0.06)
    tau2 = params.get('τ2', 0.2)
    mu_b = params.get('μb', 0.05)
    Ih_star = ic_values[ic_labels.index("Ih")]
    Iv_star = ic_values[ic_labels.index("Iv")]
    Bl_star = (tau1 * Ih_star + tau2 * Iv_star) / mu_b if mu_b > 0 else 0.0
    st.info(f"Initial Bacteria (Bl) will be set to: {Bl_star:.2f} (computed as (τ1·Ih + τ2·Iv)/μb)")
    # Construir condiciones iniciales completas (Vh si existe, luego Bl)
    initial_conditions = ic_values + [Bl_star]

    # Mostrar nombre y descripción usando param_comments
    st.subheader("Model Parameters")
    param_comments = getattr(vaccine_model_obj, "param_comments", {})
    for key in params:
        comment = param_comments.get(key, "")
        label = f"{key} ({comment})" if comment else key
        params[key] = st.number_input(label, value=params[key])

    # Actualizar condiciones iniciales y parámetros en los modelos
    vaccine_model_obj.initial_conditions = initial_conditions.copy()
    no_vaccine_model_obj.initial_conditions = initial_conditions[:8]  # Sin vacuna
    vaccine_model_obj.params = params.copy()
    no_vaccine_model_obj.params = {k: v for k, v in params.items() if k in no_vaccine_model_obj.params}

    # --- Persistir en session_state ---
    st.session_state["params"] = params.copy()
    st.session_state["initial_conditions"] = initial_conditions.copy()

    return params

