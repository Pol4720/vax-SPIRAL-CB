import pandas as pd
import numpy as np
import streamlit as st
import io
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_data_ui():
    """
    Interfaz de usuario para cargar diferentes tipos de datos reales.
    
    Returns:
        tuple: (real_data, data_type)
    """
    st.subheader("Datos Reales")
    
    # Selección del tipo de datos
    data_type = st.selectbox(
        "Tipo de datos a ajustar:",
        ["casos_diarios", "casos_acumulados", "compartimentos", "condiciones_iniciales"],
        help="Seleccione el tipo de datos que utilizará para el ajuste"
    )
    
    # Opción para cargar los datos
    data_option = st.radio(
        "Seleccione la fuente de datos:", 
        ["Cargar archivo CSV", "Ingresar datos manualmente", "Usar datos de ejemplo"],
        key="data_source"
    )
    
    real_data = None
    
    if data_option == "Cargar archivo CSV":
        uploaded_file = st.file_uploader(f"Cargar archivo CSV con datos de {data_type}", type=["csv"])
        if uploaded_file is not None:
            try:
                real_data = pd.read_csv(uploaded_file)
                st.success("Archivo cargado correctamente")
                st.dataframe(real_data.head())
                
                # Verificar columnas requeridas
                if data_type == "casos_diarios" and 'day' not in real_data.columns or 'cases' not in real_data.columns:
                    st.warning("El archivo debe contener columnas 'day' y 'cases'")
                    real_data = None
                elif data_type == "casos_acumulados" and 'day' not in real_data.columns or 'cumulative_cases' not in real_data.columns:
                    st.warning("El archivo debe contener columnas 'day' y 'cumulative_cases'")
                    real_data = None
                elif data_type == "compartimentos" and 'day' not in real_data.columns:
                    st.warning("El archivo debe contener al menos la columna 'day' y columnas para los compartimentos")
                    real_data = None
                
            except Exception as e:
                st.error(f"Error al cargar el archivo: {e}")
    
    elif data_option == "Ingresar datos manualmente":
        if data_type == "casos_diarios":
            st.markdown("Ingrese los datos en formato de días y casos:")
            col1, col2 = st.columns(2)
            with col1:
                days_str = st.text_area("Días (separados por comas)", "0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360")
            with col2:
                cases_str = st.text_area("Casos (separados por comas)", "10, 15, 22, 30, 25, 18, 12, 8, 10, 14, 20, 24, 18")
            
            if st.button("Crear Dataset"):
                try:
                    days = [int(d.strip()) for d in days_str.split(",")]
                    cases = [float(c.strip()) for c in cases_str.split(",")]
                    if len(days) != len(cases):
                        st.error("La cantidad de días y casos debe ser igual")
                    else:
                        real_data = pd.DataFrame({"day": days, "cases": cases})
                        st.success("Datos creados correctamente")
                        st.dataframe(real_data)
                except Exception as e:
                    st.error(f"Error al crear los datos: {e}")
        
        elif data_type == "casos_acumulados":
            st.markdown("Ingrese los datos en formato de días y casos acumulados:")
            col1, col2 = st.columns(2)
            with col1:
                days_str = st.text_area("Días (separados por comas)", "0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360")
            with col2:
                cumulative_str = st.text_area("Casos acumulados (separados por comas)", "10, 25, 47, 77, 102, 120, 132, 140, 150, 164, 184, 208, 226")
            
            if st.button("Crear Dataset"):
                try:
                    days = [int(d.strip()) for d in days_str.split(",")]
                    cumulative = [float(c.strip()) for c in cumulative_str.split(",")]
                    if len(days) != len(cumulative):
                        st.error("La cantidad de días y casos acumulados debe ser igual")
                    else:
                        real_data = pd.DataFrame({"day": days, "cumulative_cases": cumulative})
                        st.success("Datos creados correctamente")
                        st.dataframe(real_data)
                except Exception as e:
                    st.error(f"Error al crear los datos: {e}")
        
        elif data_type == "compartimentos":
            st.markdown("Ingrese los datos para cada compartimento (un día específico):")
            day = st.number_input("Día de medición", min_value=0, value=0)
            
            # Formulario para los compartimentos
            with st.form("compartment_form"):
                st.write("Valores para cada compartimento:")
                Sh = st.number_input("Susceptibles Humanos (Sh)", min_value=0.0, value=270.0)
                Eh = st.number_input("Expuestos Humanos (Eh)", min_value=0.0, value=20.0)
                Ih = st.number_input("Infectados Humanos (Ih)", min_value=0.0, value=10.0)
                Rh = st.number_input("Recuperados Humanos (Rh)", min_value=0.0, value=0.0)
                Sv = st.number_input("Susceptibles Vectores (Sv)", min_value=0.0, value=510.0)
                Iv = st.number_input("Infectados Vectores (Iv)", min_value=0.0, value=10.0)
                Rv = st.number_input("Recuperados Vectores (Rv)", min_value=0.0, value=0.0)
                Bl = st.number_input("Bacterias en ambiente (Bl)", min_value=0.0, value=100.0)
                
                submit = st.form_submit_button("Crear Dataset")
                
                if submit:
                    real_data = pd.DataFrame({
                        "day": [day],
                        "Sh": [Sh],
                        "Eh": [Eh],
                        "Ih": [Ih],
                        "Rh": [Rh],
                        "Sv": [Sv],
                        "Iv": [Iv],
                        "Rv": [Rv],
                        "Bl": [Bl]
                    })
                    st.success("Datos creados correctamente")
                    st.dataframe(real_data)
        
        elif data_type == "condiciones_iniciales":
            st.markdown("Ingrese las condiciones iniciales del modelo:")
            
            # Formulario para las condiciones iniciales
            with st.form("initial_conditions_form"):
                st.write("Valores iniciales para cada compartimento:")
                Sh = st.number_input("Susceptibles Humanos (Sh)", min_value=0.0, value=270.0)
                Eh = st.number_input("Expuestos Humanos (Eh)", min_value=0.0, value=20.0)
                Ih = st.number_input("Infectados Humanos (Ih)", min_value=0.0, value=10.0)
                Rh = st.number_input("Recuperados Humanos (Rh)", min_value=0.0, value=0.0)
                Sv = st.number_input("Susceptibles Vectores (Sv)", min_value=0.0, value=510.0)
                Iv = st.number_input("Infectados Vectores (Iv)", min_value=0.0, value=10.0)
                Rv = st.number_input("Recuperados Vectores (Rv)", min_value=0.0, value=0.0)
                Bl = st.number_input("Bacterias en ambiente (Bl)", min_value=0.0, value=100.0)
                
                submit = st.form_submit_button("Usar como condiciones iniciales")
                
                if submit:
                    real_data = pd.DataFrame({
                        "day": [0],
                        "Sh": [Sh],
                        "Eh": [Eh],
                        "Ih": [Ih],
                        "Rh": [Rh],
                        "Sv": [Sv],
                        "Iv": [Iv],
                        "Rv": [Rv],
                        "Bl": [Bl]
                    })
                    st.success("Condiciones iniciales definidas")
                    st.dataframe(real_data)
    
    elif data_option == "Usar datos de ejemplo":
        if data_type == "casos_diarios":
            # Datos sintéticos que simulan un brote estacional de casos diarios
            days = np.arange(0, 365, 30)
            # Simulación de casos con patrón estacional y algo de ruido
            base_cases = 10 + 20 * np.sin(2 * np.pi * (days / 365 - 30/365))
            cases = base_cases * (1 + 0.2 * np.random.randn(len(days)))
            cases = np.maximum(cases, 5)  # Asegurar que no hay casos negativos
            real_data = pd.DataFrame({"day": days, "cases": cases})
            
        elif data_type == "casos_acumulados":
            # Datos sintéticos de casos acumulados
            days = np.arange(0, 365, 30)
            # Simulación de casos diarios
            base_cases = 10 + 20 * np.sin(2 * np.pi * (days / 365 - 30/365))
            cases = base_cases * (1 + 0.2 * np.random.randn(len(days)))
            cases = np.maximum(cases, 5)  # Asegurar que no hay casos negativos
            # Convertir a casos acumulados
            cumulative_cases = np.cumsum(cases)
            real_data = pd.DataFrame({"day": days, "cumulative_cases": cumulative_cases})
            
        elif data_type == "compartimentos":
            # Datos sintéticos para compartimentos en varios días
            days = np.arange(0, 365, 30)
            n_days = len(days)
            
            # Crear tendencias básicas para cada compartimento
            Sh = 270 - np.linspace(0, 50, n_days) + 5 * np.random.randn(n_days)
            Eh = 20 + 10 * np.sin(np.linspace(0, 2*np.pi, n_days)) + 2 * np.random.randn(n_days)
            Ih = 10 + 15 * np.sin(np.linspace(0, 2*np.pi, n_days) + np.pi/4) + 2 * np.random.randn(n_days)
            Rh = np.linspace(0, 50, n_days) + 5 * np.random.randn(n_days)
            
            Sv = 510 - 20 * np.sin(np.linspace(0, 2*np.pi, n_days)) + 10 * np.random.randn(n_days)
            Iv = 10 + 8 * np.sin(np.linspace(0, 2*np.pi, n_days) + np.pi/3) + 2 * np.random.randn(n_days)
            Rv = np.linspace(0, 20, n_days) + 3 * np.random.randn(n_days)
            
            Bl = 100 + 50 * np.sin(np.linspace(0, 2*np.pi, n_days) + np.pi/6) + 10 * np.random.randn(n_days)
            
            # Asegurar valores no negativos
            Sh = np.maximum(Sh, 0)
            Eh = np.maximum(Eh, 0)
            Ih = np.maximum(Ih, 0)
            Rh = np.maximum(Rh, 0)
            Sv = np.maximum(Sv, 0)
            Iv = np.maximum(Iv, 0)
            Rv = np.maximum(Rv, 0)
            Bl = np.maximum(Bl, 0)
            
            real_data = pd.DataFrame({
                "day": days,
                "Sh": Sh,
                "Eh": Eh,
                "Ih": Ih,
                "Rh": Rh,
                "Sv": Sv,
                "Iv": Iv,
                "Rv": Rv,
                "Bl": Bl
            })
            
        elif data_type == "condiciones_iniciales":
            # Condiciones iniciales de ejemplo (un solo punto)
            real_data = pd.DataFrame({
                "day": [0],
                "Sh": [270],
                "Eh": [20],
                "Ih": [10],
                "Rh": [0],
                "Sv": [510],
                "Iv": [10],
                "Rv": [0],
                "Bl": [100]
            })
        
        st.success(f"Datos de ejemplo de {data_type} cargados")
        st.dataframe(real_data)
    
    # Visualizar los datos si están disponibles
    if real_data is not None:
        visualize_data(real_data, data_type)
    
    return real_data, data_type

def visualize_data(data, data_type):
    """
    Visualiza los datos cargados según su tipo.
    
    Args:
        data: DataFrame con los datos
        data_type: Tipo de datos
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if data_type == "casos_diarios":
        ax.scatter(data['day'], data['cases'], color='red', label='Casos diarios')
        ax.set_ylabel("Casos")
        ax.set_title("Datos de casos diarios")
    
    elif data_type == "casos_acumulados":
        ax.scatter(data['day'], data['cumulative_cases'], color='blue', label='Casos acumulados')
        ax.set_ylabel("Casos acumulados")
        ax.set_title("Datos de casos acumulados")
    
    elif data_type == "compartimentos" or data_type == "condiciones_iniciales":
        compartments = [col for col in data.columns if col != 'day']
        for comp in compartments:
            ax.plot(data['day'], data[comp], 'o-', label=comp)
        
        ax.set_ylabel("Población")
        if data_type == "compartimentos":
            ax.set_title("Datos de compartimentos a lo largo del tiempo")
        else:
            ax.set_title("Condiciones iniciales")
    
    ax.set_xlabel("Días")
    ax.grid(True)
    ax.legend()
    
    st.pyplot(fig)

def prepare_data_for_fitting(data, data_type, model_obj):
    """
    Prepara los datos para el ajuste según su tipo.
    
    Args:
        data: DataFrame con los datos
        data_type: Tipo de datos
        model_obj: Objeto del modelo
        
    Returns:
        DataFrame: Datos preparados para el ajuste
    """
    if data_type == "condiciones_iniciales":
        # Si son condiciones iniciales, actualizar el modelo directamente
        if 'day' in data.columns and data['day'].iloc[0] == 0:
            ic = []
            for comp in ['Sh', 'Eh', 'Ih', 'Rh', 'Sv', 'Iv', 'Rv', 'Bl']:
                if comp in data.columns:
                    ic.append(data[comp].iloc[0])
                else:
                    # Usar el valor por defecto del modelo
                    idx = model_obj.initial_conditions.index(comp)
                    ic.append(model_obj.initial_conditions[idx])
            
            # Actualizar condiciones iniciales del modelo
            model_obj.initial_conditions = ic
            st.success("Condiciones iniciales actualizadas en el modelo")
    
    # Para otros tipos de datos, asegurarse de que tienen el formato correcto
    if data_type == "casos_diarios" and 'cases' not in data.columns:
        st.error("Los datos deben contener una columna 'cases'")
        return None
    
    if data_type == "casos_acumulados" and 'cumulative_cases' not in data.columns:
        st.error("Los datos deben contener una columna 'cumulative_cases'")
        return None
    
    if data_type == "compartimentos":
        required_columns = ['Sh', 'Eh', 'Ih', 'Rh', 'Sv', 'Iv', 'Rv', 'Bl']
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            st.warning(f"Faltan las siguientes columnas: {', '.join(missing)}")
            # Se podría completar con valores por defecto o NA
    
    return data
