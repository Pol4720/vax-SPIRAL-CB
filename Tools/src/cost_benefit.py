import streamlit as st
import numpy as np
from scipy.integrate import simpson
import sys
import os


from cba_extended import CostParams, BenefitParams, summarize_cba

def analyze_cost_benefit(vaccine_model_obj, params, initial_conditions):
    """
    Realiza el análisis de costo-beneficio de la vacunación.
    
    Args:
        vaccine_model_obj: Instancia del modelo de vacunación
        params: Parámetros del modelo
        initial_conditions: Condiciones iniciales
    """
    st.header("Cost-Benefit Analysis")
    
    # Crear pestañas para separar el análisis simple del extendido
    tab1, tab2 = st.tabs(["Análisis Básico", "Análisis Extendido"])
    
    with tab1:
        st.subheader("Análisis Costo-Beneficio Básico")
        
        dose_cost = st.number_input("Cost per Vaccine Dose ($)", 0.0, 500.0, 15.0)
        doses_per_person = st.number_input("Doses per Person", 1, 3, 1)
        prop_mild = st.slider("Proportion Mild", 0.0, 1.0, 0.85)
        prop_mod = st.slider("Proportion Moderate", 0.0, 1.0, 0.05)
        prop_sev = st.slider("Proportion Severe", 0.0, 1.0, 0.10)
        cost_mild = st.number_input("Cost Mild Case ($)", 0.0, 1000.0, 159.0)
        cost_mod = st.number_input("Cost Moderate Case ($)", 0.0, 5000.0, 1996.0)
        cost_sev = st.number_input("Cost Severe Case ($)", 0.0, 40000.0, 33260.0)

        vaccine_model_obj.params = params.copy()
        vaccine_model_obj.initial_conditions = initial_conditions.copy()
        sol_vax = vaccine_model_obj.solve(with_vaccine=True)
        sol_no_vax = vaccine_model_obj.solve(with_vaccine=False)
        t_eval = vaccine_model_obj.t_eval

        # Calcular el número total de personas que han pasado por el compartimento de infectados
        theta = params['θ']
        total_infected_no_vax = simpson(theta * sol_no_vax.y[1], t_eval)
        total_infected_vax = simpson(theta * sol_vax.y[1], t_eval)
        avoided_cases = total_infected_no_vax - total_infected_vax

        cases_mild = avoided_cases * prop_mild
        cases_mod = avoided_cases * prop_mod
        cases_sev = avoided_cases * prop_sev

        # Calcular la cantidad total de vacunados
        Ih = sol_vax.y[2]  # Compartimento de infectados humanos
        susceptibles = sol_vax.y[0]  # S(t) del modelo con vacuna

        # Calcular la tasa de vacunación dinámica en cada punto de tiempo
        nu_dyn = np.array([
            vaccine_model_obj.nu_dinamica(Ih[i], t_eval[i])
            for i in range(len(t_eval))
        ])
        total_vaccinated = simpson(nu_dyn * susceptibles, t_eval)
        costs = dose_cost * doses_per_person * total_vaccinated
        savings = (cases_mild * cost_mild) + (cases_mod * cost_mod) + (cases_sev * cost_sev)
        net_benefit = savings - costs

        st.markdown(f"**Avoided Infections**: {avoided_cases:.0f}")
        st.markdown(f"**Healthcare Savings**: ${savings:,.2f}")
        st.markdown(f"**Vaccination Cost**: ${costs:,.2f}")
        st.markdown(f"**Net Benefit**: ${net_benefit:,.2f}")
        
        # Calcular el coeficiente de costo-beneficio
        if costs > 0:
            cost_benefit_ratio = savings / costs
        else:
            cost_benefit_ratio = float('inf')

        st.markdown(f"**Cost-Benefit Ratio**: {cost_benefit_ratio:.2f}")

        # Determinar si es rentable
        if cost_benefit_ratio > 1:
            st.success("Vaccination is cost-effective (savings exceed costs).")
        else:
            st.warning("Vaccination is NOT cost-effective (costs exceed savings).")
    
    with tab2:
        st.subheader("Análisis Costo-Beneficio Extendido")
        
        # Obtener la población objetivo y casos evitados del modelo epidemiológico
        vaccine_model_obj.params = params.copy()
        vaccine_model_obj.initial_conditions = initial_conditions.copy()
        sol_vax = vaccine_model_obj.solve(with_vaccine=True)
        sol_no_vax = vaccine_model_obj.solve(with_vaccine=False)
        t_eval = vaccine_model_obj.t_eval
        
        # Calcular casos evitados
        theta = params['θ']
        total_infected_no_vax = simpson(theta * sol_no_vax.y[1], t_eval)
        total_infected_vax = simpson(theta * sol_vax.y[1], t_eval)
        avoided_cases = total_infected_no_vax - total_infected_vax
        
        # Calcular población vacunada
        Ih = sol_vax.y[2]  # Compartimento de infectados humanos
        susceptibles = sol_vax.y[0]  # S(t) del modelo con vacuna
        nu_dyn = np.array([
            vaccine_model_obj.nu_dinamica(Ih[i], t_eval[i])
            for i in range(len(t_eval))
        ])
        total_vaccinated = simpson(nu_dyn * susceptibles, t_eval)
        
        # Configuración de parámetros extendidos
        st.markdown("### Población y Esquema")
        population_target = st.number_input("Población objetivo", min_value=1, value=int(total_vaccinated))
        coverage = st.slider("Cobertura (%)", 0.0, 100.0, 60.0) / 100.0
        doses_per_person_ext = st.number_input("Dosis por persona", min_value=1, max_value=5, value=2)
        wastage_rate = st.slider("Tasa de merma (%)", 0.0, 20.0, 5.0) / 100.0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Costos de Vacuna y Suministros")
            vaccine_price = st.number_input("Precio por dosis ($)", 0.0, 500.0, 10.0)
            freight_ins = st.slider("Flete y seguro (%)", 0.0, 20.0, 5.0) / 100.0
            syringe_price = st.number_input("Precio jeringa/aguja ($)", 0.0, 5.0, 0.05)
            safety_box_price = st.number_input("Cajas de seguridad por 100 dosis ($)", 0.0, 20.0, 5.0)
            
            st.markdown("### Costos Variables Operativos")
            admin_time = st.number_input("Tiempo administración por dosis (min)", 0.0, 30.0, 5.0)
            staff_cost = st.number_input("Costo de personal por hora ($)", 0.0, 50.0, 8.0)
            cold_chain_var = st.number_input("Cadena de frío por dosis ($)", 0.0, 1.0, 0.03)
            transport_var = st.number_input("Transporte por dosis ($)", 0.0, 1.0, 0.02)
            waste_mgmt = st.number_input("Manejo de residuos por dosis ($)", 0.0, 1.0, 0.01)
            
        with col2:
            st.markdown("### Costos Fijos")
            clinic_overhead = st.number_input("Gastos vacunatorios fijos ($)", 0.0, 10000.0, 2000.0)
            training_fixed = st.number_input("Capacitación ($)", 0.0, 5000.0, 1000.0)
            communication_fixed = st.number_input("Comunicación/difusión ($)", 0.0, 5000.0, 1500.0)
            monitoring_fixed = st.number_input("Monitoreo/IT ($)", 0.0, 5000.0, 800.0)
            other_fixed = st.number_input("Otros costos fijos ($)", 0.0, 5000.0, 0.0)
            
            st.markdown("### Eventos Adversos (AEFI)")
            aefi_rate = st.number_input("Tasa de AEFI por dosis", 0.0, 0.01, 0.0005)
            aefi_cost = st.number_input("Costo por caso AEFI ($)", 0.0, 1000.0, 100.0)
        
        st.markdown("### Beneficios y Análisis Temporal")
        col3, col4 = st.columns(2)
        
        with col3:
            incidence_rate = st.number_input("Incidencia base anual", 0.0, 1.0, 0.02)
            vaccine_efficacy = st.slider("Efectividad de vacuna (%)", 0.0, 100.0, 70.0) / 100.0
            time_horizon = st.number_input("Horizonte temporal (años)", 1, 20, 5)
            
        with col4:
            discount_rate = st.slider("Tasa de descuento anual (%)", 0.0, 20.0, 3.0) / 100.0
            medical_cost = st.number_input("Costo médico por caso ($)", 0.0, 1000.0, 200.0)
            productivity_loss = st.number_input("Pérdida productividad por caso ($)", 0.0, 1000.0, 150.0)
            waning_rate = st.slider("Tasa de atenuación anual (%)", 0.0, 50.0, 10.0) / 100.0
        
        # Crear objetos de parámetros
        cost_params = CostParams(
            population_target=population_target,
            coverage=coverage,
            doses_per_person=doses_per_person_ext,
            wastage_rate=wastage_rate,
            vaccine_price_per_dose=vaccine_price,
            freight_insurance_pct=freight_ins,
            syringe_price_per_dose=syringe_price,
            safety_box_price_per_100_doses=safety_box_price,
            admin_time_minutes_per_dose=admin_time,
            staff_cost_per_hour=staff_cost,
            cold_chain_variable_per_dose=cold_chain_var,
            transport_variable_per_dose=transport_var,
            waste_management_per_dose=waste_mgmt,
            clinic_overhead_fixed=clinic_overhead,
            training_fixed=training_fixed,
            communication_fixed=communication_fixed,
            monitoring_it_fixed=monitoring_fixed,
            other_fixed=other_fixed,
            aefi_rate_per_dose=aefi_rate,
            aefi_cost_per_case=aefi_cost
        )
        
        benefit_params = BenefitParams(
            baseline_incidence_per_person_year=incidence_rate,
            vaccine_effectiveness=vaccine_efficacy,
            time_horizon_years=time_horizon,
            discount_rate=discount_rate,
            medical_cost_per_case=medical_cost,
            productivity_loss_per_case=productivity_loss,
            waning_rate_per_year=waning_rate
        )
        
        # Ejecutar análisis extendido
        summary = summarize_cba(cost_params, benefit_params)
        
        # Mostrar resultados
        st.markdown("## Resultados del Análisis Extendido")
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("### Costos")
            st.markdown(f"**Personas vacunadas**: {summary['vaccinated_people']:.0f}")
            st.markdown(f"**Dosis administradas**: {summary['doses_administered']:.0f}")
            st.markdown(f"**Dosis adquiridas**: {summary['doses_procured']:.0f}")
            st.markdown(f"**Costo de vacunas**: ${summary['vaccine_cost']:,.2f}")
            st.markdown(f"**Costos variables**: ${summary['variable_costs']:,.2f}")
            st.markdown(f"**Costos fijos**: ${summary['fixed_costs']:,.2f}")
            st.markdown(f"**Costo esperado AEFI**: ${summary['aefi_expected_cost']:,.2f}")
            st.markdown(f"**Costo total**: ${summary['total_cost']:,.2f}")
            
        with col6:
            st.markdown("### Beneficios e Indicadores")
            st.markdown(f"**Casos evitados**: {summary['total_cases_averted']:.0f}")
            st.markdown(f"**Beneficios (VP)**: ${summary['pv_benefits']:,.2f}")
            st.markdown(f"**Beneficio neto (NPV)**: ${summary['npv']:,.2f}")
            st.markdown(f"**Ratio costo-beneficio (BCR)**: {summary['bcr']:.3f}")
            st.markdown(f"**Costo por caso evitado**: ${summary['cost_per_case_averted']:,.2f}")
            
            # Determinar si es rentable
            if summary['bcr'] > 1:
                st.success("La vacunación es costo-efectiva (beneficios > costos).")
            else:
                st.warning("La vacunación NO es costo-efectiva (costos > beneficios).")
        
        # Comparación con resultados del modelo epidemiológico
        st.markdown("### Comparación con Modelo Epidemiológico")
        st.markdown(f"**Casos evitados modelo**: {avoided_cases:.0f}")
        st.markdown(f"**Casos evitados análisis extendido**: {summary['total_cases_averted']:.0f}")
        st.markdown(f"**Personas vacunadas modelo**: {total_vaccinated:.0f}")
        st.markdown(f"**Personas vacunadas análisis extendido**: {summary['vaccinated_people']:.0f}")
        
        # Desglose detallado de costos
        with st.expander("Ver desglose detallado de costos"):
            data = {
                "Categoría": [
                    "Vacuna (incl. flete/seguro)",
                    "Jeringas y agujas", 
                    "Cajas de seguridad",
                    "Personal (administración)",
                    "Cadena de frío (variable)",
                    "Transporte (variable)",
                    "Manejo de residuos",
                    "AEFI esperados",
                    "Vacunatorios (fijos)",
                    "Capacitación",
                    "Comunicación/difusión",
                    "Monitoreo/IT",
                    "Otros costos fijos"
                ],
                "Porcentaje": [
                    summary['vaccine_cost'] / summary['total_cost'] * 100,
                    syringe_price * summary['doses_administered'] / summary['total_cost'] * 100,
                    safety_box_price * (summary['doses_administered']/100) / summary['total_cost'] * 100,
                    admin_time/60 * staff_cost * summary['doses_administered'] / summary['total_cost'] * 100,
                    cold_chain_var * summary['doses_administered'] / summary['total_cost'] * 100,
                    transport_var * summary['doses_administered'] / summary['total_cost'] * 100,
                    waste_mgmt * summary['doses_administered'] / summary['total_cost'] * 100,
                    summary['aefi_expected_cost'] / summary['total_cost'] * 100,
                    clinic_overhead / summary['total_cost'] * 100,
                    training_fixed / summary['total_cost'] * 100,
                    communication_fixed / summary['total_cost'] * 100,
                    monitoring_fixed / summary['total_cost'] * 100,
                    other_fixed / summary['total_cost'] * 100
                ]
            }
            
            # Crear un gráfico de barras
            st.bar_chart(data=data, x="Categoría", y="Porcentaje", use_container_width=True)
