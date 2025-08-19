from dataclasses import dataclass
from typing import Optional, Dict

def _df(rate: float, year: int) -> float:
    # Factor de descuento para el año 'year' (1-indexed: año 1, 2, ...)
    return 1.0 / ((1.0 + rate) ** year)

@dataclass
class CostParams:
    # Población y esquema
    population_target: int                 # población objetivo
    coverage: float                        # 0-1
    doses_per_person: float                # p.ej., 2
    wastage_rate: float = 0.05             # merma 0-1 sobre dosis adquiridas

    # Precios directos por dosis
    vaccine_price_per_dose: float = 10.0   # USD/dosis
    freight_insurance_pct: float = 0.05    # % sobre precio de vacuna
    syringe_price_per_dose: float = 0.05   # jeringa/aguja por dosis administrada
    safety_box_price_per_100_doses: float = 5.0  # cajas de seguridad (por 100 dosis)

    # Costos variables operativos por dosis administrada
    admin_time_minutes_per_dose: float = 5.0
    staff_cost_per_hour: float = 8.0
    cold_chain_variable_per_dose: float = 0.03
    transport_variable_per_dose: float = 0.02
    waste_management_per_dose: float = 0.01

    # Costos fijos de campaña/servicio
    clinic_overhead_fixed: float = 2000.0  # alquiler/servicios/utilities
    training_fixed: float = 1000.0
    communication_fixed: float = 1500.0
    monitoring_it_fixed: float = 800.0
    other_fixed: float = 0.0

    # Eventos adversos (AEFI)
    aefi_rate_per_dose: float = 0.0005
    aefi_cost_per_case: float = 100.0

@dataclass
class BenefitParams:
    baseline_incidence_per_person_year: float  # incidencia anual en no vacunados
    vaccine_effectiveness: float               # 0-1
    time_horizon_years: int                    # años de análisis
    discount_rate: float = 0.03                # tasa de descuento anual

    # Costos por caso (beneficios evitados)
    medical_cost_per_case: float = 200.0
    productivity_loss_per_case: float = 150.0

    # Opcional: atenuación de efectividad (waning) por año (0-1). Si None, no hay atenuación.
    waning_rate_per_year: Optional[float] = None

def compute_costs(costs: CostParams) -> Dict[str, float]:
    vaccinated_people = costs.population_target * costs.coverage
    doses_administered = vaccinated_people * costs.doses_per_person
    # Dosis a adquirir considerando merma
    doses_procured = doses_administered / (1.0 - costs.wastage_rate)

    # Costos variables por dosis
    vaccine_unit_cif = costs.vaccine_price_per_dose * (1.0 + costs.freight_insurance_pct)
    vaccine_cost = doses_procured * vaccine_unit_cif

    syringe_cost = doses_administered * costs.syringe_price_per_dose
    safety_box_cost = (doses_administered / 100.0) * costs.safety_box_price_per_100_doses

    admin_time_hours = (costs.admin_time_minutes_per_dose / 60.0) * doses_administered
    staff_cost = admin_time_hours * costs.staff_cost_per_hour

    cold_chain_var = doses_administered * costs.cold_chain_variable_per_dose
    transport_var = doses_administered * costs.transport_variable_per_dose
    waste_mgmt = doses_administered * costs.waste_management_per_dose

    aefi_expected = doses_administered * costs.aefi_rate_per_dose * costs.aefi_cost_per_case

    variable_costs = (
        vaccine_cost + syringe_cost + safety_box_cost +
        staff_cost + cold_chain_var + transport_var + waste_mgmt + aefi_expected
    )

    fixed_costs = (
        costs.clinic_overhead_fixed + costs.training_fixed +
        costs.communication_fixed + costs.monitoring_it_fixed + costs.other_fixed
    )

    total_cost = variable_costs + fixed_costs

    return {
        "vaccinated_people": vaccinated_people,
        "doses_administered": doses_administered,
        "doses_procured": doses_procured,
        "vaccine_cost": vaccine_cost,
        "variable_costs": variable_costs,
        "fixed_costs": fixed_costs,
        "aefi_expected_cost": aefi_expected,
        "total_cost": total_cost,
    }

def compute_benefits(costs: CostParams, benefits: BenefitParams) -> Dict[str, float]:
    vaccinated_people = costs.population_target * costs.coverage

    medical_plus_prod = benefits.medical_cost_per_case + benefits.productivity_loss_per_case

    pv_benefits = 0.0
    total_cases_averted = 0.0

    for y in range(1, benefits.time_horizon_years + 1):
        if benefits.waning_rate_per_year is None:
            ve_y = benefits.vaccine_effectiveness
        else:
            # Efectividad decae geométricamente cada año
            ve_y = benefits.vaccine_effectiveness * ((1.0 - benefits.waning_rate_per_year) ** (y - 1))

        cases_averted_y = vaccinated_people * benefits.baseline_incidence_per_person_year * ve_y
        benefit_y = cases_averted_y * medical_plus_prod

        pv = benefit_y * _df(benefits.discount_rate, y)
        pv_benefits += pv
        total_cases_averted += cases_averted_y

    return {
        "pv_benefits": pv_benefits,
        "total_cases_averted": total_cases_averted
    }

def summarize_cba(costs: CostParams, benefits: BenefitParams) -> Dict[str, float]:
    c = compute_costs(costs)
    b = compute_benefits(costs, benefits)

    total_cost = c["total_cost"]
    pv_benefits = b["pv_benefits"]

    npv = pv_benefits - total_cost
    bcr = (pv_benefits / total_cost) if total_cost > 0 else float("inf")
    cost_per_case_averted = (total_cost / b["total_cases_averted"]) if b["total_cases_averted"] > 0 else float("inf")

    return {
        # Costos
        **c,
        # Beneficios
        **b,
        # Indicadores
        "npv": npv,
        "bcr": bcr,
        "cost_per_case_averted": cost_per_case_averted
    }

if __name__ == "__main__":
    # Ejemplo mínimo (valores ilustrativos)
    cost_params = CostParams(
        population_target=100000,
        coverage=0.6,
        doses_per_person=2,
        wastage_rate=0.07,
        vaccine_price_per_dose=8.0,
        freight_insurance_pct=0.06,
        syringe_price_per_dose=0.06,
        safety_box_price_per_100_doses=6.0,
        admin_time_minutes_per_dose=6.0,
        staff_cost_per_hour=10.0,
        cold_chain_variable_per_dose=0.04,
        transport_variable_per_dose=0.03,
        waste_management_per_dose=0.015,
        clinic_overhead_fixed=3000.0,
        training_fixed=1500.0,
        communication_fixed=2000.0,
        monitoring_it_fixed=1200.0,
        other_fixed=500.0,
        aefi_rate_per_dose=0.0007,
        aefi_cost_per_case=120.0,
    )

    benefit_params = BenefitParams(
        baseline_incidence_per_person_year=0.02,
        vaccine_effectiveness=0.7,
        time_horizon_years=5,
        discount_rate=0.03,
        medical_cost_per_case=250.0,
        productivity_loss_per_case=200.0,
        waning_rate_per_year=0.1
    )

    summary = summarize_cba(cost_params, benefit_params)
    # Salida resumida
    print({
        "total_cost": round(summary["total_cost"], 2),
        "pv_benefits": round(summary["pv_benefits"], 2),
        "npv": round(summary["npv"], 2),
        "bcr": round(summary["bcr"], 3),
        "cost_per_case_averted": round(summary["cost_per_case_averted"], 2)
    })
