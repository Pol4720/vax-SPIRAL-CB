import numpy as np
from scipy.integrate import solve_ivp
import streamlit as st
import matplotlib.pyplot as plt

class LeptospirosisVaccineModel:
    def __init__(self, params=None, initial_conditions=None, t_span=(0, 365)):
        # Default parameters
        self.param_comments = {
            'Λ': "Human recruitment rate",
            'Π': "Rodent recruitment rate",
            'β1': "Rodent-to-human transmission",
            'β2': "Environment-to-human transmission",
            'β3': "Human-to-rodent transmission",
            'γ': "Immunity waning (recovery)",
            'μ': "Human natural death rate",
            'μv': "Rodent natural death rate",
            'μb': "Bacteria death rate",
            'θ': "Latent to infectious rate",
            'α': "Disease-induced death",
            'δ': "Recovery rate (infectious)",
            'ρ': "Rodent immunity waning",
            'σ': "Rodent recovery rate",
            'κ': "Pathogen environment saturation constant",
            'τ1': "Pathogen shedding from humans",
            'τ2': "Pathogen shedding from rodents",
            'ϕ': "Vaccination rate of susceptible humans",
            'ε': "Vaccine efficacy",
            'ω': "Vaccine immunity waning (about 6 months)"
        }
        self.params = params or {
            'Λ': 50,
            'Π': 30,
            'β1': 0.02,
            'β2': 0.01,
            'β3': 0.03,
            'γ': 0.01,
            'μ': 0.01,
            'μv': 0.01,
            'μb': 0.05,
            'θ': 0.2,
            'α': 0.01,
            'δ': 0.1,
            'ρ': 0.05,
            'σ': 0.1,
            'κ': 10,
            'τ1': 0.5,
            'τ2': 0.3,
            'ϕ': 0.2,
            'ε': 0.95,
            'ω': 1/180
        }
        # Default initial conditions
        self.initial_conditions = initial_conditions or [
            500,  # Sh
            10,   # Eh
            5,    # Ih
            0,    # Rh
            400,  # Sv
            10,   # Iv
            0,    # Rv
            0,    # Bl
            0     # Vh (vaccinated humans)
        ]
        self.t_span = t_span
        self.t_eval = np.linspace(*t_span, t_span[1] - t_span[0] + 1)
        self.solution_vaccine = None
        self.solution_novaccine = None

    def model(self, t, y, p):
        Sh, Eh, Ih, Rh, Sv, Iv, Rv, Bl, Vh = y
        Λ, Π, β1, β2, β3, γ, μ, μv, μb, θ, α, δ, ρ, σ, κ, τ1, τ2, ϕ, ε, ω = (
            p['Λ'], p['Π'], p['β1'], p['β2'], p['β3'], p['γ'], p['μ'], p['μv'], p['μb'],
            p['θ'], p['α'], p['δ'], p['ρ'], p['σ'], p['κ'], p['τ1'], p['τ2'],
            p['ϕ'], p['ε'], p['ω']
        )

        λh = β2 * Bl / (κ + Bl) + β1 * Iv

        # Human compartments
        dSh = Λ + γ * Rh + ω * Vh - λh * Sh - μ * Sh - ϕ * Sh
        dEh = λh * Sh - (θ + μ) * Eh
        dIh = θ * Eh - (α + δ + μ) * Ih
        dRh = δ * Ih - (γ + μ) * Rh
        dVh = ϕ * ε * Sh - (ω + μ) * Vh  # vaccinated immune
        # Rodent compartments
        dSv = Π + ρ * Rv - (β3 * Ih + μv) * Sv
        dIv = β3 * Ih * Sv - (σ + μv) * Iv
        dRv = σ * Iv - (ρ + μv) * Rv
        # Environment
        dBl = τ1 * Ih + τ2 * Iv - μb * Bl

        return [dSh, dEh, dIh, dRh, dSv, dIv, dRv, dBl, dVh]

    def solve(self, with_vaccine=True):
        params = self.params.copy()
        if not with_vaccine:
            params['ϕ'] = 0
        sol = solve_ivp(
            lambda t, y: self.model(t, y, params),
            self.t_span, self.initial_conditions, t_eval=self.t_eval, method='RK45'
        )
        if with_vaccine:
            self.solution_vaccine = sol
        else:
            self.solution_novaccine = sol
        return sol

    def plot_compartment(self):
        if self.solution_vaccine is None:
            self.solve(with_vaccine=True)
        if self.solution_novaccine is None:
            self.solve(with_vaccine=False)

        compartments = {
            "Susceptible (Sh)": 0,
            "Exposed (Eh)": 1,
            "Infectious (Ih)": 2,
            "Recovered (Rh)": 3,
            "Vaccinated (Vh)": 8
        }

        with st.expander("Selecciona el compartimento a visualizar", expanded=True):
            selected = st.selectbox(
                "Compartimento humano",
                list(compartments.keys()),
                index=2
            )

        idx = compartments[selected]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.t_eval, self.solution_vaccine.y[idx], label='Con Vacuna', linewidth=2)
        ax.plot(self.t_eval, self.solution_novaccine.y[idx], '--', label='Sin Vacuna', linewidth=2)
        ax.set_xlabel("Días")
        ax.set_ylabel(selected)
        ax.set_title(f"Dinamica de {selected} con y sin vacunación")
        ax.legend()
        ax.grid()
        fig.tight_layout()
        st.pyplot(fig)

# Example usage:
if __name__ == "__main__":
    model = LeptospirosisVaccineModel()
    model.solve(with_vaccine=True)
    model.solve(with_vaccine=False)
    model.plot_infectious_humans()
