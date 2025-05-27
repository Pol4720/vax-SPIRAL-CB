import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class LeptospirosisVaccineModel:
    def __init__(self, params=None, initial_conditions=None, t_span=(0, 365)):
        # Default parameters
        self.params = params or {
            'Λ': 50,       # Human recruitment rate
            'Π': 30,       # Rodent recruitment rate
            'β1': 0.02,    # Rodent-to-human transmission
            'β2': 0.01,    # Environment-to-human transmission
            'β3': 0.03,    # Human-to-rodent transmission
            'γ': 0.01,     # Immunity waning (recovery)
            'μ': 0.01,     # Human natural death rate
            'μv': 0.01,    # Rodent natural death rate
            'μb': 0.05,    # Bacteria death rate
            'θ': 0.2,      # Latent to infectious rate
            'α': 0.01,     # Disease-induced death
            'δ': 0.1,      # Recovery rate (infectious)
            'ρ': 0.05,     # Rodent immunity waning
            'σ': 0.1,      # Rodent recovery rate
            'κ': 10,       # Pathogen environment saturation constant
            'τ1': 0.5,     # Pathogen shedding from humans
            'τ2': 0.3,     # Pathogen shedding from rodents
            'ϕ': 0.03,     # Vaccination rate of susceptible humans
            'ε': 0.95,     # Vaccine efficacy
            'ω': 1/180     # Vaccine immunity waning (about 6 months)
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

    def plot_infectious_humans(self):
        if self.solution_vaccine is None:
            self.solve(with_vaccine=True)
        if self.solution_novaccine is None:
            self.solve(with_vaccine=False)
        plt.figure(figsize=(10, 6))
        plt.plot(self.t_eval, self.solution_vaccine.y[2], label='With Vaccine (Ih)', linewidth=2)
        plt.plot(self.t_eval, self.solution_novaccine.y[2], '--', label='No Vaccine (Ih)', linewidth=2)
        plt.xlabel("Days")
        plt.ylabel("Infectious Humans")
        plt.title("Effect of Vaccination on Leptospirosis Dynamics")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

# Example usage:
if __name__ == "__main__":
    model = LeptospirosisVaccineModel()
    model.solve(with_vaccine=True)
    model.solve(with_vaccine=False)
    model.plot_infectious_humans()
