import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class LeptospirosisModel:
    def __init__(self, params=None, initial_conditions=None):
        # Default parameters
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
            0     # Bl
        ]
        self.solution = None

    def model(self, t, y):
        Sh, Eh, Ih, Rh, Sv, Iv, Rv, Bl = y
        p = self.params
        Λ, Π, β1, β2, β3, γ, μ, μv, μb, θ, α, δ, ρ, σ, κ, τ1, τ2 = (
            p['Λ'], p['Π'], p['β1'], p['β2'], p['β3'], p['γ'], p['μ'], p['μv'], p['μb'],
            p['θ'], p['α'], p['δ'], p['ρ'], p['σ'], p['κ'], p['τ1'], p['τ2']
        )

        λh = β2 * Bl / (κ + Bl) + β1 * Iv

        dSh = Λ + γ * Rh - (λh + μ) * Sh
        dEh = λh * Sh - (θ + μ) * Eh
        dIh = θ * Eh - (α + δ + μ) * Ih
        dRh = δ * Ih - (γ + μ) * Rh
        dSv = Π + ρ * Rv - (β3 * Ih + μv) * Sv
        dIv = β3 * Ih * Sv - (σ + μv) * Iv
        dRv = σ * Iv - (ρ + μv) * Rv
        dBl = τ1 * Ih + τ2 * Iv - μb * Bl

        return [dSh, dEh, dIh, dRh, dSv, dIv, dRv, dBl]

    def solve(self, t_span=(0, 365), num_points=366):
        t_eval = np.linspace(*t_span, num_points)
        self.solution = solve_ivp(
            fun=self.model,
            t_span=t_span,
            y0=self.initial_conditions,
            t_eval=t_eval,
            method='RK45'
        )
        return self.solution

    def plot(self):
        if self.solution is None:
            raise ValueError("No solution found. Run solve() first.")
        compartments = ['Sh', 'Eh', 'Ih', 'Rh', 'Sv', 'Iv', 'Rv', 'Bl']
        for i, name in enumerate(compartments):
            plt.plot(self.solution.t, self.solution.y[i], label=name)
        plt.xlabel("Time (days)")
        plt.ylabel("Population")
        plt.title("Leptospirosis Model Dynamics")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

# Example usage:
# model = LeptospirosisModel()
# model.solve()
# model.plot()
