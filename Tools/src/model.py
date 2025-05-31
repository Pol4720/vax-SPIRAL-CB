import numpy as np
from scipy.integrate import solve_ivp
import io
import matplotlib.pyplot as plt
import streamlit as st

class LeptospirosisModel:
    def __init__(self, params=None, initial_conditions=None):
        # Default parameters
        self.params = params or {
            'Λ': 4.635,
            'Π': 2,
            'β1': 0.00033,
            'β2': 0.0815,
            'β3': 0.0007,
            'γ': 0.089,
            'μ': 0.0009,
            'μv': 0.0029,
            'μb': 0.05,
            'θ': 0.092,
            'α': 0.04,
            'δ': 0.072,
            'ρ': 0.083,
            'σ': 0.064,
            'κ': 10000,
            'τ1': 0.06,
            'τ2': 0.2
        }
        # Default initial conditions
        self.initial_conditions = initial_conditions or [
            270,  # Sh
            20,   # Eh
            10,    # Ih
            0,    # Rh
            510,  # Sv
            10,   # Iv
            0,    # Rv
            100   # Bl
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

        compartments = {
            "Susceptibles Humanos (Sh)": (0, "Número de Humanos Susceptibles"),
            "Expuestos Humanos (Eh)": (1, "Número de Humanos Expuestos"),
            "Infectados Humanos (Ih)": (2, "Número de Humanos Infectados"),
            "Recuperados Humanos (Rh)": (3, "Número de Humanos Recuperados"),
            "Susceptibles Animales (Sv)": (4, "Número de Animales Susceptibles"),
            "Infectados Animales (Iv)": (5, "Número de Animales Infectados"),
            "Recuperados Animales (Rv)": (6, "Número de Animales Recuperados"),
            "Bacterias en el ambiente (Bl)": (7, "Cantidad de Bacterias en el Ambiente"),
        }

        with st.expander("Selecciona un compartimento para graficar"):
            selected = st.selectbox(
                "Compartimento",
                list(compartments.keys()),
                index=2  # Por defecto Infectados Humanos
            )

        idx, ylabel = compartments[selected]

        fig, ax = plt.subplots()
        ax.plot(self.solution.t, self.solution.y[idx], label=selected)
        ax.set_xlabel("Tiempo (días)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{selected} a lo largo del tiempo")
        ax.legend()
        ax.grid()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption=f"{selected} a lo largo del tiempo", use_container_width=True)
        plt.close(fig)

# Example usage:
# model = LeptospirosisModel()
# model.solve()
# model.plot()
