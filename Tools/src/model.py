import numpy as np
from scipy.integrate import solve_ivp
import io
import matplotlib.pyplot as plt
import streamlit as st

class LeptospirosisModel:
    def __init__(self, params=None, initial_conditions=None):
        # Default parameters
        self.params = params or {
            'Λ': 5,
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
        if initial_conditions is None:
            # Calcular Bl* según la fórmula
            tau1 = self.params['τ1']
            tau2 = self.params['τ2']
            mu_b = self.params['μb']
            Ih_star = 10
            Iv_star = 10
            Bl_star = (tau1 * Ih_star + tau2 * Iv_star) / mu_b if mu_b > 0 else 0.0
            self.initial_conditions = [
                270,  # Sh
                20,   # Eh
                Ih_star,    # Ih
                0,    # Rh
                510,  # Sv
                Iv_star,   # Iv
                0,    # Rv
                Bl_star   # Bl calculado
            ]
        else:
            self.initial_conditions = initial_conditions
        self.solution = None
        self.A = 0.4  # Amplitud estacional
        self.t_pico = 2  # Mes de máximo (febrero)

    def beta_estacional(self, beta_media, t_dia):
        """
        Calcula el valor estacional de beta dado el día t_dia.
        """
        mes = int((t_dia // 30) % 12) + 1  # Mes del año (1-12)
        return beta_media * (1 + self.A * np.cos(2 * np.pi * (mes - self.t_pico) / 12))

    def model(self, t, y):
        Sh, Eh, Ih, Rh, Sv, Iv, Rv, Bl = y
        p = self.params

        # Calcular betas estacionales
        beta1 = self.beta_estacional(p['β1'], t)
        beta2 = self.beta_estacional(p['β2'], t)
        beta3 = self.beta_estacional(p['β3'], t)

        Λ, Π, _, _, _, γ, μ, μv, μb, θ, α, δ, ρ, σ, κ, τ1, τ2 = (
            p['Λ'], p['Π'], p['β1'], p['β2'], p['β3'], p['γ'], p['μ'], p['μv'], p['μb'],
            p['θ'], p['α'], p['δ'], p['ρ'], p['σ'], p['κ'], p['τ1'], p['τ2']
        )

        λh = beta2 * Bl / (κ + Bl) + beta1 * Iv

        dSh = Λ + γ * Rh - (λh + μ) * Sh
        dEh = λh * Sh - (θ + μ) * Eh
        dIh = θ * Eh - (α + δ + μ) * Ih
        dRh = δ * Ih - (γ + μ) * Rh
        dSv = Π + ρ * Rv - (beta3 * Ih + μv) * Sv
        dIv = beta3 * Ih * Sv - (σ + μv) * Iv
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

    def plot(self, selected=None):
        if self.solution is None:
            raise ValueError("No solution found. Run solve() first.")

        compartments = {
            "Susceptibles Humanos (Sh)": (0, "Número de Humanos Susceptibles"),
            "Expuestos Humanos (Eh)": (1, "Número de Humanos Expuestos"),
            "Infectados Humanos (Ih)": (2, "Número de Humanos Infectados"),
            "Recuperados Humanos (Rh)": (3, "Número de Humanos Recuperados"),
            "Vectores (Sv, Iv, Rv)": ("vectores", "Población de Vectores"),
            "Bacterias en el ambiente (Bl)": (7, "Cantidad de Bacterias en el Ambiente"),
        }

        # Solo mostrar selectbox si selected es None
        if selected is None:
            with st.expander("Selecciona un compartimento para graficar"):
                selected = st.selectbox(
                    "Compartimento",
                    list(compartments.keys()),
                    index=2  # Por defecto Infectados Humanos
                )

        idx, ylabel = compartments[selected]

        fig, ax = plt.subplots()
        if idx == "vectores":
            t = self.solution.t
            ax.plot(t, self.solution.y[4], label="Sv (Susceptibles)", color="tab:blue")
            ax.plot(t, self.solution.y[5], label="Iv (Infectados)", color="tab:red")
            ax.plot(t, self.solution.y[6], label="Rv (Recuperados)", color="tab:green")
            ax.set_ylabel("Población de Vectores")
            ax.set_title("Dinámica de los compartimentos de vectores")
            ax.legend()
        else:
            ax.plot(self.solution.t, self.solution.y[idx], label=selected)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{selected} a lo largo del tiempo")
            ax.legend()
        ax.set_xlabel("Tiempo (días)")
        ax.grid()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption=f"{selected} a lo largo del tiempo", use_container_width=True)
        plt.close(fig)

