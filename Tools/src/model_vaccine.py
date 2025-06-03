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
            'τ2': 0.2,
            'ϕ': 0.01,  # Este será usado como nu_max
            'ε': 0.78,
            'ω': 1/180
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
            100,    # Bl
            0     # Vh (vaccinated humans)
        ]
        self.t_span = t_span
        self.t_eval = np.linspace(*t_span, t_span[1] - t_span[0] + 1)
        self.solution_vaccine = None
        self.solution_novaccine = None
        self.A = 0.4  # Amplitud estacional
        self.t_pico = 2  # Mes de máximo (febrero)
        self.K = 0.01 * sum(self.initial_conditions[:4])  # 1% de la población humana inicial
        # Calcular beta_max para normalización (máximo anual de beta_media)
        self.beta2_max = self._calcular_beta2_max()

    def _calcular_beta2_max(self):
        # Calcula el máximo valor estacional de beta2 en el año
        meses = np.arange(1, 13)
        beta2s = [self.beta_estacional(self.params['β2'], (m-1)*30) for m in meses]
        return max(beta2s)

    def beta_media_mensual(self, t_dia):
        # Devuelve el valor medio mensual de beta2 para el mes correspondiente
        mes = int((t_dia // 30) % 12) + 1
        dias_mes = np.arange((mes-1)*30, mes*30)
        beta2s = [self.beta_estacional(self.params['β2'], d) for d in dias_mes]
        return np.mean(beta2s)

    def nu_dinamica(self, Ih, t_dia):
        # Calcula la tasa de vacunación diaria dinámica según la fórmula propuesta
        nu_max = self.params['ϕ']
        K = self.K
        beta_media = self.beta_media_mensual(t_dia)
        beta_max = self.beta2_max
        frac_infectados = Ih / (Ih + K) if (Ih + K) > 0 else 0
        frac_beta = beta_media / beta_max if beta_max > 0 else 0
        return nu_max * frac_infectados * frac_beta

    def beta_estacional(self, beta_media, t_dia):
        """
        Calcula el valor estacional de beta dado el día t_dia.
        """
        mes = int((t_dia // 30) % 12) + 1  # Mes del año (1-12)
        return beta_media * (1 + self.A * np.cos(2 * np.pi * (mes - self.t_pico) / 12))

    def model(self, t, y, p):
        Sh, Eh, Ih, Rh, Sv, Iv, Rv, Bl, Vh = y

        # Calcular betas estacionales
        beta1 = self.beta_estacional(p['β1'], t)
        beta2 = self.beta_estacional(p['β2'], t)
        beta3 = self.beta_estacional(p['β3'], t)

        Λ, Π, _, _, _, γ, μ, μv, μb, θ, α, δ, ρ, σ, κ, τ1, τ2, ϕ, ε, ω = (
            p['Λ'], p['Π'], p['β1'], p['β2'], p['β3'], p['γ'], p['μ'], p['μv'], p['μb'],
            p['θ'], p['α'], p['δ'], p['ρ'], p['σ'], p['κ'], p['τ1'], p['τ2'],
            p['ϕ'], p['ε'], p['ω']
        )

        λh = beta2 * Bl / (κ + Bl) + beta1 * Iv

        # --- NUEVO: tasa de vacunación dinámica ---
        if p['ϕ'] > 0:
            phi_t = self.nu_dinamica(Ih, t)
        else:
            phi_t = 0

        # Human compartments
        dSh = Λ + γ * Rh + ω * Vh - λh * Sh - μ * Sh - phi_t * ε * Sh
        dEh = λh * Sh - (θ + μ) * Eh
        dIh = θ * Eh - (α + δ + μ) * Ih
        dRh = δ * Ih - (γ + μ) * Rh
        dVh = phi_t * ε * Sh - (ω + μ) * Vh  # vaccinated immune
        # Rodent compartments
        dSv = Π + ρ * Rv - (beta3 * Ih + μv) * Sv
        dIv = beta3 * Ih * Sv - (σ + μv) * Iv
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
            "Vaccinated (Vh)": 8,
            "Bacterias en ambiente (Bl)": 7,
            "Todos los compartimentos de vectores": "all_vectors",
            "Vaccination rate (personas/día)": "vaccination_rate"
        }

        with st.expander("Selecciona el compartimento a visualizar", expanded=True):
            selected = st.selectbox(
                "Compartimento humano/vector/ambiente",
                list(compartments.keys()),
                index=2
            )

        idx = compartments[selected]
        fig, ax = plt.subplots(figsize=(10, 6))

        if idx == "vaccination_rate":
            Ih = self.solution_vaccine.y[2]
            t_eval = self.t_eval
            vaccination_rate = []
            for t, ih in zip(t_eval, Ih):
                phi_t = self.nu_dinamica(ih, t)
                vaccination_rate.append(phi_t)
            ax.plot(t_eval, vaccination_rate, color='tab:green', label='Tasa de vacunación diaria', linewidth=2)
            ax.set_ylabel("Tasa de vacunación diaria (1/día)")
            ax.set_title("Tasa de vacunación diaria")
        elif idx == 7:  # Bacterias en ambiente (Bl)
            ax.plot(self.t_eval, self.solution_vaccine.y[7], label='Con Vacuna', color='tab:blue', linewidth=2)
            ax.plot(self.t_eval, self.solution_novaccine.y[7], '--', label='Sin Vacuna', color='tab:orange', linewidth=2)
            ax.set_ylabel("Bacterias en ambiente (Bl)")
            ax.set_title("Dinámica de bacterias en ambiente con y sin vacunación")
        elif idx == "all_vectors":
            # Compartimentos de vectores: Sv (4), Iv (5), Rv (6)
            colors = ['tab:blue', 'tab:orange', 'tab:green']
            labels = ['Susceptibles (Sv)', 'Infectados (Iv)', 'Recuperados (Rv)']
            for i, color, label in zip([4, 5, 6], colors, labels):
                ax.plot(self.t_eval, self.solution_vaccine.y[i], label=f'{label} (con vacuna)', color=color, linewidth=2)
                ax.plot(self.t_eval, self.solution_novaccine.y[i], '--', label=f'{label} (sin vacuna)', color=color, linewidth=2)
            ax.set_ylabel("Población de vectores")
            ax.set_title("Dinámica de compartimentos de vectores con y sin vacunación")
        else:
            ax.plot(self.t_eval, self.solution_vaccine.y[idx], label='Con Vacuna', linewidth=2)
            ax.plot(self.t_eval, self.solution_novaccine.y[idx], '--', label='Sin Vacuna', linewidth=2)
            ax.set_ylabel(selected)
            ax.set_title(f"Dinamica de {selected} con y sin vacunación")

        ax.set_xlabel("Días")
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
