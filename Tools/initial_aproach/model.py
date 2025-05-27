import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo (basados en el artículo y añadidos de vacuna)
params = {
    'Λ': 0.27,         # Tasa de reclutamiento humanos
    'Π': 2,            # Tasa de reclutamiento roedores
    'β1': 0.00033,     # Transmisión roedores a humanos
    'β2': 0.0815,      # Transmisión ambiente a humanos
    'β3': 0.0007,      # Transmisión humanos a roedores
    'μ': 0.0009,       # Tasa de mortalidad humanos
    'μ_v': 0.0029,     # Tasa de mortalidad roedores
    'μ_b': 0.05,       # Tasa de muerte bacterias
    'θ': 0.092,        # Tasa de latencia a infeccioso
    'α': 0.04,         # Tasa de muerte por enfermedad
    'δ': 0.072,        # Tasa de recuperación humanos
    'σ': 0.064,        # Tasa de recuperación roedores
    'γ': 0.089,        # Tasa de pérdida inmunidad humanos
    'ρ': 0.083,        # Tasa de pérdida inmunidad roedores
    'τ1': 0.06,        # Tasa de liberación bacterias por humanos
    'τ2': 0.2,         # Tasa de liberación bacterias por roedores
    'κ': 10000,        # Concentración bacterias en ambiente
    'ν': 0.01,         # Tasa de vacunación (nuevo)
    'ε': 0.8,          # Efectividad vacuna (nuevo)
    'γ_v': 1/365       # Pérdida inmunidad vacuna (nuevo)
}

# Condiciones iniciales (con vacunados)
initial_conditions = [270, 20, 10, 0, 510, 10, 0, 100, 0]  # [S_h, E_h, I_h, R_h, S_v, I_v, R_v, B_l, V_h]

# Sistema de ecuaciones diferenciales
def model(t, y, params):
    S_h, E_h, I_h, R_h, S_v, I_v, R_v, B_l, V_h = y
    
    λ_h = (params['β1'] * I_v) + (params['β2'] * B_l) / (params['κ'] + B_l)
    λ_h_vac = (1 - params['ε']) * λ_h  # Fuerza de infección reducida
    
    dS_h = params['Λ'] + params['γ'] * R_h + params['γ_v'] * V_h - (λ_h + params['μ'] + params['ν']) * S_h
    dE_h = λ_h * S_h + λ_h_vac * V_h - (params['θ'] + params['μ']) * E_h
    dI_h = params['θ'] * E_h - (params['α'] + params['δ'] + params['μ']) * I_h
    dR_h = params['δ'] * I_h - (params['γ'] + params['μ']) * R_h
    
    dS_v = params['Π'] + params['ρ'] * R_v - (params['β3'] * I_h + params['μ_v']) * S_v
    dI_v = params['β3'] * I_h * S_v - (params['σ'] + params['μ_v']) * I_v
    dR_v = params['σ'] * I_v - (params['ρ'] + params['μ_v']) * R_v
    
    dB_l = params['τ1'] * I_h + params['τ2'] * I_v - params['μ_b'] * B_l
    
    dV_h = params['ν'] * S_h - (params['γ_v'] + params['μ']) * V_h
    
    return [dS_h, dE_h, dI_h, dR_h, dS_v, dI_v, dR_v, dB_l, dV_h]

# Método de Runge-Kutta de cuarto orden
def runge_kutta(f, t0, y0, t_end, dt, params):
    t = np.arange(t0, t_end + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = np.array(f(t[i-1], y[i-1], params))
        k2 = np.array(f(t[i-1] + dt/2, y[i-1] + dt/2 * k1, params))
        k3 = np.array(f(t[i-1] + dt/2, y[i-1] + dt/2 * k2, params))
        k4 = np.array(f(t[i-1] + dt, y[i-1] + dt * k3, params))
        y[i] = y[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, y

# Simulación
t_end = 350  # Días
dt = 1
t, y = runge_kutta(model, 0, initial_conditions, t_end, dt, params)

# Visualización
plt.figure(figsize=(12, 8))
plt.plot(t, y[:, 2], label='Infectados humanos (I_h)')
plt.plot(t, y[:, 8], label='Vacunados (V_h)')
plt.xlabel('Tiempo (días)')
plt.ylabel('Población')
plt.legend()
plt.title('Impacto de la Vacunación en la Dinámica de Leptospirosis')
plt.show()