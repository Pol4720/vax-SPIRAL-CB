import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.model import LeptospirosisModel
from src.model_vaccine import LeptospirosisVaccineModel

def initial_conditions_experiment(model_type="no_vaccine"):
    st.header("Initial Conditions Analysis")
    st.markdown(
        """
        Experiment with the initial populations of each compartment to see their influence on model dynamics.
        Adjust values below and visualize the results.
        """
    )

    # Select model type
    model_choice = st.radio("Model Type", ["No Vaccine", "With Vaccine"])
    if model_choice == "No Vaccine":
        model_cls = LeptospirosisModel
        ic_labels = ["Sh", "Eh", "Ih", "Rh", "Sv", "Iv", "Rv", "Bl"]
        default_ic = LeptospirosisModel().initial_conditions
    else:
        model_cls = LeptospirosisVaccineModel
        ic_labels = ["Sh", "Eh", "Ih", "Rh", "Sv", "Iv", "Rv", "Bl", "Vh"]
        default_ic = LeptospirosisVaccineModel().initial_conditions

    # Editable initial conditions
    st.subheader("Set Initial Conditions")
    cols = st.columns(len(ic_labels))
    ic_values = []
    for i, label in enumerate(ic_labels):
        val = cols[i].number_input(label, min_value=0.0, value=float(default_ic[i]), step=1.0)
        ic_values.append(val)

    # Simulation period
    t_end = st.slider("Simulation Period (days)", 30, 3650, 365)
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, t_end + 1)

    # Run simulation
    st.subheader("Simulate and Visualize")
    if st.button("Run Simulation"):
        model = model_cls()
        model.initial_conditions = ic_values
        if hasattr(model, "t_span"):
            model.t_span = t_span
            model.t_eval = t_eval
        sol = model.solve(t_span=t_span, num_points=len(t_eval)) if model_choice == "No Vaccine" else model.solve(with_vaccine=True)
        st.success("Simulation completed.")

        # Plot all compartments
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, label in enumerate(ic_labels):
            ax.plot(sol.t, sol.y[i], label=label)
        ax.set_xlabel("Days")
        ax.set_ylabel("Population")
        ax.set_title("Compartment Dynamics")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # State-space trajectory (Sh vs Ih)
        st.subheader("State-Space Trajectory (Sh vs Ih)")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        idx_sh = ic_labels.index("Sh")
        idx_ih = ic_labels.index("Ih")
        ax2.plot(sol.y[idx_sh], sol.y[idx_ih], lw=2)
        ax2.set_xlabel("Susceptible Humans (Sh)")
        ax2.set_ylabel("Infectious Humans (Ih)")
        ax2.set_title("Trajectory in State Space")
        ax2.grid(True)
        st.pyplot(fig2)

        # Sensitivity to initial conditions (one-at-a-time)
        st.subheader("Sensitivity to Initial Conditions")
        delta = 0.05  # 5% perturbation
        metrics = []
        for i, label in enumerate(ic_labels):
            perturbed_ic = ic_values.copy()
            perturbed_ic[i] *= (1 + delta)
            model_pert = model_cls()
            model_pert.initial_conditions = perturbed_ic
            if hasattr(model_pert, "t_span"):
                model_pert.t_span = t_span
                model_pert.t_eval = t_eval
            sol_pert = model_pert.solve(t_span=t_span, num_points=len(t_eval)) if model_choice == "No Vaccine" else model_pert.solve(with_vaccine=True)
            # Metric: max Ih difference
            max_diff = np.max(np.abs(sol_pert.y[idx_ih] - sol.y[idx_ih]))
            metrics.append((label, max_diff))

        df_metrics = {label: diff for label, diff in metrics}
        st.write("Max difference in Infectious Humans (Ih) after 5% increase in each compartment:")
        st.dataframe(df_metrics, use_container_width=True)

        # Bar plot of sensitivity
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.bar(df_metrics.keys(), df_metrics.values(), color='skyblue')
        ax3.set_ylabel("Max ΔIh")
        ax3.set_title("Sensitivity of Ih to Initial Conditions")
        st.pyplot(fig3)

        st.info("You can use these analyses to understand which compartments most influence the epidemic dynamics.")

# For direct Streamlit usage
if __name__ == "__main__":
    initial_conditions_experiment()
