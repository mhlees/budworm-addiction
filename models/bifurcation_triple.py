
# models/bifurcation_triple.py

"""
Bifurcation Analysis for Budworm Model

Generates a bifurcation diagram by sweeping the bird predation parameter (B)
and examining steady-state budworm populations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from budworm_model import budworm_dynamics


def run_bifurcation_analysis(B_values, N0=1.0, r=1.5, K=10.0, a=1.0, t_end=200, t_sample_start=150):
    """
    Simulates the budworm model for a range of B values and collects population values at late time.

    Parameters:
    - B_values (np.ndarray): Range of bird predation values
    - N0 (float): Initial population
    - r, K, a: Model parameters
    - t_end (float): Total simulation time
    - t_sample_start (float): Time after which to sample steady states

    Returns:
    - results (list of tuples): List of (B, steady_state_N) values
    """
    t = np.linspace(0, t_end, 2000)
    results = []

    for B in B_values:
        sol = odeint(budworm_dynamics, N0, t, args=(r, K, B, a)).flatten()
        steady_states = sol[t > t_sample_start]  # Take only values after transients
        for N in steady_states:
            results.append((B, N))

    return results


def plot_bifurcation(results):
    """
    Plots the bifurcation diagram.

    Parameters:
    - results (list of tuples): (B, N) points
    """
    B_vals, N_vals = zip(*results)
    plt.figure(figsize=(10, 5))
    plt.plot(B_vals, N_vals, 'k.', markersize=0.5)
    plt.xlabel("Bird predation strength (B)")
    plt.ylabel("Budworm population (N)")
    plt.title("Bifurcation Diagram of Budworm Model")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    B_range = np.linspace(0.1, 5.0, 500)
    bifurcation_data = run_bifurcation_analysis(B_range)
    plot_bifurcation(bifurcation_data)
